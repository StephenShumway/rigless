import importlib.util
import json
import urllib.request
from pathlib import Path
from types import ModuleType, TracebackType

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "download_models.py"


def _script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("download_models", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Response:
    """A download answering with body, though it may promise a different length."""

    status = 200

    def __init__(self, body: bytes, length: int | None = None) -> None:
        self.headers = {"Content-Length": str(len(body) if length is None else length)}
        self._chunks = [body] if body else []

    def read(self, size: int) -> bytes:
        return self._chunks.pop() if self._chunks else b""

    def __enter__(self) -> "_Response":
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        trace: TracebackType | None,
    ) -> None:
        pass


def test_a_download_cut_short_is_not_kept_as_the_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda *args, **kwargs: _Response(b"abcd", 10)
    )
    dest = tmp_path / "model.safetensors"

    with pytest.raises(OSError):
        _script().download("https://example.com/model.safetensors", dest)

    assert not dest.exists()


def test_progress_counts_the_models_already_on_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _script()
    lora = "https://example.com/lora.safetensors"
    monkeypatch.setattr(
        script, "MODELS", [(lora, "loras"), ("https://example.com/vae.bin", "vae")]
    )
    (tmp_path / "loras").mkdir()
    (tmp_path / "loras" / "lora.safetensors").write_bytes(b"abc")
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda *args, **kwargs: _Response(b"wxyz")
    )
    progress = tmp_path / "progress.json"

    script.fetch_all(tmp_path, progress)

    assert json.loads(progress.read_text()) == {
        "stage": "models",
        "done": 7,
        "total": 7,
    }
