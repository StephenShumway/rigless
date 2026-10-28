import importlib.util
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


class _CutShort:
    """A response promising ten bytes that ends after four, like a dropped link."""

    status = 200
    headers = {"Content-Length": "10"}

    def __init__(self) -> None:
        self._chunks = [b"abcd"]

    def read(self, size: int) -> bytes:
        return self._chunks.pop() if self._chunks else b""

    def __enter__(self) -> "_CutShort":
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
    monkeypatch.setattr(urllib.request, "urlopen", lambda *args, **kwargs: _CutShort())
    dest = tmp_path / "model.safetensors"

    with pytest.raises(OSError):
        _script().download("https://example.com/model.safetensors", dest)

    assert not dest.exists()
