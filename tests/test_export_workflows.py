import importlib.util
import json
from pathlib import Path
from types import ModuleType

SCRIPT = Path(__file__).parents[1] / "scripts" / "export_workflows.py"


def _script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("export_workflows", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_committed_workflows_match_their_builders() -> None:
    script = _script()
    # Through JSON, as the files hold them: a tuple in a graph reads back as a list.
    built = {
        name: json.loads(json.dumps(graph))
        for name, graph in script.workflows().items()
    }
    committed = {
        path.name: json.loads(path.read_text("utf-8"))
        for path in script.FOLDER.glob("*.json")
    }

    assert committed == built, "Run scripts/export_workflows.py to update workflows/."
