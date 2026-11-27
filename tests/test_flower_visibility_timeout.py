"""Flower must restore unacked messages no sooner than the workers (#54)."""

import ast
import importlib.util
import os
from pathlib import Path

import pytest

# In the containers only tests/ is mounted next to the app: QI_REPO_ROOT points
# at a full checkout there.
_ROOT = Path(os.environ.get("QI_REPO_ROOT", Path(__file__).resolve().parent.parent))
_FLOWER_APP = _ROOT / "flower" / "flower_app.py"


def _flower_visibility_timeout():
    """Read the value from flower_app.py without importing it (Celery 4 app)."""
    tree = ast.parse(_FLOWER_APP.read_text())
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Attribute)
            and node.targets[0].attr == "broker_transport_options"
        ):
            return ast.literal_eval(node.value)["visibility_timeout"]
    raise AssertionError("broker_transport_options not set in flower_app.py")


@pytest.mark.skipif(not _FLOWER_APP.exists(), reason="flower/ not mounted here")
def test_flower_uses_the_workers_visibility_timeout(monkeypatch):
    monkeypatch.setenv("CELERY_WORKER_CONCURRENCY", "4")
    spec = importlib.util.spec_from_file_location(
        "celeryconfig_for_flower_test", _ROOT / "workers" / "celeryconfig.py"
    )
    celeryconfig = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(celeryconfig)

    workers_value = celeryconfig.broker_transport_options["visibility_timeout"]
    assert _flower_visibility_timeout() == workers_value
