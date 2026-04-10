"""Tests for runtime_runner.py.

These tests stub ``comfy_runtime.execute_node`` in-process instead of
actually running a real workflow — unit-level, fast, does not require the
benchmark venvs.
"""

import json
import sys
import types
from pathlib import Path

import pytest

from benchmarks.e2e.runners import runtime_runner


@pytest.fixture
def fake_comfy_runtime(monkeypatch):
    """Insert a fake ``comfy_runtime`` module into sys.modules."""
    calls: list[tuple[str, dict]] = []

    def fake_execute_node(class_type, **kwargs):
        calls.append((class_type, kwargs))
        return ("fake-output",)

    def fake_configure(**kwargs):
        pass

    def fake_list_nodes():
        return ["CheckpointLoaderSimple", "KSampler"]

    def fake_load_nodes_from_path(path):
        return []

    fake_mod = types.SimpleNamespace(
        execute_node=fake_execute_node,
        configure=fake_configure,
        list_nodes=fake_list_nodes,
        load_nodes_from_path=fake_load_nodes_from_path,
        __version__="0.3.1-fake",
    )
    monkeypatch.setitem(sys.modules, "comfy_runtime", fake_mod)
    return fake_mod, calls


def _write_stages_yaml(path: Path, mapping: dict) -> None:
    import yaml
    path.write_text(yaml.safe_dump({"stages": mapping}))


def test_run_writes_valid_json(tmp_path, fake_comfy_runtime):
    fake_mod, calls = fake_comfy_runtime

    # Dummy workflow main.py
    workflow_dir = tmp_path / "dummy_workflow"
    workflow_dir.mkdir()
    (workflow_dir / "main.py").write_text(
        "import random\n"
        "SEED = random.randint(0, 2**63)\n"
        "def main():\n"
        "    import comfy_runtime\n"
        "    comfy_runtime.execute_node('CheckpointLoaderSimple', ckpt_name='fake')\n"
        "    comfy_runtime.execute_node('KSampler', seed=SEED, steps=1)\n"
    )

    # stages.yaml in a sibling metadata dir
    stages_path = tmp_path / "stages.yaml"
    _write_stages_yaml(stages_path, {"model_load": ["CheckpointLoaderSimple"], "sample": ["KSampler"]})

    out_json = tmp_path / "out.json"

    runtime_runner.run(
        workflow_name="dummy",
        workflow_main_py=workflow_dir / "main.py",
        stages_yaml=stages_path,
        run_idx=1,
        out_path=out_json,
    )

    assert out_json.exists()
    data = json.loads(out_json.read_text())
    assert data["workflow"] == "dummy"
    assert data["side"] == "runtime"
    assert data["run_idx"] == 1
    assert data["status"] == "ok"
    assert len(data["nodes"]) == 2
    assert data["nodes"][0]["class_type"] == "CheckpointLoaderSimple"
    assert data["nodes"][1]["class_type"] == "KSampler"
    assert {s["name"] for s in data["stages"]} == {"model_load", "sample"}
    assert data["total_ns"] > 0


def test_run_deterministic_seed_overwrite(tmp_path, fake_comfy_runtime):
    fake_mod, calls = fake_comfy_runtime
    workflow_dir = tmp_path / "wf"
    workflow_dir.mkdir()
    (workflow_dir / "main.py").write_text(
        "import random\n"
        "SEED = random.randint(0, 2**63)\n"
        "def main():\n"
        "    import comfy_runtime\n"
        "    comfy_runtime.execute_node('KSampler', seed=SEED)\n"
    )
    stages_path = tmp_path / "stages.yaml"
    _write_stages_yaml(stages_path, {"sample": ["KSampler"]})
    out_json = tmp_path / "out.json"

    runtime_runner.run(
        workflow_name="wf",
        workflow_main_py=workflow_dir / "main.py",
        stages_yaml=stages_path,
        run_idx=1,
        out_path=out_json,
    )
    # The runner must overwrite SEED to 42 before calling main().
    assert calls[-1] == ("KSampler", {"seed": 42})


def test_run_captures_failure(tmp_path, fake_comfy_runtime):
    fake_mod, calls = fake_comfy_runtime
    workflow_dir = tmp_path / "wf"
    workflow_dir.mkdir()
    (workflow_dir / "main.py").write_text(
        "def main():\n"
        "    raise RuntimeError('boom from workflow')\n"
    )
    stages_path = tmp_path / "stages.yaml"
    _write_stages_yaml(stages_path, {"sample": ["KSampler"]})
    out_json = tmp_path / "out.json"

    runtime_runner.run(
        workflow_name="wf",
        workflow_main_py=workflow_dir / "main.py",
        stages_yaml=stages_path,
        run_idx=1,
        out_path=out_json,
    )
    data = json.loads(out_json.read_text())
    assert data["status"] == "failed"
    assert "boom from workflow" in data["error"]
