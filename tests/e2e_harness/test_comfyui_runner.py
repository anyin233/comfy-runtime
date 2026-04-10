"""Tests for comfyui_runner.py.

The comfyui runner imports upstream ComfyUI from disk. For unit tests we
stub the upstream modules via sys.modules injection.
"""

import json
import sys
import types
from pathlib import Path

import pytest
import yaml


@pytest.fixture
def stub_comfyui(monkeypatch):
    """Insert minimal stubs for ``nodes``, ``execution``, ``folder_paths``."""

    # --- nodes stub ---
    class _FakeLoader:
        @staticmethod
        def execute(**kwargs):
            return ("fake-model",)

    class _FakeSampler:
        @staticmethod
        def execute(**kwargs):
            return ("fake-latent",)

    nodes_mod = types.SimpleNamespace(
        NODE_CLASS_MAPPINGS={
            "CheckpointLoaderSimple": _FakeLoader,
            "KSampler": _FakeSampler,
        },
        load_custom_node=lambda p: [],
    )

    # --- folder_paths stub ---
    folder_paths_mod = types.SimpleNamespace(
        add_model_folder_path=lambda *a, **k: None,
        get_folder_paths=lambda cat: [],
    )

    # --- execution stub: a PromptExecutor that just calls each node in order ---
    class _StubPromptExecutor:
        def __init__(self, server, cache_type=False, cache_args=None):
            self.server = server

        def reset(self):
            pass

        def execute(self, prompt, prompt_id, extra_data=None, execute_outputs=None):
            # Run each node in numeric id order, call the patched execute().
            for node_id in sorted(prompt, key=int):
                node = prompt[node_id]
                cls = nodes_mod.NODE_CLASS_MAPPINGS[node["class_type"]]
                execution_mod.execute(node_id, node["class_type"], cls, node.get("inputs", {}))

    def _default_execute(node_id, class_type, cls, inputs):
        return cls.execute(**inputs)

    execution_mod = types.SimpleNamespace(
        PromptExecutor=_StubPromptExecutor,
        execute=_default_execute,
    )

    monkeypatch.setitem(sys.modules, "nodes", nodes_mod)
    monkeypatch.setitem(sys.modules, "execution", execution_mod)
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths_mod)

    return nodes_mod, execution_mod, folder_paths_mod


def test_runner_writes_json(tmp_path, stub_comfyui):
    from benchmarks.e2e.runners import comfyui_runner

    prompt = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "fake"}},
        "2": {"class_type": "KSampler", "inputs": {"seed": 42, "steps": 1}},
    }
    prompt_path = tmp_path / "prompt.json"
    prompt_path.write_text(json.dumps(prompt))

    stages_path = tmp_path / "stages.yaml"
    stages_path.write_text(yaml.safe_dump({"stages": {
        "model_load": ["CheckpointLoaderSimple"],
        "sample": ["KSampler"],
    }}))

    out_path = tmp_path / "out.json"
    comfyui_runner.run(
        workflow_name="dummy",
        prompt_json=prompt_path,
        stages_yaml=stages_path,
        workflow_models_dir=tmp_path / "models",
        workflow_nodes_dir=None,
        run_idx=1,
        out_path=out_path,
    )

    data = json.loads(out_path.read_text())
    assert data["side"] == "comfyui"
    assert data["status"] == "ok"
    assert len(data["nodes"]) == 2
    names = {n["class_type"] for n in data["nodes"]}
    assert names == {"CheckpointLoaderSimple", "KSampler"}
    assert data["total_ns"] > 0
