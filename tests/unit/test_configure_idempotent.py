"""Tests for configure() idempotency (Step 4)."""

import tempfile

import comfy_runtime
from comfy_runtime import config as cfg_mod


def _reset_config_state():
    cfg_mod._LAST_CONFIG = None


def test_configure_twice_does_not_duplicate_paths():
    """Calling configure() twice with identical models_dir must not prepend twice."""
    with tempfile.TemporaryDirectory() as tmp:
        _reset_config_state()
        comfy_runtime.configure(models_dir=tmp)
        import folder_paths  # noqa: E402 — exposed via shim

        snapshot = list(folder_paths.folder_names_and_paths["checkpoints"][0])

        comfy_runtime.configure(models_dir=tmp)
        second = list(folder_paths.folder_names_and_paths["checkpoints"][0])

        assert snapshot == second, (
            f"paths duplicated after second configure(): {snapshot} vs {second}"
        )


def test_configure_snapshot_shortcircuits(monkeypatch):
    """Identical kwargs must short-circuit and skip vendor bridge re-activation."""
    with tempfile.TemporaryDirectory() as tmp:
        _reset_config_state()
        comfy_runtime.configure(models_dir=tmp)

        call_count = {"n": 0}
        orig = cfg_mod._activate_vendor_bridge_if_available

        def spy():
            call_count["n"] += 1
            orig()

        monkeypatch.setattr(cfg_mod, "_activate_vendor_bridge_if_available", spy)
        comfy_runtime.configure(models_dir=tmp)

        assert call_count["n"] == 0, "identical configure should short-circuit"


def test_configure_with_different_args_does_not_shortcircuit(monkeypatch):
    """Changed kwargs must re-run the full configure path."""
    with tempfile.TemporaryDirectory() as tmp_a, tempfile.TemporaryDirectory() as tmp_b:
        _reset_config_state()
        comfy_runtime.configure(models_dir=tmp_a)

        call_count = {"n": 0}
        orig = cfg_mod._activate_vendor_bridge_if_available

        def spy():
            call_count["n"] += 1
            orig()

        monkeypatch.setattr(cfg_mod, "_activate_vendor_bridge_if_available", spy)
        comfy_runtime.configure(models_dir=tmp_b)

        assert call_count["n"] == 1
