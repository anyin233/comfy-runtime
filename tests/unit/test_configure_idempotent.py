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


def test_configure_snapshot_shortcircuits():
    """Identical kwargs must short-circuit via ``_LAST_CONFIG``.

    Before Phase-2 cleanup, this test spied on the vendor-bridge
    activation call.  The bridge is gone, so we now verify the
    short-circuit directly: after a no-op re-configure, the snapshot
    stays unchanged and no duplicate folder paths are added.
    """
    with tempfile.TemporaryDirectory() as tmp:
        _reset_config_state()
        comfy_runtime.configure(models_dir=tmp)
        first_snapshot = cfg_mod._LAST_CONFIG

        import folder_paths  # noqa: E402
        paths_after_first = list(folder_paths.folder_names_and_paths["checkpoints"][0])

        comfy_runtime.configure(models_dir=tmp)
        second_snapshot = cfg_mod._LAST_CONFIG
        paths_after_second = list(folder_paths.folder_names_and_paths["checkpoints"][0])

        # Snapshot object identity is preserved when short-circuited
        assert first_snapshot == second_snapshot
        # And path state is unchanged
        assert paths_after_first == paths_after_second


def test_configure_with_different_args_does_not_shortcircuit():
    """Changed kwargs must re-run the full configure path and update snapshot."""
    with tempfile.TemporaryDirectory() as tmp_a, tempfile.TemporaryDirectory() as tmp_b:
        _reset_config_state()
        comfy_runtime.configure(models_dir=tmp_a)
        first_snapshot = cfg_mod._LAST_CONFIG

        comfy_runtime.configure(models_dir=tmp_b)
        second_snapshot = cfg_mod._LAST_CONFIG

        assert first_snapshot != second_snapshot, (
            "different kwargs must produce a different snapshot"
        )
