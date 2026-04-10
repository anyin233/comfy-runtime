"""Tests for environment metadata collector."""

from benchmarks.e2e._harness.env import gather_env


REQUIRED_KEYS = {
    "hostname",
    "timestamp_utc",
    "python_version",
    "torch_version",
    "gpu_name",
    "gpu_memory_total_mb",
    "comfyui_commit",
}


def test_gather_env_contains_required_keys():
    env = gather_env()
    missing = REQUIRED_KEYS - set(env.keys())
    assert not missing, f"missing keys: {missing}"


def test_gather_env_types():
    env = gather_env()
    assert isinstance(env["hostname"], str)
    assert isinstance(env["timestamp_utc"], str)
    assert isinstance(env["python_version"], str)
    assert isinstance(env["gpu_memory_total_mb"], int) or env["gpu_memory_total_mb"] == "unknown"


def test_gather_env_timestamp_is_iso_format():
    from datetime import datetime
    env = gather_env()
    # Should parse without raising.
    datetime.fromisoformat(env["timestamp_utc"].rstrip("Z"))
