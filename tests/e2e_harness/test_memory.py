"""Tests for memory helpers."""

import os

import pytest

from benchmarks.e2e._harness.memory import read_gpu_peak, read_vmhwm


def test_read_vmhwm_returns_positive_bytes():
    """VmHWM reader returns an int number of bytes greater than zero.

    We do not mock /proc because any live Python process has a real VmHWM.
    """
    value = read_vmhwm()
    assert isinstance(value, int)
    assert value > 0           # any process has some resident set
    assert value < 10 ** 13    # 10 TB sanity bound


def test_read_vmhwm_from_custom_status_file(tmp_path):
    """Reader accepts an explicit path argument for testability."""
    status_file = tmp_path / "status"
    status_file.write_text(
        "Name:\ttest\n"
        "VmPeak:\t  123456 kB\n"
        "VmHWM:\t  98765 kB\n"
        "VmRSS:\t  50000 kB\n"
    )
    value = read_vmhwm(str(status_file))
    assert value == 98765 * 1024


def test_read_vmhwm_raises_if_missing(tmp_path):
    status_file = tmp_path / "status"
    status_file.write_text("Name:\ttest\n")
    with pytest.raises(RuntimeError, match="VmHWM"):
        read_vmhwm(str(status_file))


def test_read_gpu_peak_structure():
    """When torch is available, return a dict with allocated/reserved keys.

    When torch is unavailable or CUDA is absent, return zeros.
    """
    result = read_gpu_peak()
    assert set(result.keys()) == {"allocated_bytes", "reserved_bytes"}
    assert isinstance(result["allocated_bytes"], int)
    assert isinstance(result["reserved_bytes"], int)
    assert result["allocated_bytes"] >= 0
    assert result["reserved_bytes"] >= 0
