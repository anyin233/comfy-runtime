"""Tests for verify.py image-comparison logic."""

import numpy as np
import pytest

from benchmarks.e2e.verify import (
    ImageStats,
    compare_stats,
    compute_stats,
)


def test_compute_stats_from_ndarray():
    arr = np.full((64, 64, 3), 128, dtype=np.uint8)
    stats = compute_stats(arr)
    assert stats.shape == (64, 64, 3)
    assert stats.dtype == "uint8"
    assert stats.mean == pytest.approx(128.0)
    assert stats.std == pytest.approx(0.0)


def test_compare_stats_identical_passes():
    a = ImageStats(shape=(10, 10, 3), dtype="uint8", mean=100.0, std=30.0)
    b = ImageStats(shape=(10, 10, 3), dtype="uint8", mean=100.3, std=30.1)
    ok, reason = compare_stats(a, b, rel_tol=0.01)
    assert ok, reason


def test_compare_stats_shape_mismatch_fails():
    a = ImageStats(shape=(10, 10, 3), dtype="uint8", mean=100.0, std=30.0)
    b = ImageStats(shape=(10, 20, 3), dtype="uint8", mean=100.0, std=30.0)
    ok, reason = compare_stats(a, b, rel_tol=0.01)
    assert not ok
    assert "shape" in reason


def test_compare_stats_mean_drift_fails():
    a = ImageStats(shape=(10, 10, 3), dtype="uint8", mean=100.0, std=30.0)
    b = ImageStats(shape=(10, 10, 3), dtype="uint8", mean=150.0, std=30.0)
    ok, reason = compare_stats(a, b, rel_tol=0.01)
    assert not ok
    assert "mean" in reason
