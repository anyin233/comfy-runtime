"""Tests for the benchmarks._harness module.

The harness provides stopwatch primitives used by all bench_*.py scripts.
These tests exercise the pure-function parts so we can verify correctness
before trusting any benchmark output.
"""


def test_run_block_reports_iters_and_positive_median():
    """run_block must return a dict with iters + monotonic stats."""
    from benchmarks._harness import run_block

    result = run_block("noop", lambda: None, warmup=2, iters=10)

    assert result["name"] == "noop"
    assert result["iters"] == 10
    assert result["median_ns"] > 0
    assert result["min_ns"] > 0
    assert result["min_ns"] <= result["median_ns"] <= result["p95_ns"]
    assert result["mean_ns"] > 0
    assert result["stddev_ns"] >= 0


def test_run_block_executes_function_correct_number_of_times():
    """run_block should invoke the function exactly (warmup + iters) times."""
    from benchmarks._harness import run_block

    counter = {"n": 0}

    def work():
        counter["n"] += 1

    run_block("counter", work, warmup=3, iters=7)
    assert counter["n"] == 10


def test_compare_marks_improvement():
    """compare() should produce a markdown table containing bench names."""
    from benchmarks._harness import compare

    before = {
        "noop": {
            "median_ns": 1000,
            "min_ns": 900,
            "mean_ns": 1100,
            "p95_ns": 1200,
            "stddev_ns": 50,
            "iters": 10,
        }
    }
    after_faster = {
        "noop": {
            "median_ns": 500,
            "min_ns": 450,
            "mean_ns": 550,
            "p95_ns": 600,
            "stddev_ns": 25,
            "iters": 10,
        }
    }

    md = compare(before, after_faster)

    assert "noop" in md
    # Some representation of the 0.5x ratio or 50% improvement must appear.
    assert ("0.50" in md) or ("50%" in md) or ("50.0" in md)


def test_compare_handles_new_benchmarks_gracefully():
    """compare() must not crash when a bench exists only in the 'after' set."""
    from benchmarks._harness import compare

    md = compare(
        {},
        {
            "new_bench": {
                "median_ns": 100,
                "min_ns": 90,
                "mean_ns": 110,
                "p95_ns": 120,
                "stddev_ns": 5,
                "iters": 10,
            }
        },
    )

    assert "new_bench" in md


def test_compare_handles_removed_benchmarks_gracefully():
    """compare() must not crash when a bench exists only in the 'before' set."""
    from benchmarks._harness import compare

    md = compare(
        {
            "gone_bench": {
                "median_ns": 100,
                "min_ns": 90,
                "mean_ns": 110,
                "p95_ns": 120,
                "stddev_ns": 5,
                "iters": 10,
            }
        },
        {},
    )

    assert "gone_bench" in md
