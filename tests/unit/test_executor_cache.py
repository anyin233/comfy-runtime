"""Tests for executor-level caches (introspection + instance pool).

Every test starts by clearing the relevant cache so results are
independent of test ordering.
"""

import inspect

import pytest

import comfy_runtime  # noqa: F401 — triggers bootstrap
from comfy_runtime import executor
from comfy_runtime.compat import nodes as compat_nodes


# ---------------------------------------------------------------------------
# Step 1 — introspection memoize
# ---------------------------------------------------------------------------


def test_is_v3_node_memoizes_result(monkeypatch):
    cls = compat_nodes.NODE_CLASS_MAPPINGS["EmptyLatentImage"]
    executor._V3_CHECK_CACHE.clear()

    call_count = {"n": 0}
    real = executor._compute_is_v3_node

    def spy(c):
        call_count["n"] += 1
        return real(c)

    monkeypatch.setattr(executor, "_compute_is_v3_node", spy)

    executor._is_v3_node(cls)
    executor._is_v3_node(cls)
    executor._is_v3_node(cls)

    assert call_count["n"] == 1


def test_is_v3_node_caches_both_true_and_false(monkeypatch):
    """A non-type argument should be cached as False on the fast path."""
    cls = compat_nodes.NODE_CLASS_MAPPINGS["EmptyLatentImage"]
    executor._V3_CHECK_CACHE.clear()

    # First call populates cache
    assert executor._is_v3_node(cls) is False
    assert cls in executor._V3_CHECK_CACHE
    assert executor._V3_CHECK_CACHE[cls] is False


def test_is_async_memoizes_result(monkeypatch):
    cls = compat_nodes.NODE_CLASS_MAPPINGS["EmptyLatentImage"]
    executor._COROUTINE_CACHE.clear()

    call_count = {"n": 0}
    real = inspect.iscoroutinefunction

    def spy(f):
        call_count["n"] += 1
        return real(f)

    monkeypatch.setattr(executor.inspect, "iscoroutinefunction", spy)

    func_name = getattr(cls, "FUNCTION", "generate")
    executor._is_async(cls, func_name)
    executor._is_async(cls, func_name)
    executor._is_async(cls, func_name)

    assert call_count["n"] == 1


def test_is_async_returns_correct_value_for_sync_method():
    cls = compat_nodes.NODE_CLASS_MAPPINGS["EmptyLatentImage"]
    executor._COROUTINE_CACHE.clear()

    func_name = getattr(cls, "FUNCTION")
    assert executor._is_async(cls, func_name) is False


def test_execute_node_still_works_after_memoization():
    """Smoke test: the pool/memoize changes must not break the happy path."""
    latent = executor.execute_node(
        "EmptyLatentImage", width=64, height=64, batch_size=1
    )
    # EmptyLatentImage returns a tuple with a dict containing 'samples' key
    assert isinstance(latent, tuple)
    assert "samples" in latent[0]


# ---------------------------------------------------------------------------
# Step 2 — list_nodes memoize
# ---------------------------------------------------------------------------


def test_list_nodes_returns_sorted_names():
    executor._invalidate_list_nodes_cache()
    names = executor.list_nodes()
    assert names == sorted(names)
    assert "EmptyLatentImage" in names


def test_list_nodes_uses_cache_on_second_call(monkeypatch):
    executor._invalidate_list_nodes_cache()
    first = executor.list_nodes()

    # Poison sorted() to blow up if called again during a cache hit.
    sort_calls = {"n": 0}
    real_sorted = executor.__builtins__["sorted"] if isinstance(
        executor.__builtins__, dict
    ) else sorted

    def spy_sorted(*args, **kwargs):
        sort_calls["n"] += 1
        return real_sorted(*args, **kwargs)

    # Patch the global sorted lookup inside the executor module.
    monkeypatch.setitem(executor.__dict__, "sorted", spy_sorted)

    second = executor.list_nodes()
    third = executor.list_nodes()

    assert first == second == third
    assert sort_calls["n"] == 0, "cache hit should not re-sort"


def test_list_nodes_cache_invalidates_on_register():
    from comfy_runtime.registry import register_node, unregister_node

    executor._invalidate_list_nodes_cache()
    first = executor.list_nodes()
    assert "__cache_test_dummy__" not in first

    class Dummy:
        FUNCTION = "run"

        def run(self):
            return ()

    try:
        register_node("__cache_test_dummy__", Dummy)
        second = executor.list_nodes()
        assert "__cache_test_dummy__" in second
    finally:
        unregister_node("__cache_test_dummy__")

    third = executor.list_nodes()
    assert "__cache_test_dummy__" not in third


# ---------------------------------------------------------------------------
# Step 3 — V1 instance pool
# ---------------------------------------------------------------------------


def test_v1_node_instance_is_pooled():
    """Repeated execute_node calls reuse the same V1 instance."""
    cls = compat_nodes.NODE_CLASS_MAPPINGS["EmptyLatentImage"]
    executor._V1_INSTANCE_POOL.pop(cls, None)

    init_count = {"n": 0}
    orig_init = cls.__init__

    def counting_init(self, *a, **kw):
        init_count["n"] += 1
        orig_init(self, *a, **kw)

    cls.__init__ = counting_init
    try:
        executor.execute_node("EmptyLatentImage", width=64, height=64, batch_size=1)
        executor.execute_node("EmptyLatentImage", width=64, height=64, batch_size=1)
        executor.execute_node("EmptyLatentImage", width=64, height=64, batch_size=1)
        assert init_count["n"] == 1, (
            f"expected pooled instance (1 __init__), got {init_count['n']}"
        )
    finally:
        cls.__init__ = orig_init
        executor._V1_INSTANCE_POOL.pop(cls, None)


def test_v1_node_opt_out_disables_pooling():
    """Classes with _COMFY_RUNTIME_NO_POOL get a fresh instance every call."""
    cls = compat_nodes.NODE_CLASS_MAPPINGS["EmptyLatentImage"]
    cls._COMFY_RUNTIME_NO_POOL = True
    executor._V1_INSTANCE_POOL.pop(cls, None)

    init_count = {"n": 0}
    orig_init = cls.__init__

    def counting_init(self, *a, **kw):
        init_count["n"] += 1
        orig_init(self, *a, **kw)

    cls.__init__ = counting_init
    try:
        executor.execute_node("EmptyLatentImage", width=64, height=64, batch_size=1)
        executor.execute_node("EmptyLatentImage", width=64, height=64, batch_size=1)
        assert init_count["n"] == 2
    finally:
        cls.__init__ = orig_init
        delattr(cls, "_COMFY_RUNTIME_NO_POOL")
        executor._V1_INSTANCE_POOL.pop(cls, None)


def test_v1_pool_invalidated_on_reregister():
    """register_node replacing a class must drop its pooled instance."""
    from comfy_runtime.registry import register_node, unregister_node

    class Dummy:
        FUNCTION = "run"

        def run(self):
            return ({},)

    class Dummy2:
        FUNCTION = "run"

        def run(self):
            return ({},)

    register_node("__pool_test_dummy__", Dummy)
    try:
        executor.execute_node("__pool_test_dummy__")
        assert Dummy in executor._V1_INSTANCE_POOL

        register_node("__pool_test_dummy__", Dummy2)
        # Old class must be dropped from the pool after re-registration.
        assert Dummy not in executor._V1_INSTANCE_POOL
    finally:
        unregister_node("__pool_test_dummy__")
