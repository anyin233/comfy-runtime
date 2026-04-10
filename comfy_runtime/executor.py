"""Single-node execution helpers for comfy-runtime."""

import asyncio
import inspect
from importlib import import_module
from typing import Any, cast

from comfy_runtime.compat.comfy_api.internal import _ComfyNodeInternal
from comfy_runtime.compat import nodes


class NodeNotFoundError(KeyError):
    """Raised when a requested node type is not registered."""


class NodeExecutionError(RuntimeError):
    """Raised when a node function fails during execution."""


# Per-class caches for introspection results. Entries survive until the
# registry explicitly invalidates them via _invalidate_caches_for(cls),
# which is called from registry.register_node / unregister_node when a
# class is replaced or removed.
_V3_CHECK_CACHE: dict[type, bool] = {}
_COROUTINE_CACHE: dict[tuple[type, str], bool] = {}
# Cache: (id(mappings), len(mappings)) -> tuple of sorted node names.
# Invalidated explicitly by registry.register_node / unregister_node so
# delete-then-add-different-name sequences with equal cardinality still work.
_LIST_NODES_CACHE: tuple[int, int, tuple[str, ...]] | None = None
# V1 node singleton pool. One instance per class_type. Node authors who
# rely on per-call __init__ side effects can set _COMFY_RUNTIME_NO_POOL = True
# on their class to opt out. Not thread-safe for concurrent same-class calls
# (same implicit contract as before).
_V1_INSTANCE_POOL: dict[type, object] = {}


def get_node_class(class_type: str) -> type:
    """Return the registered class for a node type."""
    mappings = nodes.NODE_CLASS_MAPPINGS
    if class_type not in mappings:
        raise NodeNotFoundError(class_type)
    return mappings[class_type]


def create_node_instance(class_type: str):
    """Create a fresh instance of a registered node type."""
    cls = get_node_class(class_type)
    return cls()


def list_nodes() -> list[str]:
    """Return all registered node type names (memoized)."""
    global _LIST_NODES_CACHE
    mappings = nodes.NODE_CLASS_MAPPINGS
    key = (id(mappings), len(mappings))
    if _LIST_NODES_CACHE is not None and _LIST_NODES_CACHE[:2] == key:
        return list(_LIST_NODES_CACHE[2])
    sorted_names = tuple(sorted(mappings.keys()))
    _LIST_NODES_CACHE = (id(mappings), len(mappings), sorted_names)
    return list(sorted_names)


def _invalidate_list_nodes_cache() -> None:
    """Clear the list_nodes() memoization cache."""
    global _LIST_NODES_CACHE
    _LIST_NODES_CACHE = None


def _compute_is_v3_node(cls: type) -> bool:
    """Raw V3-detection logic (no caching)."""
    if not isinstance(cls, type):
        return False

    v3_bases = [_ComfyNodeInternal]
    try:
        shimmed_internal = import_module("comfy_api.internal")
        shimmed_base = getattr(shimmed_internal, "_ComfyNodeInternal", None)
        if isinstance(shimmed_base, type):
            v3_bases.append(shimmed_base)
    except ImportError:
        pass

    return issubclass(cls, tuple(v3_bases)) or (
        hasattr(cls, "define_schema")
        and hasattr(cls, "execute")
        and not hasattr(cls, "FUNCTION")
    )


def _is_v3_node(cls: type) -> bool:
    """Cached wrapper around _compute_is_v3_node.

    Cache is keyed on the class object itself; entries are cleared by
    _invalidate_caches_for when the registry replaces/removes a class.
    """
    cached = _V3_CHECK_CACHE.get(cls)
    if cached is not None:
        return cached
    # Indirect lookup to allow monkeypatching _compute_is_v3_node in tests.
    result = _compute_is_v3_node(cls)
    _V3_CHECK_CACHE[cls] = result
    return result


def _is_async(cls: type, func_name: str) -> bool:
    """Cached check whether cls.func_name is a coroutine function."""
    key = (cls, func_name)
    cached = _COROUTINE_CACHE.get(key)
    if cached is not None:
        return cached
    func = getattr(cls, func_name, None)
    result = bool(func is not None and inspect.iscoroutinefunction(func))
    _COROUTINE_CACHE[key] = result
    return result


def _get_v1_instance(cls: type):
    """Return a cached V1 node instance for *cls*, or create one.

    Classes can opt out with ``_COMFY_RUNTIME_NO_POOL = True``.
    """
    if getattr(cls, "_COMFY_RUNTIME_NO_POOL", False):
        return cls()
    inst = _V1_INSTANCE_POOL.get(cls)
    if inst is None:
        inst = cls()
        _V1_INSTANCE_POOL[cls] = inst
    return inst


def _invalidate_caches_for(cls: type) -> None:
    """Drop all cached entries for *cls*.

    Called by registry mutators when a node class is replaced or removed.
    """
    _V3_CHECK_CACHE.pop(cls, None)
    stale_keys = [k for k in _COROUTINE_CACHE if k[0] is cls]
    for k in stale_keys:
        _COROUTINE_CACHE.pop(k, None)
    _V1_INSTANCE_POOL.pop(cls, None)


def _get_v3_schema(cls: type):
    get_schema = getattr(cls, "GET_SCHEMA", None)
    if callable(get_schema):
        return get_schema()
    define_schema = getattr(cls, "define_schema", None)
    if not callable(define_schema):
        raise NodeExecutionError(f"Node {cls.__name__} does not define a V3 schema")
    schema = define_schema()
    finalize = getattr(schema, "finalize", None)
    if callable(finalize):
        finalize()
    return schema


def _get_v3_node_info(class_type: str, cls: type) -> dict[str, Any]:
    schema = cast(Any, _get_v3_schema(cls))
    if hasattr(schema, "get_v1_info"):
        info = schema.get_v1_info(cls)
        return {
            "class_type": class_type,
            "input_types": info.input,
            "return_types": tuple(info.output),
            "return_names": tuple(info.output_name)
            if info.output_name is not None
            else None,
            "function": "execute",
            "category": info.category,
            "output_node": info.output_node,
            "input_is_list": info.is_input_list,
            "output_is_list": info.output_is_list,
        }

    return {
        "class_type": class_type,
        "input_types": {},
        "return_types": getattr(cls, "RETURN_TYPES", ()),
        "return_names": getattr(cls, "RETURN_NAMES", None),
        "function": "execute",
        "category": getattr(schema, "category", ""),
        "output_node": getattr(schema, "is_output_node", False),
        "input_is_list": getattr(schema, "is_input_list", False),
        "output_is_list": [
            getattr(output, "is_output_list", False)
            for output in getattr(schema, "outputs", [])
        ],
    }


def _unwrap_v3_result(result: Any):
    if hasattr(result, "result"):
        return result.result
    if isinstance(result, dict) and "result" in result:
        return result["result"]
    return result


def get_node_info(class_type: str) -> dict:
    """Return public metadata for a registered node type."""
    cls = get_node_class(class_type)
    if _is_v3_node(cls):
        return _get_v3_node_info(class_type, cls)

    input_types = cls.INPUT_TYPES() if hasattr(cls, "INPUT_TYPES") else {}
    return {
        "class_type": class_type,
        "input_types": input_types,
        "return_types": getattr(cls, "RETURN_TYPES", ()),
        "return_names": getattr(cls, "RETURN_NAMES", None),
        "function": getattr(cls, "FUNCTION", None),
        "category": getattr(cls, "CATEGORY", ""),
        "output_node": getattr(cls, "OUTPUT_NODE", False),
        "input_is_list": getattr(cls, "INPUT_IS_LIST", False),
        "output_is_list": getattr(cls, "OUTPUT_IS_LIST", False),
    }


def execute_node(class_type: str, **kwargs):
    """Execute a single registered node with keyword arguments."""
    cls = get_node_class(class_type)

    if _is_v3_node(cls):
        func = getattr(cls, "execute", None)
        if func is None:
            raise NodeExecutionError(f"Node {class_type} has no execute classmethod")

        try:
            if _is_async(cls, "execute"):
                result = asyncio.run(func(**kwargs))
            else:
                result = func(**kwargs)
        except Exception as exc:
            raise NodeExecutionError(
                f"Node {class_type}.execute failed: {exc}"
            ) from exc

        return _unwrap_v3_result(result)

    instance = _get_v1_instance(cls)
    func_name = getattr(cls, "FUNCTION", None)
    if func_name is None:
        raise NodeExecutionError(f"Node {class_type} has no FUNCTION attribute")

    func = getattr(instance, func_name)

    try:
        if _is_async(cls, func_name):
            result = asyncio.run(func(**kwargs))
        else:
            result = func(**kwargs)
    except Exception as exc:
        raise NodeExecutionError(
            f"Node {class_type}.{func_name} failed: {exc}"
        ) from exc

    if isinstance(result, dict) and "result" in result:
        return result["result"]

    return result
