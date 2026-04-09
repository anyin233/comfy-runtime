"""Single-node execution helpers for comfy-runtime."""

import asyncio
import inspect
from importlib import import_module
from typing import Any, cast

from comfy_runtime._vendor.comfy_api.internal import _ComfyNodeInternal
from comfy_runtime._vendor import nodes


class NodeNotFoundError(KeyError):
    """Raised when a requested node type is not registered."""


class NodeExecutionError(RuntimeError):
    """Raised when a node function fails during execution."""


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
    """Return all registered node type names."""
    return sorted(nodes.NODE_CLASS_MAPPINGS.keys())


def _is_v3_node(cls: type) -> bool:
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
            if inspect.iscoroutinefunction(func):
                result = asyncio.run(func(**kwargs))
            else:
                result = func(**kwargs)
        except Exception as exc:
            raise NodeExecutionError(
                f"Node {class_type}.execute failed: {exc}"
            ) from exc

        return _unwrap_v3_result(result)

    instance = cls()
    func_name = getattr(cls, "FUNCTION", None)
    if func_name is None:
        raise NodeExecutionError(f"Node {class_type} has no FUNCTION attribute")

    func = getattr(instance, func_name)

    try:
        if inspect.iscoroutinefunction(func):
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
