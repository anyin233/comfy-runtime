"""Node registration and loading utilities for comfy-runtime."""

import asyncio
import importlib
import importlib.util
import os
import sys

from comfy_runtime.compat import nodes as _nodes_mod


def register_node(class_type: str, node_cls: type, display_name: str | None = None):
    _nodes_mod.NODE_CLASS_MAPPINGS[class_type] = node_cls
    if display_name is not None:
        _nodes_mod.NODE_DISPLAY_NAME_MAPPINGS[class_type] = display_name


def register_nodes(mappings: dict, display_names: dict | None = None):
    _nodes_mod.NODE_CLASS_MAPPINGS.update(mappings)
    if display_names:
        _nodes_mod.NODE_DISPLAY_NAME_MAPPINGS.update(display_names)


def unregister_node(class_type: str):
    _nodes_mod.NODE_CLASS_MAPPINGS.pop(class_type, None)
    _nodes_mod.NODE_DISPLAY_NAME_MAPPINGS.pop(class_type, None)


def load_nodes_from_path(path: str) -> list[str]:
    """Load node module(s) from a .py file or directory path.

    Handles both V1 (NODE_CLASS_MAPPINGS in module) and V3
    (comfy_entrypoint async function) node formats.

    Args:
        path: Absolute or relative path to a .py file or directory.

    Returns:
        List of registered class_type names.

    Raises:
        ValueError: If path is not a .py file or directory.
    """
    path = os.path.abspath(path)
    if os.path.isdir(path):
        return _load_from_directory(path)
    elif os.path.isfile(path) and path.endswith(".py"):
        return _load_from_file(path)
    else:
        raise ValueError(f"Invalid path (must be .py file or directory): {path}")


def _load_from_file(filepath: str) -> list[str]:
    module_name = os.path.splitext(os.path.basename(filepath))[0]

    spec = importlib.util.spec_from_file_location(module_name, filepath)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create module spec for {filepath}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception as e:
        sys.modules.pop(module_name, None)
        raise ImportError(f"Failed to load {filepath}: {e}") from e

    registered = []

    # V1: module has NODE_CLASS_MAPPINGS dict
    if hasattr(module, "NODE_CLASS_MAPPINGS"):
        mappings = module.NODE_CLASS_MAPPINGS
        if isinstance(mappings, dict):
            for class_type, node_cls in mappings.items():
                _nodes_mod.NODE_CLASS_MAPPINGS[class_type] = node_cls
                node_cls.RELATIVE_PYTHON_MODULE = module_name
                registered.append(class_type)

    # V3: module has comfy_entrypoint() async function
    if hasattr(module, "comfy_entrypoint"):
        registered.extend(_load_v3_nodes(module, module_name))

    return registered


def _load_v3_nodes(module, module_name: str) -> list[str]:
    registered = []
    try:
        entrypoint = module.comfy_entrypoint
        if asyncio.iscoroutinefunction(entrypoint):
            ext = asyncio.run(entrypoint())
        else:
            ext = entrypoint()

        if hasattr(ext, "get_node_list"):
            get_nodes = ext.get_node_list
            if asyncio.iscoroutinefunction(get_nodes):
                node_list = asyncio.run(get_nodes())
            else:
                node_list = get_nodes()
            for node_cls in node_list:
                try:
                    class_type = _get_v3_class_type(node_cls)
                    _nodes_mod.NODE_CLASS_MAPPINGS[class_type] = node_cls
                    node_cls.RELATIVE_PYTHON_MODULE = module_name
                    registered.append(class_type)
                except Exception:
                    pass  # Skip individual nodes that fail schema creation
    except Exception:
        pass  # V3 loading is best-effort

    return registered


def _get_v3_class_type(node_cls: type) -> str:
    schema = node_cls.define_schema()
    # Try _node_id first (set by finalize), then node_id, then class name
    if hasattr(schema, "_node_id"):
        return schema._node_id
    if hasattr(schema, "node_id") and schema.node_id:
        return schema.node_id
    return node_cls.__name__


def _load_from_directory(dirpath: str) -> list[str]:
    registered = []

    # If directory has __init__.py, load it as a package first
    init_path = os.path.join(dirpath, "__init__.py")
    if os.path.isfile(init_path):
        module_name = os.path.basename(dirpath)
        parent_dir = os.path.dirname(dirpath)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)

        spec = importlib.util.spec_from_file_location(
            module_name,
            init_path,
            submodule_search_locations=[dirpath],
        )
        if spec is not None and spec.loader is not None:
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)
            except Exception:
                sys.modules.pop(module_name, None)
                # Fall through to individual file loading below
            else:
                # Check V1 mappings
                if hasattr(module, "NODE_CLASS_MAPPINGS"):
                    mappings = module.NODE_CLASS_MAPPINGS
                    if isinstance(mappings, dict):
                        for class_type, node_cls in mappings.items():
                            _nodes_mod.NODE_CLASS_MAPPINGS[class_type] = node_cls
                            node_cls.RELATIVE_PYTHON_MODULE = module_name
                            registered.append(class_type)

                # Check V3
                if hasattr(module, "comfy_entrypoint"):
                    registered.extend(_load_v3_nodes(module, module_name))

                if registered:
                    return registered

    # Fallback: load individual .py files
    for fname in sorted(os.listdir(dirpath)):
        if fname.endswith(".py") and not fname.startswith("_"):
            try:
                registered.extend(_load_from_file(os.path.join(dirpath, fname)))
            except Exception:
                pass
    return registered
