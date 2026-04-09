# comfy-runtime

Minimal ComfyUI Runtime for Custom Nodes

`comfy-runtime` is a minimal Python runtime for loading and executing ComfyUI nodes outside the full ComfyUI app. It is designed for packaging custom nodes into scripts, workers, and microservices. It is not a server, not a UI, and not a workflow engine.

## Installation

```bash
pip install comfy-runtime
# Optional: built-in extra nodes
pip install comfy-builtin-nodes
```

## Quick Start

```python
import comfy_runtime

# Configure model paths (optional — has defaults)
comfy_runtime.configure(
    models_dir="/path/to/models",
    output_dir="/path/to/output",
)

# Execute a built-in node
result = comfy_runtime.execute_node(
    "EmptyLatentImage",
    width=512, height=512, batch_size=1,
)
latent = result[0]
print(f"Latent shape: {latent['samples'].shape}")

# Load extra nodes from a file
comfy_runtime.load_nodes_from_path("/path/to/my_custom_node.py")

# List all registered nodes
print(comfy_runtime.list_nodes())

# Get node info
info = comfy_runtime.get_node_info("EmptyLatentImage")
print(info["input_types"])
```

## API Reference

### `configure(models_dir, output_dir, input_dir, temp_dir, vram_mode, device)`

Configures runtime directories and device flags. This must be called before importing `comfy.model_management`.

### `execute_node(class_type, **kwargs) -> tuple`

Runs one registered node and returns its output tuple. Handles both V1 and V3 node styles transparently.

### `create_node_instance(class_type) -> object`

Creates a reusable instance of a registered node class. Useful for stateful nodes.

### `register_node(class_type, node_cls, display_name)`

Registers a node class manually under a given class type.

### `load_nodes_from_path(path) -> list[str]`

Loads nodes from a `.py` file or a directory of Python files. Supports both V1 mapping based nodes and V3 entrypoint based nodes.

### `list_nodes() -> list[str]`

Returns all registered node class names.

### `get_node_info(class_type) -> dict`

Returns public metadata for a node, including input types, return types, category, and execution method.

### `unregister_node(class_type)`

Removes a registered node from the runtime registry.

### `get_config() -> dict`

Returns the current runtime configuration.

### Exceptions

- `NodeNotFoundError`
- `NodeExecutionError`

## Microservice Example

```python
# Example: wrapping a node with FastAPI
from fastapi import FastAPI
import comfy_runtime

app = FastAPI()
comfy_runtime.configure(models_dir="/data/models", output_dir="/data/output")

@app.post("/generate")
def generate(width: int = 512, height: int = 512):
    result = comfy_runtime.execute_node(
        "EmptyLatentImage", width=width, height=height, batch_size=1,
    )
    return {"shape": list(result[0]["samples"].shape)}
```

## Memory Management

```python
from comfy.model_management import soft_empty_cache, cleanup_models
soft_empty_cache()  # Free unused VRAM
```

## Limitations

- Single-node execution only, no workflow graph
- Single-process only, no interrupt signaling across processes
- No built-in HTTP server, wrap it with FastAPI or gRPC yourself
- `configure()` must be called before importing `model_management`

## License

GPL-3.0, same as ComfyUI
