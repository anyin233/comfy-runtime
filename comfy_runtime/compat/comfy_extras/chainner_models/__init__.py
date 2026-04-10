"""Import-compat stub for ``comfy_extras.chainner_models``.

ComfyUI's chainner_models package wraps the chaiNNer ESRGAN model
loaders.  was-node-suite imports it for upscaler discovery; the real
implementation needs spandrel.  We expose only the names so import
succeeds; calling them raises NotImplementedError pointing at
spandrel as the canonical loader.
"""

from comfy_runtime.compat.comfy_extras.chainner_models import model_loading
