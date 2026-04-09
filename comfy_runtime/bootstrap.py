"""Register all stub modules in sys.modules before any comfy imports.

Bootstrap order is critical — comfy_aimdo must come first because many
ComfyUI modules import it at the top level.  The sequence is:

1. comfy_aimdo stubs (7 import sites across ComfyUI)
2. server stub (nodes_images.py imports PromptServer at module level)
3. latent_preview stub (avoids torch/PIL/heavy-comfy imports)
4. comfy.options shim (sets args_parsing = False so cli_args skips argparse)
5. vendor shims (registers _vendor/ packages under short names in sys.modules)
"""

import importlib
from importlib.machinery import ModuleSpec
import sys
import types


def bootstrap():
    """Register all stubs in sys.modules BEFORE any comfy imports."""
    from comfy_runtime.stubs.comfy_aimdo_stub import install_aimdo_stubs

    install_aimdo_stubs()

    import comfy_runtime.stubs.server_stub as server_module

    sys.modules["server"] = server_module

    import comfy_runtime.stubs.latent_preview_stub as latent_preview_module

    sys.modules["latent_preview"] = latent_preview_module

    try:
        importlib.import_module("torchvision")
    except Exception:

        def _torchvision_unavailable(*args, **kwargs):
            raise RuntimeError(
                "torchvision is unavailable in this comfy_runtime environment"
            )

        torchvision_module = types.ModuleType("torchvision")
        torchvision_models = types.ModuleType("torchvision.models")
        torchvision_ops = types.ModuleType("torchvision.ops")
        torchvision_transforms = types.ModuleType("torchvision.transforms")

        setattr(
            torchvision_module,
            "__spec__",
            ModuleSpec("torchvision", loader=None, is_package=True),
        )
        setattr(torchvision_module, "__path__", [])
        setattr(
            torchvision_models,
            "__spec__",
            ModuleSpec("torchvision.models", loader=None),
        )
        setattr(
            torchvision_ops,
            "__spec__",
            ModuleSpec("torchvision.ops", loader=None),
        )
        setattr(
            torchvision_transforms,
            "__spec__",
            ModuleSpec("torchvision.transforms", loader=None),
        )

        setattr(torchvision_models, "efficientnet_v2_s", _torchvision_unavailable)
        setattr(torchvision_ops, "box_convert", _torchvision_unavailable)
        setattr(torchvision_transforms, "Compose", _torchvision_unavailable)
        setattr(torchvision_transforms, "Normalize", _torchvision_unavailable)

        setattr(torchvision_module, "models", torchvision_models)
        setattr(torchvision_module, "ops", torchvision_ops)
        setattr(torchvision_module, "transforms", torchvision_transforms)

        sys.modules["torchvision"] = torchvision_module
        sys.modules["torchvision.models"] = torchvision_models
        sys.modules["torchvision.ops"] = torchvision_ops
        sys.modules["torchvision.transforms"] = torchvision_transforms

    try:
        importlib.import_module("av")
    except Exception:

        def _av_unavailable(*args, **kwargs):
            raise RuntimeError("av is unavailable in this comfy_runtime environment")

        av_module = types.ModuleType("av")
        av_container = types.ModuleType("av.container")
        av_subtitles = types.ModuleType("av.subtitles")
        av_subtitles_stream = types.ModuleType("av.subtitles.stream")
        input_container = type("InputContainer", (), {})
        subtitle_stream = type("SubtitleStream", (), {})
        video_stream = type("VideoStream", (), {})

        setattr(av_module, "__spec__", ModuleSpec("av", loader=None, is_package=True))
        setattr(av_module, "__path__", [])
        setattr(av_container, "__spec__", ModuleSpec("av.container", loader=None))
        setattr(
            av_subtitles,
            "__spec__",
            ModuleSpec("av.subtitles", loader=None, is_package=True),
        )
        setattr(av_subtitles, "__path__", [])
        setattr(
            av_subtitles_stream,
            "__spec__",
            ModuleSpec("av.subtitles.stream", loader=None),
        )

        setattr(av_container, "InputContainer", input_container)
        setattr(av_subtitles_stream, "SubtitleStream", subtitle_stream)
        setattr(av_subtitles, "stream", av_subtitles_stream)
        setattr(av_module, "container", av_container)
        setattr(av_module, "subtitles", av_subtitles)
        setattr(av_module, "VideoStream", video_stream)
        setattr(av_module, "open", _av_unavailable)
        setattr(av_module, "time_base", 1)

        sys.modules["av"] = av_module
        sys.modules["av.container"] = av_container
        sys.modules["av.subtitles"] = av_subtitles
        sys.modules["av.subtitles.stream"] = av_subtitles_stream

    if "comfy" not in sys.modules:
        try:
            import comfy  # noqa: F401  # type: ignore[import-not-found]
        except ImportError:
            comfy_parent = types.ModuleType("comfy")
            comfy_parent.__path__ = []
            sys.modules["comfy"] = comfy_parent

    options_module = types.ModuleType("comfy.options")
    options_module.args_parsing = False  # type: ignore[attr-defined]
    options_module.enable_args_parsing = lambda enable=True: None  # type: ignore[attr-defined]
    sys.modules["comfy.options"] = options_module
    sys.modules["comfy"].options = options_module  # type: ignore[attr-defined]

    from comfy_runtime.shim import install_shims

    try:
        cli_args_mod = importlib.import_module("comfy_runtime._vendor.comfy.cli_args")
        torch = importlib.import_module("torch")

        if not torch.cuda.is_available():
            cli_args_mod.args.cpu = True
        sys.modules["comfy.cli_args"] = cli_args_mod
    except Exception:
        pass

    try:
        diffusionmodules_mod = importlib.import_module(
            "comfy_runtime._vendor.comfy.ldm.modules.diffusionmodules"
        )
        diffusion_mmdit_mod = importlib.import_module(
            "comfy_runtime._vendor.comfy.ldm.modules.diffusionmodules.mmdit"
        )
        setattr(diffusionmodules_mod, "mmdit", diffusion_mmdit_mod)
        sys.modules["comfy.ldm.modules.diffusionmodules"] = diffusionmodules_mod
        sys.modules["comfy.ldm.modules.diffusionmodules.mmdit"] = diffusion_mmdit_mod
    except Exception:
        pass

    install_shims()
