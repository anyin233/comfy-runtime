"""Device and VRAM management for comfy_runtime.

MIT reimplementation of comfy.model_management — provides the same
public API surface using standard PyTorch and psutil operations.
"""

import enum
import gc
import logging
import threading
from typing import Optional

import psutil
import torch

from comfy_runtime.compat.comfy.cli_args import args

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class VRAMState(enum.Enum):
    DISABLED = 0
    NO_VRAM = 1
    LOW_VRAM = 2
    NORMAL_VRAM = 3
    HIGH_VRAM = 4
    SHARED = 5


class CPUState(enum.Enum):
    GPU = 0
    CPU = 1
    MPS = 2


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

vram_state = VRAMState.NORMAL_VRAM
set_vram_to = VRAMState.NORMAL_VRAM
cpu_state = CPUState.GPU
total_vram = 0

lowvram_available = True
directml_enabled = False

XFORMERS_IS_AVAILABLE = False
XFORMERS_ENABLED_VAE = False
XFORMERS_VERSION = ""
ENABLE_PYTORCH_ATTENTION = False
SUPPORT_FP8_OPS = False
FORCE_FP32 = False
PRIORITIZE_FP16 = False
DISABLE_SMART_MEMORY = False
EXTRA_RESERVED_VRAM = 0

FLOAT8_TYPES = [torch.float8_e4m3fn, torch.float8_e5m2]
if hasattr(torch, "float8_e8m0fnu"):
    FLOAT8_TYPES.append(torch.float8_e8m0fnu)

current_loaded_models: list = []

_interrupt_processing = False
_interrupt_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Device detection (at import time)
# ---------------------------------------------------------------------------


def _detect_state():
    global cpu_state, vram_state, set_vram_to, total_vram, lowvram_available
    global XFORMERS_IS_AVAILABLE, XFORMERS_ENABLED_VAE, ENABLE_PYTORCH_ATTENTION
    global FORCE_FP32, PRIORITIZE_FP16, DISABLE_SMART_MEMORY, EXTRA_RESERVED_VRAM

    if args.cpu:
        cpu_state = CPUState.CPU

    if cpu_state == CPUState.CPU:
        vram_state = VRAMState.DISABLED
        set_vram_to = VRAMState.DISABLED
        return

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        cpu_state = CPUState.MPS
        vram_state = VRAMState.SHARED
        set_vram_to = VRAMState.SHARED
        return

    if not torch.cuda.is_available():
        cpu_state = CPUState.CPU
        vram_state = VRAMState.DISABLED
        set_vram_to = VRAMState.DISABLED
        return

    # CUDA available
    cpu_state = CPUState.GPU
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    total_vram = getattr(props, "total_memory", None) or getattr(props, "total_mem", 0)

    if args.novram:
        vram_state = VRAMState.NO_VRAM
    elif args.lowvram:
        vram_state = VRAMState.LOW_VRAM
    elif args.normalvram:
        vram_state = VRAMState.NORMAL_VRAM
    elif args.highvram or args.gpu_only:
        vram_state = VRAMState.HIGH_VRAM
    else:
        vram_state = VRAMState.NORMAL_VRAM

    set_vram_to = vram_state

    if args.force_fp32:
        FORCE_FP32 = True
    if args.force_fp16:
        PRIORITIZE_FP16 = True
    if args.disable_smart_memory:
        DISABLE_SMART_MEMORY = True
    if args.reserve_vram is not None:
        EXTRA_RESERVED_VRAM = int(args.reserve_vram * 1024 * 1024 * 1024)

    # xformers detection
    if not args.disable_xformers:
        try:
            import xformers
            import xformers.ops

            XFORMERS_IS_AVAILABLE = True
            XFORMERS_ENABLED_VAE = True
            XFORMERS_VERSION = xformers.__version__
        except ImportError:
            pass

    if args.use_pytorch_cross_attention or (not XFORMERS_IS_AVAILABLE):
        ENABLE_PYTORCH_ATTENTION = True


_detect_state()


# ---------------------------------------------------------------------------
# Device query functions
# ---------------------------------------------------------------------------


# Per-sub-model device pins.  None entries fall back to get_torch_device().
# Populated by set_device_assignment() — the Phase-3 multi-GPU hook.
_device_assignment = {
    "unet": None,
    "text_encoder": None,
    "vae": None,
    "clip_vision": None,
    "controlnet": None,
}


def _coerce_device(dev):
    """Accept ``None``, ``str``, or ``torch.device`` → ``torch.device | None``."""
    if dev is None:
        return None
    if isinstance(dev, torch.device):
        return dev
    return torch.device(dev)


def set_device_assignment(
    unet=None,
    text_encoder=None,
    vae=None,
    clip_vision=None,
    controlnet=None,
    reset: bool = True,
) -> None:
    """Pin sub-models to specific devices for multi-GPU deployments.

    Pass individual devices (``torch.device`` objects or strings like
    ``"cuda:0"`` / ``"cpu"``) to pin a given sub-model onto it.  Any
    argument left as ``None`` falls back to :func:`get_torch_device`.

    Calling :func:`set_device_assignment` with no arguments resets all
    pins to ``None`` (and therefore to the default single-device
    behavior).  Pass ``reset=False`` to update only the specified
    slots and leave the others as-is.

    Example::

        # Pin text encoder to cuda:1, leave UNet on the default device
        set_device_assignment(text_encoder="cuda:1")

    Args:
        unet:         Device for the UNet.
        text_encoder: Device for CLIP/T5 text encoders.
        vae:          Device for the VAE.
        clip_vision:  Device for CLIP vision encoders.
        controlnet:   Device for ControlNet models.
        reset:        When ``True`` (the default), clears all other pins
            back to ``None``.  When ``False``, leaves them untouched.
    """
    global _device_assignment
    if reset:
        _device_assignment = {k: None for k in _device_assignment}
    _device_assignment["unet"] = _coerce_device(unet)
    _device_assignment["text_encoder"] = _coerce_device(text_encoder)
    _device_assignment["vae"] = _coerce_device(vae)
    _device_assignment["clip_vision"] = _coerce_device(clip_vision)
    _device_assignment["controlnet"] = _coerce_device(controlnet)


def get_device_assignment(slot: str):
    """Return the pinned device for ``slot``, or ``None`` if unpinned."""
    return _device_assignment.get(slot)


def get_device_list() -> list:
    """Return the list of available compute devices.

    Honors ``CUDA_VISIBLE_DEVICES`` (via ``torch.cuda.device_count()``).
    On CPU-only hosts returns ``[torch.device("cpu")]``.  On MPS hosts
    returns ``[torch.device("mps")]``.
    """
    if cpu_state == CPUState.CPU:
        return [torch.device("cpu")]
    if cpu_state == CPUState.MPS:
        return [torch.device("mps")]
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        return [torch.device("cuda", i) for i in range(count)]
    return [torch.device("cpu")]


def get_torch_device() -> torch.device:
    """Return the primary compute device."""
    if cpu_state == CPUState.MPS:
        return torch.device("mps")
    if cpu_state == CPUState.CPU:
        return torch.device("cpu")
    if args.cuda_device is not None:
        return torch.device("cuda", args.cuda_device)
    return torch.device("cuda")


def is_device_cpu(device) -> bool:
    if isinstance(device, str):
        return device == "cpu"
    return device.type == "cpu"


def is_device_mps(device) -> bool:
    if isinstance(device, str):
        return device == "mps"
    return device.type == "mps"


def is_device_cuda(device) -> bool:
    if isinstance(device, str):
        return device.startswith("cuda")
    return device.type == "cuda"


def is_device_xpu(device) -> bool:
    if isinstance(device, str):
        return device.startswith("xpu")
    return device.type == "xpu"


def cpu_mode() -> bool:
    return cpu_state == CPUState.CPU


def mps_mode() -> bool:
    return cpu_state == CPUState.MPS


def is_nvidia() -> bool:
    if cpu_state != CPUState.GPU:
        return False
    try:
        name = torch.cuda.get_device_name().lower()
        return (
            "nvidia" in name or "geforce" in name or "quadro" in name or "tesla" in name
        )
    except Exception:
        return False


def is_amd() -> bool:
    if cpu_state != CPUState.GPU:
        return False
    try:
        name = torch.cuda.get_device_name().lower()
        return "amd" in name or "radeon" in name
    except Exception:
        return False


def is_intel_xpu() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def is_ascend_npu() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()


def is_mlu() -> bool:
    return hasattr(torch, "mlu") and torch.mlu.is_available()


def is_ixuca() -> bool:
    return False


def is_wsl() -> bool:
    try:
        with open("/proc/version", "r") as f:
            return "microsoft" in f.read().lower()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Memory query functions
# ---------------------------------------------------------------------------


def get_total_memory(dev=None, torch_total_too=False):
    """Return total memory (in bytes) on *dev*."""
    if dev is None:
        dev = get_torch_device()
    if is_device_cpu(dev) or is_device_mps(dev):
        mem = psutil.virtual_memory().total
        if torch_total_too:
            return mem, mem
        return mem
    try:
        stats = torch.cuda.mem_get_info(dev)
        total = stats[1]
        if torch_total_too:
            return total, torch.cuda.memory_reserved(dev)
        return total
    except Exception:
        if torch_total_too:
            return 1024**3, 0
        return 1024**3


def get_free_memory(dev=None, torch_free_too=False):
    """Return free memory (in bytes) on *dev*."""
    if dev is None:
        dev = get_torch_device()
    if is_device_cpu(dev) or is_device_mps(dev):
        mem = psutil.virtual_memory().available
        if torch_free_too:
            return mem, mem
        return mem
    try:
        stats = torch.cuda.mem_get_info(dev)
        mem_free_total = stats[0]
        mem_free_torch = torch.cuda.memory_reserved(dev) - torch.cuda.memory_allocated(
            dev
        )
        mem_free_total += mem_free_torch
        if torch_free_too:
            return mem_free_total, mem_free_torch
        return mem_free_total
    except Exception:
        if torch_free_too:
            return 1024**3, 0
        return 1024**3


# ---------------------------------------------------------------------------
# Dtype selection
# ---------------------------------------------------------------------------


def dtype_size(dtype) -> int:
    """Return the byte size of a single element of *dtype*."""
    return torch.tensor([], dtype=dtype).element_size()


def module_size(module) -> int:
    """Return the total byte size of a module's ``state_dict`` tensors.

    Used by ComfyUI's upscale / controlnet helper nodes to estimate how
    much VRAM to free before loading another model.  Matches the
    implementation in upstream ``comfy.model_management.module_size``.

    Args:
        module: Any object with a ``state_dict()`` method (typically an
            ``nn.Module``).

    Returns:
        Total bytes occupied by the module's parameters and buffers.
    """
    total = 0
    sd = module.state_dict()
    for k in sd:
        total += sd[k].nbytes
    return total


def supports_dtype(device, dtype) -> bool:
    """Check if *device* supports *dtype*."""
    if dtype == torch.float32:
        return True
    if is_device_cpu(device):
        return dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
    if is_device_mps(device):
        return dtype in (torch.float16, torch.float32)
    # CUDA — check capability
    try:
        if dtype == torch.bfloat16:
            cap = torch.cuda.get_device_capability(device)
            return cap[0] >= 8  # Ampere+
        return True
    except Exception:
        return True


def supports_cast(device, dtype) -> bool:
    """Check if *device* supports casting to *dtype*."""
    return supports_dtype(device, dtype)


def supports_fp8_compute(device=None) -> bool:
    """Check if *device* supports native fp8 matrix multiply."""
    if args.supports_fp8_compute:
        return True
    if device is None:
        device = get_torch_device()
    if is_device_cpu(device) or is_device_mps(device):
        return False
    try:
        cap = torch.cuda.get_device_capability(device)
        return cap[0] >= 9  # Hopper+
    except Exception:
        return False


def should_use_fp16(
    device=None, model_params=0, prioritize_performance=True, manual_cast=False
) -> bool:
    """Determine whether fp16 should be used on *device*."""
    if FORCE_FP32:
        return False
    if device is None:
        device = get_torch_device()
    if is_device_cpu(device):
        return False
    if is_device_mps(device):
        return True
    if PRIORITIZE_FP16:
        return True
    # CUDA — check capability
    try:
        cap = torch.cuda.get_device_capability(device)
        if cap[0] < 7:
            return False  # Pre-Volta
        props = torch.cuda.get_device_properties(device)
        if prioritize_performance:
            # FP16 tensor cores available on Volta+ (compute >= 7.0)
            return True
        if manual_cast:
            free_mem = get_free_memory(device)
            if model_params * 2 > free_mem:
                return True
        return True
    except Exception:
        return False


def should_use_bf16(
    device=None, model_params=0, prioritize_performance=True, manual_cast=False
) -> bool:
    """Determine whether bf16 should be used on *device*."""
    if FORCE_FP32:
        return False
    if device is None:
        device = get_torch_device()
    if is_device_cpu(device):
        return False  # bf16 on CPU is slow
    if is_device_mps(device):
        return False  # MPS doesn't support bf16 well
    try:
        cap = torch.cuda.get_device_capability(device)
        return cap[0] >= 8  # Ampere+
    except Exception:
        return False


def unet_dtype(
    device=None,
    model_params=0,
    supported_dtypes=None,
    weight_dtype=None,
) -> torch.dtype:
    """Determine optimal dtype for the diffusion model.

    Checks CLI overrides first, then device capabilities.
    """
    if supported_dtypes is None:
        supported_dtypes = [torch.float16, torch.bfloat16, torch.float32]

    # CLI overrides (highest priority)
    if args.fp32_unet:
        return torch.float32
    if args.fp64_unet:
        return torch.float64
    if args.bf16_unet:
        return torch.bfloat16
    if args.fp16_unet:
        return torch.float16
    if args.fp8_e4m3fn_unet:
        return torch.float8_e4m3fn
    if args.fp8_e5m2_unet:
        return torch.float8_e5m2
    if (
        hasattr(args, "fp8_e8m0fnu_unet")
        and args.fp8_e8m0fnu_unet
        and hasattr(torch, "float8_e8m0fnu")
    ):
        return torch.float8_e8m0fnu

    if device is None:
        device = get_torch_device()

    # Check weight_dtype for fp8
    if weight_dtype is not None and weight_dtype in FLOAT8_TYPES:
        if supports_fp8_compute(device):
            return weight_dtype
        free_mem = get_free_memory(device)
        if model_params * 2 > free_mem:
            return weight_dtype

    if PRIORITIZE_FP16 and torch.float16 in supported_dtypes:
        return torch.float16

    # Try supported dtypes in order
    for dtype in supported_dtypes:
        if dtype == torch.float16 and should_use_fp16(device, model_params):
            return torch.float16
        if dtype == torch.bfloat16 and should_use_bf16(device, model_params):
            return torch.bfloat16

    # Fallback with manual_cast
    for dtype in supported_dtypes:
        if dtype == torch.float16 and should_use_fp16(
            device, model_params, manual_cast=True
        ):
            return torch.float16
        if dtype == torch.bfloat16 and should_use_bf16(
            device, model_params, manual_cast=True
        ):
            return torch.bfloat16

    return torch.float32


def unet_manual_cast(weight_dtype, inference_device, supported_dtypes=None):
    """Return the cast dtype if manual casting is needed, else None."""
    if supported_dtypes is None:
        supported_dtypes = [torch.float16, torch.bfloat16, torch.float32]
    if weight_dtype == torch.float32:
        return None
    fp_dtype = unet_dtype(
        device=inference_device,
        supported_dtypes=supported_dtypes,
    )
    if fp_dtype == weight_dtype:
        return None
    return fp_dtype


def pick_weight_dtype(dtype, fallback_dtype, device=None):
    """Pick *dtype* if supported on *device*, else *fallback_dtype*."""
    if dtype is None:
        return fallback_dtype
    if device is None:
        device = get_torch_device()
    if supports_cast(device, dtype):
        return dtype
    return fallback_dtype


def text_encoder_dtype(device=None) -> torch.dtype:
    """Return optimal dtype for text encoders."""
    if args.fp32_text_enc:
        return torch.float32
    if args.fp16_text_enc:
        return torch.float16
    if args.bf16_text_enc:
        return torch.bfloat16
    if args.fp8_e4m3fn_text_enc:
        return torch.float8_e4m3fn
    if args.fp8_e5m2_text_enc:
        return torch.float8_e5m2
    if device is None:
        device = text_encoder_device()
    if should_use_fp16(device, prioritize_performance=False):
        return torch.float16
    return torch.float32


def vae_dtype(device=None, allowed_dtypes=None) -> torch.dtype:
    """Return optimal dtype for the VAE."""
    if args.fp16_vae:
        return torch.float16
    if args.bf16_vae:
        return torch.bfloat16
    if args.fp32_vae:
        return torch.float32
    return torch.float32


def intermediate_dtype() -> torch.dtype:
    """Return dtype for intermediate node-to-node tensors."""
    if args.fp16_intermediates:
        return torch.float16
    return torch.float32


# ---------------------------------------------------------------------------
# Device placement functions
# ---------------------------------------------------------------------------


def vae_device() -> torch.device:
    """Return device for VAE inference.

    Honors the multi-GPU pin from :func:`set_device_assignment` first,
    then ``args.cpu_vae``, then the default compute device.
    """
    pinned = _device_assignment.get("vae")
    if pinned is not None:
        return pinned
    if args.cpu_vae:
        return torch.device("cpu")
    return get_torch_device()


def vae_offload_device() -> torch.device:
    """Return device for offloading VAE weights."""
    return torch.device("cpu")


def text_encoder_device() -> torch.device:
    """Return device for text encoder inference.

    Multi-GPU pin wins over ``args.gpu_only`` and the fp16 heuristic.
    """
    pinned = _device_assignment.get("text_encoder")
    if pinned is not None:
        return pinned
    if args.gpu_only:
        return get_torch_device()
    if vram_state in (VRAMState.HIGH_VRAM, VRAMState.NORMAL_VRAM):
        if should_use_fp16(prioritize_performance=False):
            return get_torch_device()
        return torch.device("cpu")
    return torch.device("cpu")


def text_encoder_offload_device() -> torch.device:
    """Return device for offloading text encoder weights."""
    return torch.device("cpu")


def text_encoder_initial_device(
    load_device, offload_device, model_size=0
) -> torch.device:
    """Return initial device for loading text encoders."""
    if args.gpu_only:
        return load_device
    if vram_state == VRAMState.HIGH_VRAM:
        return load_device
    return offload_device


def unet_offload_device() -> torch.device:
    """Return device for offloading UNet weights."""
    if vram_state == VRAMState.HIGH_VRAM:
        return get_torch_device()
    return torch.device("cpu")


def unet_inital_load_device(parameters, dtype) -> torch.device:
    """Return initial device for loading UNet weights.

    Multi-GPU pin wins over the HIGH_VRAM heuristic.
    """
    pinned = _device_assignment.get("unet")
    if pinned is not None:
        return pinned
    if vram_state == VRAMState.HIGH_VRAM:
        return get_torch_device()
    return torch.device("cpu")


def intermediate_device() -> torch.device:
    """Return device for intermediate tensors."""
    if args.gpu_only:
        return get_torch_device()
    return torch.device("cpu")


def maximum_vram_for_weights(device=None) -> int:
    """Return maximum VRAM available for model weights."""
    if device is None:
        device = get_torch_device()
    if is_device_cpu(device) or is_device_mps(device):
        return 0
    return get_free_memory(device)


# ---------------------------------------------------------------------------
# LoadedModel tracker
# ---------------------------------------------------------------------------


class LoadedModel:
    """Tracks a model loaded into memory with device placement."""

    def __init__(self, model):
        self._model = model
        self.device = torch.device("cpu")
        self.currently_used = True

    @property
    def model(self):
        return self._model

    def model_memory(self) -> int:
        """Total memory footprint of the model in bytes."""
        model = self._model
        if hasattr(model, "model_size"):
            return model.model_size()
        if hasattr(model, "parameters"):
            return sum(p.nelement() * p.element_size() for p in model.parameters())
        return 0

    def model_memory_required(self, device) -> int:
        """Memory required to load the model onto *device*."""
        return self.model_memory()

    def model_loaded_memory(self) -> int:
        return self.model_memory()

    def model_offloaded_memory(self) -> int:
        return 0

    def model_mmap_residency(self, free=False):
        return (0, 0)

    def model_load(self, lowvram_model_memory=0, force_patch_weights=False):
        """Load the model onto the compute device.

        Routes through :meth:`ModelPatcher.partially_load` so LOW_VRAM
        / NO_VRAM budget semantics are honored in one place.  When
        ``lowvram_model_memory == 0``, we interpret that as "full
        load" (consistent with ComfyUI's convention for NORMAL_VRAM
        and HIGH_VRAM callers).

        For non-ModelPatcher models we fall back to ``model.to(device)``
        (tests, standalone nn.Module fixtures, ...).

        Args:
            lowvram_model_memory: Byte budget for partial load.  0 means
                "load everything".
            force_patch_weights: Unused in Phase 3; reserved for the
                weight-repatch-on-reload flow.
        """
        model = self._model
        device = get_torch_device()
        self.device = device

        if hasattr(model, "partially_load"):
            if lowvram_model_memory and lowvram_model_memory > 0:
                budget = int(lowvram_model_memory)
            else:
                total = (
                    model.model_size() if hasattr(model, "model_size") else 0
                )
                # "Full load" == budget larger than the model.
                budget = max(total + 1, 1 << 62)
            model.partially_load(device, extra_memory=budget)
            return

        if hasattr(model, "to"):
            model.to(device)

    def should_reload_model(self, force_patch_weights=False) -> bool:
        return False

    def model_unload(self, memory_to_free=None, unpatch_weights=True) -> bool:
        """Unload the model back to its offload device.

        For ModelPatcher instances this calls :meth:`partially_unload`
        with a budget large enough to evict everything.  For plain
        nn.Modules we fall back to ``.to("cpu")``.

        Args:
            memory_to_free: Minimum bytes to free.  If ``None`` (the
                default) we evict everything.
            unpatch_weights: Unused in Phase 3.
        """
        model = self._model

        if hasattr(model, "partially_unload"):
            offload = getattr(
                model, "offload_device", torch.device("cpu")
            )
            if memory_to_free is None:
                # Evict everything: pass a budget larger than any model
                budget = 1 << 62
            else:
                budget = int(memory_to_free)
            model.partially_unload(offload, extra_memory=budget)
            self.device = offload
            return True

        if hasattr(model, "to"):
            model.to(torch.device("cpu"))
        self.device = torch.device("cpu")
        return True

    def model_use_more_vram(self, extra_memory, force_patch_weights=False):
        pass

    def is_dead(self) -> bool:
        return self._model is None

    def __eq__(self, other):
        if isinstance(other, LoadedModel):
            return self._model is other._model
        return False


# ---------------------------------------------------------------------------
# Model loading / unloading
# ---------------------------------------------------------------------------


def load_models_gpu(
    models,
    memory_required=0,
    force_patch_weights=False,
    minimum_memory_required=None,
    force_full_load=False,
):
    """Load a list of models onto the compute device.

    The behavior depends on :data:`vram_state`:

    * ``HIGH_VRAM`` / ``NORMAL_VRAM`` — full load via
      :meth:`ModelPatcher.partially_load` with an infinite budget.
    * ``LOW_VRAM`` — partial residency via ``partially_load`` with
      ``budget = minimum_memory_required`` (the caller tells us how
      much fits).  When the caller doesn't pass a budget we fall back
      to half the model size, matching ComfyUI's "fit half" heuristic.
    * ``NO_VRAM``  — budget = 0, nothing moves to the compute device.
    * ``DISABLED`` — the caller shouldn't be invoking us, but we
      treat it as ``NO_VRAM`` defensively.

    Args:
        models: List of :class:`ModelPatcher`-like objects (or raw
            nn.Module fallbacks).
        memory_required: Extra memory headroom the caller wants.
            Currently unused in Phase 3 — reserved for when we add
            real accelerate-based dispatch.
        force_patch_weights: Forwarded to :meth:`LoadedModel.model_load`.
        minimum_memory_required: Byte budget for LOW_VRAM partial load.
        force_full_load: When True, ignore vram_state and force a full
            load (used by explicit high-priority ops).
    """
    global current_loaded_models
    cleanup_models_gc()

    device = get_torch_device()
    models_to_load = []

    for model in models:
        if model is None:
            continue

        already_loaded = False
        for loaded in current_loaded_models:
            if loaded.model is model or (
                hasattr(model, "model") and loaded.model is model.model
            ):
                loaded.currently_used = True
                already_loaded = True
                break

        if not already_loaded:
            models_to_load.append(model)

    if not models_to_load:
        return

    total_mem_needed = (
        sum(m.model_size() if hasattr(m, "model_size") else 0 for m in models_to_load)
        + memory_required
    )

    if total_mem_needed > 0:
        free_memory(
            total_mem_needed, device, keep_loaded=[m for m in models if m is not None]
        )

    # Resolve the per-model byte budget from vram_state.
    for model in models_to_load:
        model_size = (
            model.model_size() if hasattr(model, "model_size") else 0
        )

        if force_full_load or vram_state in (
            VRAMState.HIGH_VRAM,
            VRAMState.NORMAL_VRAM,
            VRAMState.SHARED,
        ):
            budget = 0  # 0 → LoadedModel.model_load interprets as "full"
        elif vram_state == VRAMState.LOW_VRAM:
            if minimum_memory_required and minimum_memory_required > 0:
                budget = int(minimum_memory_required)
            else:
                # ComfyUI's fallback: fit as much as half the model.
                budget = max(1, model_size // 2)
        elif vram_state in (VRAMState.NO_VRAM, VRAMState.DISABLED):
            budget = 1  # positive to bypass the "0 means full" branch,
            # but smaller than any real parameter so nothing moves.
        else:
            budget = 0

        loaded = LoadedModel(model)
        try:
            loaded.model_load(
                lowvram_model_memory=budget,
                force_patch_weights=force_patch_weights,
            )
        except Exception:
            logger.warning("Failed to load model onto device, using CPU fallback")
            loaded.device = torch.device("cpu")
        current_loaded_models.insert(0, loaded)


def load_model_gpu(model):
    """Load a single model onto the compute device."""
    load_models_gpu([model])


def free_memory(
    memory_required,
    device,
    keep_loaded=None,
    for_dynamic=False,
    pins_required=0,
    ram_required=0,
):
    """Free device memory by unloading models.

    Args:
        memory_required: Bytes to free.
        device: Target device.
        keep_loaded: Models to skip.
        for_dynamic: Used for dynamic VRAM.
        pins_required: Pinned memory to free.
        ram_required: RAM to free.

    Returns:
        List of unloaded LoadedModel objects.
    """
    global current_loaded_models
    if keep_loaded is None:
        keep_loaded = []

    unloaded = []
    for loaded in reversed(current_loaded_models[:]):
        if loaded.is_dead():
            current_loaded_models.remove(loaded)
            continue

        if loaded.model in keep_loaded:
            continue

        free = get_free_memory(device)
        if free >= memory_required:
            break

        if loaded.model_unload():
            unloaded.append(loaded)
            current_loaded_models.remove(loaded)

    if unloaded:
        soft_empty_cache()

    return unloaded


def use_more_memory(extra_memory, loaded_models, device):
    """Try to load more model weights into VRAM."""
    for loaded in current_loaded_models:
        if loaded.model in loaded_models:
            loaded.model_use_more_vram(extra_memory)


def offloaded_memory(loaded_models, device) -> int:
    """Return total offloaded memory for given models."""
    total = 0
    for loaded in current_loaded_models:
        if loaded.model in loaded_models:
            total += loaded.model_offloaded_memory()
    return total


def loaded_models(only_currently_used=False):
    """Return list of currently loaded models."""
    if only_currently_used:
        return [lm for lm in current_loaded_models if lm.currently_used]
    return list(current_loaded_models)


def cleanup_models():
    """Remove dead references from loaded models."""
    global current_loaded_models
    current_loaded_models = [lm for lm in current_loaded_models if not lm.is_dead()]


def cleanup_models_gc():
    """Run GC then cleanup dead model references."""
    gc.collect()
    cleanup_models()


def unload_all_models():
    """Unload every loaded model."""
    global current_loaded_models
    for loaded in current_loaded_models:
        loaded.model_unload()
    current_loaded_models.clear()
    soft_empty_cache(force=True)


def soft_empty_cache(force=False):
    """Release GPU cache memory."""
    if cpu_state == CPUState.GPU:
        torch.cuda.empty_cache()
    elif cpu_state == CPUState.MPS:
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()


# ---------------------------------------------------------------------------
# Attention flags
# ---------------------------------------------------------------------------


def xformers_enabled() -> bool:
    return XFORMERS_IS_AVAILABLE and not args.disable_xformers


def xformers_enabled_vae() -> bool:
    return XFORMERS_ENABLED_VAE and not args.disable_xformers


def pytorch_attention_enabled() -> bool:
    return ENABLE_PYTORCH_ATTENTION


def pytorch_attention_enabled_vae() -> bool:
    return pytorch_attention_enabled()


def pytorch_attention_flash_attention() -> bool:
    return args.use_flash_attention


def flash_attention_enabled() -> bool:
    return args.use_flash_attention


def sage_attention_enabled() -> bool:
    return args.use_sage_attention


def force_upcast_attention_dtype() -> Optional[dict]:
    if args.dont_upcast_attention:
        return None
    if args.force_upcast_attention:
        return {
            "q": torch.float32,
            "k": torch.float32,
            "v": torch.float32,
            "out": torch.float32,
        }
    return None


# ---------------------------------------------------------------------------
# Synchronize / interrupt
# ---------------------------------------------------------------------------


def synchronize():
    """Synchronize the compute device."""
    if cpu_state == CPUState.GPU:
        torch.cuda.synchronize()
    elif cpu_state == CPUState.MPS:
        if hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()


class InterruptProcessingException(Exception):
    """Raised when processing is interrupted by the user."""

    pass


def processing_interrupted() -> bool:
    return _interrupt_processing


def interrupt_current_processing(value=True):
    global _interrupt_processing
    with _interrupt_lock:
        _interrupt_processing = value


def throw_exception_if_processing_interrupted():
    if _interrupt_processing:
        interrupt_current_processing(False)
        raise InterruptProcessingException()


# ---------------------------------------------------------------------------
# Tensor utilities
# ---------------------------------------------------------------------------


def pin_memory(tensor):
    """Pin *tensor* in CPU memory for faster transfers."""
    if args.disable_pinned_memory:
        return tensor
    if tensor is not None and tensor.device.type == "cpu":
        try:
            return tensor.pin_memory()
        except Exception:
            pass
    return tensor


def unpin_memory(tensor):
    """No-op — PyTorch handles unpinning on deletion."""
    return tensor


def cast_to_device(tensor, device, dtype, copy=False):
    """Cast *tensor* to *device* and *dtype*."""
    if tensor.device == device and tensor.dtype == dtype:
        if copy:
            return tensor.clone()
        return tensor
    return tensor.to(device=device, dtype=dtype, copy=copy)


def cast_to(
    weight, dtype=None, device=None, non_blocking=False, copy=False, stream=None, r=None
):
    """Cast *weight* to given dtype/device."""
    if device is None and dtype is None:
        return weight
    kw = {}
    if device is not None:
        kw["device"] = device
    if dtype is not None:
        kw["dtype"] = dtype
    kw["non_blocking"] = non_blocking
    kw["copy"] = copy
    return weight.to(**kw)


def extra_reserved_memory() -> int:
    """Return extra reserved memory in bytes."""
    if EXTRA_RESERVED_VRAM > 0:
        return EXTRA_RESERVED_VRAM
    return 400 * 1024 * 1024  # 400 MB default


def minimum_inference_memory() -> int:
    """Return minimum memory needed for inference operations."""
    return 800 * 1024 * 1024 + extra_reserved_memory()


def debug_memory_summary() -> str:
    """Return a string summary of memory usage."""
    device = get_torch_device()
    free = get_free_memory(device)
    total = get_total_memory(device)
    return (
        f"Device: {device}\n"
        f"Total memory: {total / (1024**3):.2f} GB\n"
        f"Free memory: {free / (1024**3):.2f} GB\n"
        f"Loaded models: {len(current_loaded_models)}\n"
    )
