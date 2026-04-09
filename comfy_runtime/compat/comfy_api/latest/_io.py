"""MIT reimplementation of the ComfyUI V3 node protocol IO type system.

This module provides a self-contained, MIT-licensed implementation of the
V3 node protocol's IO types, base classes, schema, and node infrastructure.
It does NOT import from ``comfy_runtime._vendor`` or any GPL-licensed code.

Imports are restricted to the Python standard library, torch, and sibling
compat modules.
"""

from __future__ import annotations

import copy
import inspect
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Literal, TypedDict, TypeVar, TYPE_CHECKING

import torch

from comfy_api.internal import _ComfyNodeInternal, _NodeOutputInternal

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def prune_dict(d: dict) -> dict:
    """Return a copy of *d* with all ``None``-valued entries removed."""
    return {k: v for k, v in d.items() if v is not None}


def copy_class(cls):
    """Shallow-copy a class so that modifying it doesn't mutate the original."""
    if cls is None:
        return None
    cls_dict = {
        k: v
        for k, v in cls.__dict__.items()
        if k not in ("__dict__", "__weakref__", "__module__", "__doc__")
    }
    new_cls = type(cls.__name__, (cls,), cls_dict)
    new_cls.__module__ = cls.__module__
    new_cls.__doc__ = cls.__doc__
    return new_cls


def is_class(obj) -> bool:
    """Return ``True`` if *obj* is a class (i.e. an instance of ``type``)."""
    return isinstance(obj, type)


def shallow_clone_class(cls, new_name=None):
    """Create a shallow clone of *cls* that inherits from it."""
    new_name = new_name or f"{cls.__name__}Clone"
    new_bases = (cls,) + cls.__bases__
    return type(new_name, new_bases, dict(cls.__dict__))


class classproperty:
    """Descriptor that acts like ``@property`` but for the class itself."""

    def __init__(self, f):
        self.f = f

    def __get__(self, obj, owner):
        return self.f(owner)


def first_real_override(cls: type, name: str, *, base: type = None):
    """Return the override of *name* on *cls* that is not the base placeholder."""
    if base is None:
        if not hasattr(cls, "GET_BASE_CLASS"):
            raise ValueError("base is required if cls does not have a GET_BASE_CLASS")
        base = cls.GET_BASE_CLASS()
    base_attr = getattr(base, name, None)
    if base_attr is None:
        return None
    base_func = base_attr.__func__
    for c in cls.mro():
        if c is base:
            break
        if name in c.__dict__:
            func = getattr(c, name).__func__
            if func is not base_func:
                return getattr(cls, name)
    return None


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class FolderType(str, Enum):
    """Folder location on the server."""

    input = "input"
    output = "output"
    temp = "temp"


class UploadType(str, Enum):
    """Upload widget types."""

    image = "image_upload"
    audio = "audio_upload"
    video = "video_upload"
    model = "file_upload"


class RemoteOptions:
    """Options for a remote data source on a Combo widget."""

    def __init__(
        self,
        route: str,
        refresh_button: bool,
        control_after_refresh: Literal["first", "last"] = "first",
        timeout: int = None,
        max_retries: int = None,
        refresh: int = None,
    ):
        self.route = route
        """The route to the remote source."""
        self.refresh_button = refresh_button
        """Whether to show a refresh button in the UI below the widget."""
        self.control_after_refresh = control_after_refresh
        """Item to select after refresh: ``"first"`` or ``"last"``."""
        self.timeout = timeout
        """Maximum time (ms) to wait for a response."""
        self.max_retries = max_retries
        """Maximum number of request retries."""
        self.refresh = refresh
        """TTL (ms) between automatic refreshes."""

    def as_dict(self) -> dict:
        """Serialize to dict, omitting ``None`` values."""
        return prune_dict(
            {
                "route": self.route,
                "refresh_button": self.refresh_button,
                "control_after_refresh": self.control_after_refresh,
                "timeout": self.timeout,
                "max_retries": self.max_retries,
                "refresh": self.refresh,
            }
        )


class NumberDisplay(str, Enum):
    """Display mode for numeric widgets."""

    number = "number"
    slider = "slider"
    gradient_slider = "gradientslider"


class ControlAfterGenerate(str, Enum):
    """Behaviour of a numeric widget after a generation finishes."""

    fixed = "fixed"
    increment = "increment"
    decrement = "decrement"
    randomize = "randomize"


# ---------------------------------------------------------------------------
# Base ComfyType hierarchy
# ---------------------------------------------------------------------------


class _ComfyType(ABC):
    """Abstract base for all IO type descriptors."""

    Type = Any
    io_type: str = None


T = TypeVar("T", bound=type)


def comfytype(io_type: str, **kwargs):
    """Decorator that marks a class as a ComfyType with the given *io_type*.

    A ComfyType may define:

    * ``Type`` -- Python type hint for the runtime value.
    * ``class Input(Input): ...`` -- custom input descriptor.
    * ``class Output(Output): ...`` -- custom output descriptor.
    """

    def decorator(cls: T) -> T:
        if isinstance(cls, _ComfyType) or (
            isinstance(cls, type) and issubclass(cls, _ComfyType)
        ):
            # Already a _ComfyType subclass -- clone Input/Output to avoid mutations.
            new_cls = cls
            if hasattr(new_cls, "Input"):
                new_cls.Input = copy_class(new_cls.Input)
            if hasattr(new_cls, "Output"):
                new_cls.Output = copy_class(new_cls.Output)
        else:
            # Wrap in a new class that also inherits from ComfyTypeIO.
            cls_dict = {
                k: v
                for k, v in cls.__dict__.items()
                if k not in ("__dict__", "__weakref__", "__module__", "__doc__")
            }
            new_cls = type(cls.__name__, (cls, ComfyTypeIO), cls_dict)
            new_cls.__module__ = cls.__module__
            new_cls.__doc__ = cls.__doc__

        new_cls.io_type = io_type
        if hasattr(new_cls, "Input") and new_cls.Input is not None:
            new_cls.Input.Parent = new_cls
        if hasattr(new_cls, "Output") and new_cls.Output is not None:
            new_cls.Output.Parent = new_cls
        return new_cls

    return decorator


def Custom(io_type: str) -> type[ComfyTypeIO]:
    """Factory that creates a ``ComfyTypeIO`` for a custom *io_type* string."""

    @comfytype(io_type=io_type)
    class CustomComfyType(ComfyTypeIO): ...

    return CustomComfyType


# ---------------------------------------------------------------------------
# IO base classes: _IO_V3, Input, WidgetInput, Output
# ---------------------------------------------------------------------------


class _IO_V3:
    """Base class for V3 Inputs and Outputs."""

    Parent: _ComfyType = None

    def __init__(self):
        pass

    def validate(self):
        """Validate this IO descriptor (override in subclasses)."""
        pass

    @property
    def io_type(self):
        """The string IO type, delegated to the owning ComfyType."""
        return self.Parent.io_type

    @property
    def Type(self):
        """The Python type hint, delegated to the owning ComfyType."""
        return self.Parent.Type


class Input(_IO_V3):
    """Base class for a V3 Input descriptor."""

    def __init__(
        self,
        id: str,
        display_name: str = None,
        optional=False,
        tooltip: str = None,
        lazy: bool = None,
        extra_dict=None,
        raw_link: bool = None,
        advanced: bool = None,
    ):
        super().__init__()
        self.id = id
        self.display_name = display_name
        self.optional = optional
        self.tooltip = tooltip
        self.lazy = lazy
        self.extra_dict = extra_dict if extra_dict is not None else {}
        self.rawLink = raw_link
        self.advanced = advanced

    def as_dict(self) -> dict:
        """Serialize to a V1-compatible options dict."""
        return prune_dict(
            {
                "display_name": self.display_name,
                "optional": self.optional,
                "tooltip": self.tooltip,
                "lazy": self.lazy,
                "rawLink": self.rawLink,
                "advanced": self.advanced,
            }
        ) | prune_dict(self.extra_dict)

    def get_io_type(self) -> str:
        """Return the string IO type for this input."""
        return self.io_type

    def get_all(self) -> list[Input]:
        """Return a flat list of all concrete inputs (for dynamic types)."""
        return [self]


class WidgetInput(Input):
    """Base class for a V3 Input that renders as a widget."""

    def __init__(
        self,
        id: str,
        display_name: str = None,
        optional=False,
        tooltip: str = None,
        lazy: bool = None,
        default: Any = None,
        socketless: bool = None,
        widget_type: str = None,
        force_input: bool = None,
        extra_dict=None,
        raw_link: bool = None,
        advanced: bool = None,
    ):
        super().__init__(
            id, display_name, optional, tooltip, lazy, extra_dict, raw_link, advanced
        )
        self.default = default
        self.socketless = socketless
        self.widget_type = widget_type
        self.force_input = force_input

    def as_dict(self) -> dict:
        return super().as_dict() | prune_dict(
            {
                "default": self.default,
                "socketless": self.socketless,
                "widgetType": self.widget_type,
                "forceInput": self.force_input,
            }
        )

    def get_io_type(self) -> str:
        return (
            self.widget_type if self.widget_type is not None else super().get_io_type()
        )


class Output(_IO_V3):
    """Base class for a V3 Output descriptor."""

    def __init__(
        self,
        id: str = None,
        display_name: str = None,
        tooltip: str = None,
        is_output_list=False,
    ):
        self.id = id
        self.display_name = display_name if display_name else id
        self.tooltip = tooltip
        self.is_output_list = is_output_list

    def as_dict(self) -> dict:
        display_name = self.display_name if self.display_name else self.id
        return prune_dict(
            {
                "display_name": display_name,
                "tooltip": self.tooltip,
                "is_output_list": self.is_output_list,
            }
        )

    def get_io_type(self) -> str:
        return self.io_type


# ---------------------------------------------------------------------------
# ComfyTypeI / ComfyTypeIO -- convenience bases with default Input/Output
# ---------------------------------------------------------------------------


class ComfyTypeI(_ComfyType):
    """ComfyType subclass that only has a default Input class."""

    class Input(Input): ...


class ComfyTypeIO(ComfyTypeI):
    """ComfyType subclass that has default Input and Output classes."""

    class Output(Output): ...


# ---------------------------------------------------------------------------
# Concrete IO Types
# ---------------------------------------------------------------------------


@comfytype(io_type="BOOLEAN")
class Boolean(ComfyTypeIO):
    """Boolean IO type."""

    Type = bool

    class Input(WidgetInput):
        """Boolean input with optional on/off labels."""

        def __init__(
            self,
            id: str,
            display_name: str = None,
            optional=False,
            tooltip: str = None,
            lazy: bool = None,
            default: bool = None,
            label_on: str = None,
            label_off: str = None,
            socketless: bool = None,
            force_input: bool = None,
            extra_dict=None,
            raw_link: bool = None,
            advanced: bool = None,
        ):
            super().__init__(
                id,
                display_name,
                optional,
                tooltip,
                lazy,
                default,
                socketless,
                None,
                force_input,
                extra_dict,
                raw_link,
                advanced,
            )
            self.label_on = label_on
            self.label_off = label_off
            self.default: bool

        def as_dict(self):
            return super().as_dict() | prune_dict(
                {
                    "label_on": self.label_on,
                    "label_off": self.label_off,
                }
            )


@comfytype(io_type="INT")
class Int(ComfyTypeIO):
    """Integer IO type."""

    Type = int

    class Input(WidgetInput):
        """Integer input with optional range, step, and display mode."""

        def __init__(
            self,
            id: str,
            display_name: str = None,
            optional=False,
            tooltip: str = None,
            lazy: bool = None,
            default: int = None,
            min: int = None,
            max: int = None,
            step: int = None,
            control_after_generate: bool | ControlAfterGenerate = None,
            display_mode: NumberDisplay = None,
            socketless: bool = None,
            force_input: bool = None,
            extra_dict=None,
            raw_link: bool = None,
            advanced: bool = None,
        ):
            super().__init__(
                id,
                display_name,
                optional,
                tooltip,
                lazy,
                default,
                socketless,
                None,
                force_input,
                extra_dict,
                raw_link,
                advanced,
            )
            self.min = min
            self.max = max
            self.step = step
            self.control_after_generate = control_after_generate
            self.display_mode = display_mode
            self.default: int

        def as_dict(self):
            return super().as_dict() | prune_dict(
                {
                    "min": self.min,
                    "max": self.max,
                    "step": self.step,
                    "control_after_generate": self.control_after_generate,
                    "display": self.display_mode.value if self.display_mode else None,
                }
            )


@comfytype(io_type="FLOAT")
class Float(ComfyTypeIO):
    """Float IO type."""

    Type = float

    class Input(WidgetInput):
        """Float input with optional range, step, rounding, and display mode."""

        def __init__(
            self,
            id: str,
            display_name: str = None,
            optional=False,
            tooltip: str = None,
            lazy: bool = None,
            default: float = None,
            min: float = None,
            max: float = None,
            step: float = None,
            round: float = None,
            display_mode: NumberDisplay = None,
            gradient_stops: list[dict] = None,
            socketless: bool = None,
            force_input: bool = None,
            extra_dict=None,
            raw_link: bool = None,
            advanced: bool = None,
        ):
            super().__init__(
                id,
                display_name,
                optional,
                tooltip,
                lazy,
                default,
                socketless,
                None,
                force_input,
                extra_dict,
                raw_link,
                advanced,
            )
            self.min = min
            self.max = max
            self.step = step
            self.round = round
            self.display_mode = display_mode
            self.gradient_stops = gradient_stops
            self.default: float

        def as_dict(self):
            return super().as_dict() | prune_dict(
                {
                    "min": self.min,
                    "max": self.max,
                    "step": self.step,
                    "round": self.round,
                    "display": self.display_mode,
                    "gradient_stops": self.gradient_stops,
                }
            )


@comfytype(io_type="STRING")
class String(ComfyTypeIO):
    """String IO type."""

    Type = str

    class Input(WidgetInput):
        """String input with optional multiline, placeholder, and dynamic prompts."""

        def __init__(
            self,
            id: str,
            display_name: str = None,
            optional=False,
            tooltip: str = None,
            lazy: bool = None,
            multiline=False,
            placeholder: str = None,
            default: str = None,
            dynamic_prompts: bool = None,
            socketless: bool = None,
            force_input: bool = None,
            extra_dict=None,
            raw_link: bool = None,
            advanced: bool = None,
        ):
            super().__init__(
                id,
                display_name,
                optional,
                tooltip,
                lazy,
                default,
                socketless,
                None,
                force_input,
                extra_dict,
                raw_link,
                advanced,
            )
            self.multiline = multiline
            self.placeholder = placeholder
            self.dynamic_prompts = dynamic_prompts
            self.default: str

        def as_dict(self):
            return super().as_dict() | prune_dict(
                {
                    "multiline": self.multiline,
                    "placeholder": self.placeholder,
                    "dynamicPrompts": self.dynamic_prompts,
                }
            )


@comfytype(io_type="COMBO")
class Combo(ComfyTypeIO):
    """Combo (dropdown) IO type."""

    Type = str

    class Input(WidgetInput):
        """Combo input (dropdown)."""

        Type = str

        def __init__(
            self,
            id: str,
            options: list[str] | list[int] | type[Enum] = None,
            display_name: str = None,
            optional=False,
            tooltip: str = None,
            lazy: bool = None,
            default: str | int | Enum = None,
            control_after_generate: bool | ControlAfterGenerate = None,
            upload: UploadType = None,
            image_folder: FolderType = None,
            remote: RemoteOptions = None,
            socketless: bool = None,
            extra_dict=None,
            raw_link: bool = None,
            advanced: bool = None,
        ):
            if isinstance(options, type) and issubclass(options, Enum):
                options = [v.value for v in options]
            if isinstance(default, Enum):
                default = default.value
            super().__init__(
                id,
                display_name,
                optional,
                tooltip,
                lazy,
                default,
                socketless,
                None,
                None,
                extra_dict,
                raw_link,
                advanced,
            )
            self.multiselect = False
            self.options = options
            self.control_after_generate = control_after_generate
            self.upload = upload
            self.image_folder = image_folder
            self.remote = remote
            self.default: str

        def as_dict(self):
            return super().as_dict() | prune_dict(
                {
                    "multiselect": self.multiselect,
                    "options": self.options,
                    "control_after_generate": self.control_after_generate,
                    **({self.upload.value: True} if self.upload is not None else {}),
                    "image_folder": self.image_folder.value
                    if self.image_folder
                    else None,
                    "remote": self.remote.as_dict() if self.remote else None,
                }
            )

    class Output(Output):
        def __init__(
            self,
            id: str = None,
            display_name: str = None,
            options: list[str] = None,
            tooltip: str = None,
            is_output_list=False,
        ):
            super().__init__(id, display_name, tooltip, is_output_list)
            self.options = options if options is not None else []


@comfytype(io_type="COMBO")
class MultiCombo(ComfyTypeI):
    """Multiselect Combo input (dropdown for selecting more than one value)."""

    Type = list[str]

    class Input(Combo.Input):
        def __init__(
            self,
            id: str,
            options: list[str],
            display_name: str = None,
            optional=False,
            tooltip: str = None,
            lazy: bool = None,
            default: list[str] = None,
            placeholder: str = None,
            chip: bool = None,
            control_after_generate: bool | ControlAfterGenerate = None,
            socketless: bool = None,
            extra_dict=None,
            raw_link: bool = None,
            advanced: bool = None,
        ):
            super().__init__(
                id,
                options,
                display_name,
                optional,
                tooltip,
                lazy,
                default,
                control_after_generate,
                socketless=socketless,
                extra_dict=extra_dict,
                raw_link=raw_link,
                advanced=advanced,
            )
            self.multiselect = True
            self.placeholder = placeholder
            self.chip = chip
            self.default: list[str]

        def as_dict(self):
            return super().as_dict() | prune_dict(
                {
                    "multi_select": self.multiselect,
                    "placeholder": self.placeholder,
                    "chip": self.chip,
                }
            )


# --- Tensor-based IO types ---


@comfytype(io_type="IMAGE")
class Image(ComfyTypeIO):
    """Image tensor IO type."""

    Type = torch.Tensor


@comfytype(io_type="WAN_CAMERA_EMBEDDING")
class WanCameraEmbedding(ComfyTypeIO):
    """WAN camera embedding tensor IO type."""

    Type = torch.Tensor


@comfytype(io_type="WEBCAM")
class Webcam(ComfyTypeIO):
    """Webcam IO type."""

    Type = str

    class Input(WidgetInput):
        """Webcam input."""

        Type = str

        def __init__(
            self,
            id: str,
            display_name: str = None,
            optional=False,
            tooltip: str = None,
            lazy: bool = None,
            default: str = None,
            socketless: bool = None,
            extra_dict=None,
            raw_link: bool = None,
            advanced: bool = None,
        ):
            super().__init__(
                id,
                display_name,
                optional,
                tooltip,
                lazy,
                default,
                socketless,
                None,
                None,
                extra_dict,
                raw_link,
                advanced,
            )


@comfytype(io_type="MASK")
class Mask(ComfyTypeIO):
    """Mask tensor IO type."""

    Type = torch.Tensor


@comfytype(io_type="LATENT")
class Latent(ComfyTypeIO):
    """Latents, stored as a dictionary."""

    Type = dict


@comfytype(io_type="CONDITIONING")
class Conditioning(ComfyTypeIO):
    """Conditioning data."""

    Type = list


@comfytype(io_type="SAMPLER")
class Sampler(ComfyTypeIO):
    """Sampler IO type."""

    Type = Any


@comfytype(io_type="SIGMAS")
class Sigmas(ComfyTypeIO):
    """Sigma schedule tensor IO type."""

    Type = torch.Tensor


@comfytype(io_type="NOISE")
class Noise(ComfyTypeIO):
    """Noise IO type."""

    Type = torch.Tensor


@comfytype(io_type="GUIDER")
class Guider(ComfyTypeIO):
    """Guider IO type."""

    Type = Any


@comfytype(io_type="CLIP")
class Clip(ComfyTypeIO):
    """CLIP model IO type."""

    Type = Any


@comfytype(io_type="CONTROL_NET")
class ControlNet(ComfyTypeIO):
    """ControlNet IO type."""

    Type = Any


@comfytype(io_type="VAE")
class Vae(ComfyTypeIO):
    """VAE model IO type."""

    Type = Any


@comfytype(io_type="MODEL")
class Model(ComfyTypeIO):
    """Diffusion model IO type."""

    Type = Any


@comfytype(io_type="CLIP_VISION")
class ClipVision(ComfyTypeIO):
    """CLIP Vision model IO type."""

    Type = Any


@comfytype(io_type="CLIP_VISION_OUTPUT")
class ClipVisionOutput(ComfyTypeIO):
    """CLIP Vision output IO type."""

    Type = Any


@comfytype(io_type="STYLE_MODEL")
class StyleModel(ComfyTypeIO):
    """Style model IO type."""

    Type = Any


@comfytype(io_type="GLIGEN")
class Gligen(ComfyTypeIO):
    """Gligen (grounded language-image generation) IO type."""

    Type = Any


@comfytype(io_type="UPSCALE_MODEL")
class UpscaleModel(ComfyTypeIO):
    """Upscale model IO type."""

    Type = Any


@comfytype(io_type="LATENT_UPSCALE_MODEL")
class LatentUpscaleModel(ComfyTypeIO):
    """Latent upscale model IO type."""

    Type = Any


@comfytype(io_type="AUDIO")
class Audio(ComfyTypeIO):
    """Audio IO type (dict with ``waveform`` and ``sample_rate``)."""

    Type = dict


@comfytype(io_type="VIDEO")
class Video(ComfyTypeIO):
    """Video IO type."""

    Type = Any


@comfytype(io_type="LORA_MODEL")
class LoraModel(ComfyTypeIO):
    """LoRA model weights IO type."""

    Type = dict


@comfytype(io_type="LOSS_MAP")
class LossMap(ComfyTypeIO):
    """Loss map IO type."""

    Type = dict


@comfytype(io_type="HOOKS")
class Hooks(ComfyTypeIO):
    """Hook group IO type."""

    Type = Any


@comfytype(io_type="HOOK_KEYFRAMES")
class HookKeyframes(ComfyTypeIO):
    """Hook keyframe group IO type."""

    Type = Any


@comfytype(io_type="TIMESTEPS_RANGE")
class TimestepsRange(ComfyTypeIO):
    """Range defined by start and end point, between 0.0 and 1.0."""

    Type = tuple


@comfytype(io_type="LATENT_OPERATION")
class LatentOperation(ComfyTypeIO):
    """Latent operation callable IO type."""

    Type = Any


@comfytype(io_type="FLOW_CONTROL")
class FlowControl(ComfyTypeIO):
    """Flow control IO type (used in testing)."""

    Type = tuple


@comfytype(io_type="ACCUMULATION")
class Accumulation(ComfyTypeIO):
    """Accumulation IO type (used in testing)."""

    Type = dict


@comfytype(io_type="AUDIO_ENCODER")
class AudioEncoder(ComfyTypeIO):
    """Audio encoder IO type."""

    Type = Any


@comfytype(io_type="AUDIO_ENCODER_OUTPUT")
class AudioEncoderOutput(ComfyTypeIO):
    """Audio encoder output IO type."""

    Type = Any


@comfytype(io_type="TRACKS")
class Tracks(ComfyTypeIO):
    """Point tracks IO type."""

    Type = dict


@comfytype(io_type="PHOTOMAKER")
class Photomaker(ComfyTypeIO):
    """Photomaker IO type."""

    Type = Any


@comfytype(io_type="POINT")
class Point(ComfyTypeIO):
    """Point IO type."""

    Type = Any


@comfytype(io_type="FACE_ANALYSIS")
class FaceAnalysis(ComfyTypeIO):
    """Face analysis IO type."""

    Type = Any


@comfytype(io_type="BBOX")
class BBOX(ComfyTypeIO):
    """Bounding box IO type."""

    Type = Any


@comfytype(io_type="SEGS")
class SEGS(ComfyTypeIO):
    """Segments IO type."""

    Type = Any


@comfytype(io_type="*")
class AnyType(ComfyTypeIO):
    """Wildcard IO type that matches any other type."""

    Type = Any


@comfytype(io_type="MODEL_PATCH")
class ModelPatch(ComfyTypeIO):
    """Model patch IO type."""

    Type = Any


@comfytype(io_type="LOAD3D_CAMERA")
class Load3DCamera(ComfyTypeIO):
    """3D camera info IO type."""

    Type = dict


@comfytype(io_type="LOAD_3D")
class Load3D(ComfyTypeIO):
    """3D model IO type."""

    Type = dict


@comfytype(io_type="LOAD_3D_ANIMATION")
class Load3DAnimation(Load3D):
    """3D animation model IO type."""

    ...


@comfytype(io_type="SVG")
class SVG(ComfyTypeIO):
    """SVG IO type."""

    Type = Any


@comfytype(io_type="VOXEL")
class Voxel(ComfyTypeIO):
    """Voxel IO type."""

    Type = Any


@comfytype(io_type="MESH")
class Mesh(ComfyTypeIO):
    """Mesh IO type."""

    Type = Any


@comfytype(io_type="FILE_3D")
class File3DAny(ComfyTypeIO):
    """General 3D file type -- accepts any supported 3D format."""

    Type = Any


@comfytype(io_type="FILE_3D_GLB")
class File3DGLB(ComfyTypeIO):
    """GLB format 3D file."""

    Type = Any


@comfytype(io_type="FILE_3D_GLTF")
class File3DGLTF(ComfyTypeIO):
    """GLTF format 3D file."""

    Type = Any


@comfytype(io_type="FILE_3D_FBX")
class File3DFBX(ComfyTypeIO):
    """FBX format 3D file."""

    Type = Any


@comfytype(io_type="FILE_3D_OBJ")
class File3DOBJ(ComfyTypeIO):
    """OBJ format 3D file."""

    Type = Any


@comfytype(io_type="FILE_3D_STL")
class File3DSTL(ComfyTypeIO):
    """STL format 3D file."""

    Type = Any


@comfytype(io_type="FILE_3D_USDZ")
class File3DUSDZ(ComfyTypeIO):
    """USDZ format 3D file."""

    Type = Any


@comfytype(io_type="IMAGECOMPARE")
class ImageCompare(ComfyTypeI):
    """Image comparison widget IO type."""

    Type = dict

    class Input(WidgetInput):
        def __init__(
            self,
            id: str,
            display_name: str = None,
            optional=False,
            tooltip: str = None,
            socketless: bool = True,
            advanced: bool = None,
        ):
            super().__init__(
                id,
                display_name,
                optional,
                tooltip,
                None,
                None,
                socketless,
                None,
                None,
                None,
                None,
                advanced,
            )

        def as_dict(self):
            return super().as_dict()


@comfytype(io_type="COLOR")
class Color(ComfyTypeIO):
    """Colour picker widget IO type."""

    Type = str

    class Input(WidgetInput):
        def __init__(
            self,
            id: str,
            display_name: str = None,
            optional=False,
            tooltip: str = None,
            socketless: bool = True,
            advanced: bool = None,
            default: str = "#ffffff",
        ):
            super().__init__(
                id,
                display_name,
                optional,
                tooltip,
                None,
                default,
                socketless,
                None,
                None,
                None,
                None,
                advanced,
            )
            self.default: str

        def as_dict(self):
            return super().as_dict()


@comfytype(io_type="BOUNDING_BOX")
class BoundingBox(ComfyTypeIO):
    """Bounding box widget IO type."""

    Type = dict

    class Input(WidgetInput):
        def __init__(
            self,
            id: str,
            display_name: str = None,
            optional=False,
            tooltip: str = None,
            socketless: bool = True,
            default: dict = None,
            component: str = None,
            force_input: bool = None,
        ):
            super().__init__(
                id, display_name, optional, tooltip, None, default, socketless
            )
            self.component = component
            self.force_input = force_input
            if default is None:
                self.default = {"x": 0, "y": 0, "width": 512, "height": 512}

        def as_dict(self):
            d = super().as_dict()
            if self.component:
                d["component"] = self.component
            if self.force_input is not None:
                d["forceInput"] = self.force_input
            return d


@comfytype(io_type="CURVE")
class Curve(ComfyTypeIO):
    """Curve editor widget IO type."""

    Type = Any

    class Input(WidgetInput):
        def __init__(
            self,
            id: str,
            display_name: str = None,
            optional=False,
            tooltip: str = None,
            socketless: bool = True,
            default: list[tuple[float, float]] = None,
            advanced: bool = None,
        ):
            super().__init__(
                id,
                display_name,
                optional,
                tooltip,
                None,
                default,
                socketless,
                None,
                None,
                None,
                None,
                advanced,
            )
            if default is None:
                self.default = [(0.0, 0.0), (1.0, 1.0)]

        def as_dict(self):
            d = super().as_dict()
            if self.default is not None:
                d["default"] = {
                    "points": [list(p) for p in self.default],
                    "interpolation": "monotone_cubic",
                }
            return d


@comfytype(io_type="HISTOGRAM")
class Histogram(ComfyTypeIO):
    """Histogram IO type."""

    Type = list


# ---------------------------------------------------------------------------
# Dynamic input infrastructure
# ---------------------------------------------------------------------------


class DynamicInput(Input, ABC):
    """Abstract class for dynamic input registration."""

    pass


class DynamicOutput(Output, ABC):
    """Abstract class for dynamic output registration."""

    pass


def handle_prefix(prefix_list: list[str] | None, id: str | None = None) -> list[str]:
    """Append *id* to *prefix_list* (or create a new list)."""
    if prefix_list is None:
        prefix_list = []
    if id is not None:
        prefix_list = prefix_list + [id]
    return prefix_list


def finalize_prefix(prefix_list: list[str] | None, id: str | None = None) -> str:
    """Join *prefix_list* and *id* with ``'.'``."""
    assert not (prefix_list is None and id is None)
    if prefix_list is None:
        return id
    elif id is not None:
        prefix_list = prefix_list + [id]
    return ".".join(prefix_list)


# ---------------------------------------------------------------------------
# MultiType
# ---------------------------------------------------------------------------


@comfytype(io_type="COMFY_MULTITYPED_V3")
class MultiType:
    """Input that permits more than one input type."""

    Type = Any

    class Input(Input):
        """Input that permits more than one input type.

        If *id* is an instance of ``ComfyType.Input`` then that input is
        used to create a widget (if applicable) with overridden values.
        """

        def __init__(
            self,
            id: str | Input,
            types: list[type[_ComfyType] | _ComfyType],
            display_name: str = None,
            optional=False,
            tooltip: str = None,
            lazy: bool = None,
            extra_dict=None,
            raw_link: bool = None,
            advanced: bool = None,
        ):
            self.input_override = None
            if isinstance(id, Input):
                self.input_override = copy.copy(id)
                optional = id.optional if id.optional is True else optional
                tooltip = id.tooltip if id.tooltip is not None else tooltip
                display_name = (
                    id.display_name if id.display_name is not None else display_name
                )
                lazy = id.lazy if id.lazy is not None else lazy
                id = id.id
                if isinstance(self.input_override, WidgetInput):
                    self.input_override.widget_type = self.input_override.get_io_type()
            super().__init__(
                id,
                display_name,
                optional,
                tooltip,
                lazy,
                extra_dict,
                raw_link,
                advanced,
            )
            self._io_types = types

        @property
        def io_types(self) -> list[type[Input]]:
            """Return list of Input class types permitted."""
            io_types = []
            for x in self._io_types:
                if not is_class(x):
                    io_types.append(type(x))
                else:
                    io_types.append(x)
            return io_types

        def get_io_type(self):
            str_types = [x.io_type for x in self.io_types]
            if self.input_override is not None:
                str_types.insert(0, self.input_override.get_io_type())
            return ",".join(list(dict.fromkeys(str_types)))

        def as_dict(self):
            if self.input_override is not None:
                return self.input_override.as_dict() | super().as_dict()
            else:
                return super().as_dict()


# ---------------------------------------------------------------------------
# MatchType
# ---------------------------------------------------------------------------


@comfytype(io_type="COMFY_MATCHTYPE_V3")
class MatchType(ComfyTypeIO):
    """Type-matching constraint for dynamic IO."""

    class Template:
        """Declares a match-type template with allowed types."""

        def __init__(
            self, template_id: str, allowed_types: _ComfyType | list[_ComfyType] = None
        ):
            self.template_id = template_id
            if allowed_types is None:
                allowed_types = [AnyType]
            if not isinstance(allowed_types, Iterable):
                allowed_types = [allowed_types]
            for t in allowed_types:
                if not isinstance(t, type):
                    if not isinstance(t, _ComfyType):
                        raise ValueError(
                            f"Allowed types must be a ComfyType or a list of ComfyTypes, got {t.__class__.__name__}"
                        )
                else:
                    if not issubclass(t, _ComfyType):
                        raise ValueError(
                            f"Allowed types must be a ComfyType or a list of ComfyTypes, got {t.__name__}"
                        )
            self.allowed_types = allowed_types

        def as_dict(self):
            return {
                "template_id": self.template_id,
                "allowed_types": ",".join([t.io_type for t in self.allowed_types]),
            }

    class Input(Input):
        def __init__(
            self,
            id: str,
            template: "MatchType.Template",
            display_name: str = None,
            optional=False,
            tooltip: str = None,
            lazy: bool = None,
            extra_dict=None,
            raw_link: bool = None,
            advanced: bool = None,
        ):
            super().__init__(
                id,
                display_name,
                optional,
                tooltip,
                lazy,
                extra_dict,
                raw_link,
                advanced,
            )
            self.template = template

        def as_dict(self):
            return super().as_dict() | prune_dict(
                {
                    "template": self.template.as_dict(),
                }
            )

    class Output(Output):
        def __init__(
            self,
            template: "MatchType.Template",
            id: str = None,
            display_name: str = None,
            tooltip: str = None,
            is_output_list=False,
        ):
            if not id and not display_name:
                display_name = "MATCHTYPE"
            super().__init__(id, display_name, tooltip, is_output_list)
            self.template = template

        def as_dict(self):
            return super().as_dict() | prune_dict(
                {
                    "template": self.template.as_dict(),
                }
            )


# ---------------------------------------------------------------------------
# Autogrow
# ---------------------------------------------------------------------------


@comfytype(io_type="COMFY_AUTOGROW_V3")
class Autogrow(ComfyTypeI):
    """Dynamic input that can grow/shrink its slot count."""

    Type = dict
    _MaxNames = 100

    class _AutogrowTemplate:
        """Base for prefix/name autogrow templates."""

        def __init__(self, input: Input):
            assert not isinstance(input, DynamicInput)
            self.input = copy.copy(input)
            if isinstance(self.input, WidgetInput):
                self.input.force_input = True
            self.names: list[str] = []
            self.cached_inputs: dict[str, Input] = {}

        def _create_input(self, input: Input, name: str):
            new_input = copy.copy(self.input)
            new_input.id = name
            return new_input

        def _create_cached_inputs(self):
            for name in self.names:
                self.cached_inputs[name] = self._create_input(self.input, name)

        def get_all(self) -> list[Input]:
            return list(self.cached_inputs.values())

        def as_dict(self):
            return prune_dict(
                {
                    "input": create_input_dict_v1([self.input]),
                }
            )

        def validate(self):
            self.input.validate()

    class TemplatePrefix(_AutogrowTemplate):
        """Autogrow by numeric prefix (e.g. ``img0``, ``img1``, ...)."""

        def __init__(self, input: Input, prefix: str, min: int = 1, max: int = 10):
            super().__init__(input)
            self.prefix = prefix
            assert min >= 0
            assert max >= 1
            assert max <= Autogrow._MaxNames
            self.min = min
            self.max = max
            self.names = [f"{self.prefix}{i}" for i in range(self.max)]
            self._create_cached_inputs()

        def as_dict(self):
            return super().as_dict() | prune_dict(
                {
                    "prefix": self.prefix,
                    "min": self.min,
                    "max": self.max,
                }
            )

    class TemplateNames(_AutogrowTemplate):
        """Autogrow with explicit slot names."""

        def __init__(self, input: Input, names: list[str], min: int = 1):
            super().__init__(input)
            self.names = names[: Autogrow._MaxNames]
            assert min >= 0
            self.min = min
            self._create_cached_inputs()

        def as_dict(self):
            return super().as_dict() | prune_dict(
                {
                    "names": self.names,
                    "min": self.min,
                }
            )

    class Input(DynamicInput):
        def __init__(
            self,
            id: str,
            template: "Autogrow.TemplatePrefix | Autogrow.TemplateNames",
            display_name: str = None,
            optional=False,
            tooltip: str = None,
            lazy: bool = None,
            extra_dict=None,
        ):
            super().__init__(id, display_name, optional, tooltip, lazy, extra_dict)
            self.template = template

        def as_dict(self):
            return super().as_dict() | prune_dict(
                {
                    "template": self.template.as_dict(),
                }
            )

        def get_all(self) -> list[Input]:
            return [self] + self.template.get_all()

        def validate(self):
            self.template.validate()

    @staticmethod
    def _expand_schema_for_dynamic(
        out_dict, live_inputs, value, input_type, curr_prefix
    ):
        """Expand autogrow template into concrete input slots."""
        is_names = "names" in value[1]["template"]
        is_prefix = "prefix" in value[1]["template"]
        input_tmpl = value[1]["template"]["input"]
        if is_names:
            min_count = value[1]["template"]["min"]
            names = value[1]["template"]["names"]
            max_count = len(names)
        elif is_prefix:
            prefix = value[1]["template"]["prefix"]
            min_count = value[1]["template"]["min"]
            max_count = value[1]["template"]["max"]
            names = [f"{prefix}{i}" for i in range(max_count)]
        else:
            return

        template_input = None
        template_required = True
        for _input_type, dict_input in input_tmpl.items():
            if len(dict_input) == 0:
                continue
            template_input = list(dict_input.values())[0]
            template_required = _input_type == "required"
            break
        if template_input is None:
            raise Exception(
                "template_input could not be determined; this should never happen."
            )

        new_dict = {}
        new_dict_added_to = False
        for i, name in enumerate(names):
            expected_id = finalize_prefix(curr_prefix, name)
            if i < min_count and template_required:
                out_dict["required"][expected_id] = template_input
                type_dict = new_dict.setdefault("required", {})
            else:
                out_dict["optional"][expected_id] = template_input
                type_dict = new_dict.setdefault("optional", {})
            if expected_id in live_inputs:
                type_dict[name] = template_input
                new_dict_added_to = True
        if not new_dict_added_to:
            finalized_prefix = finalize_prefix(curr_prefix)
            out_dict["dynamic_paths"][finalized_prefix] = finalized_prefix
            out_dict["dynamic_paths_default_value"][finalized_prefix] = (
                DynamicPathsDefaultValue.EMPTY_DICT
            )
        parse_class_inputs(out_dict, live_inputs, new_dict, curr_prefix)


# ---------------------------------------------------------------------------
# DynamicCombo
# ---------------------------------------------------------------------------


@comfytype(io_type="COMFY_DYNAMICCOMBO_V3")
class DynamicCombo(ComfyTypeI):
    """Combo whose selection dynamically reveals additional inputs."""

    Type = dict

    class Option:
        """One option in a DynamicCombo, with its associated inputs."""

        def __init__(self, key: str, inputs: list[Input]):
            self.key = key
            self.inputs = inputs

        def as_dict(self):
            return {
                "key": self.key,
                "inputs": create_input_dict_v1(self.inputs),
            }

    class Input(DynamicInput):
        def __init__(
            self,
            id: str,
            options: list["DynamicCombo.Option"],
            display_name: str = None,
            optional=False,
            tooltip: str = None,
            lazy: bool = None,
            extra_dict=None,
        ):
            super().__init__(id, display_name, optional, tooltip, lazy, extra_dict)
            self.options = options

        def get_all(self) -> list[Input]:
            return [self] + [inp for option in self.options for inp in option.inputs]

        def as_dict(self):
            return super().as_dict() | prune_dict(
                {
                    "options": [o.as_dict() for o in self.options],
                }
            )

        def validate(self):
            for option in self.options:
                for inp in option.inputs:
                    inp.validate()

    @staticmethod
    def _expand_schema_for_dynamic(
        out_dict, live_inputs, value, input_type, curr_prefix
    ):
        """Expand DynamicCombo into the selected option's inputs."""
        finalized_id = finalize_prefix(curr_prefix)
        if finalized_id in live_inputs:
            key = live_inputs[finalized_id]
            selected_option = None
            options = value[1]["options"]
            for option in options:
                if option["key"] == key:
                    selected_option = option
                    break
            if selected_option is not None:
                parse_class_inputs(
                    out_dict, live_inputs, selected_option["inputs"], curr_prefix
                )
                out_dict[input_type][finalized_id] = value
                out_dict["dynamic_paths"][finalized_id] = finalize_prefix(
                    curr_prefix, curr_prefix[-1]
                )


# ---------------------------------------------------------------------------
# DynamicSlot
# ---------------------------------------------------------------------------


@comfytype(io_type="COMFY_DYNAMICSLOT_V3")
class DynamicSlot(ComfyTypeI):
    """Dynamic slot: when a connection is made, additional inputs appear."""

    Type = dict

    class Input(DynamicInput):
        def __init__(
            self,
            slot: Input,
            inputs: list[Input],
            display_name: str = None,
            tooltip: str = None,
            lazy: bool = None,
            extra_dict=None,
        ):
            assert not isinstance(slot, DynamicInput)
            self.slot = copy.copy(slot)
            self.slot.display_name = (
                slot.display_name if slot.display_name is not None else display_name
            )
            optional = True
            self.slot.tooltip = slot.tooltip if slot.tooltip is not None else tooltip
            self.slot.lazy = slot.lazy if slot.lazy is not None else lazy
            self.slot.extra_dict = (
                slot.extra_dict if slot.extra_dict is not None else extra_dict
            )
            super().__init__(
                slot.id,
                self.slot.display_name,
                optional,
                self.slot.tooltip,
                self.slot.lazy,
                self.slot.extra_dict,
            )
            self.inputs = inputs
            self.force_input = None
            if isinstance(self.slot, WidgetInput):
                self.force_input = True
                self.slot.force_input = True

        def get_all(self) -> list[Input]:
            return [self.slot] + self.inputs

        def as_dict(self):
            return super().as_dict() | prune_dict(
                {
                    "slotType": str(self.slot.get_io_type()),
                    "inputs": create_input_dict_v1(self.inputs),
                    "forceInput": self.force_input,
                }
            )

        def validate(self):
            self.slot.validate()
            for inp in self.inputs:
                inp.validate()

    @staticmethod
    def _expand_schema_for_dynamic(
        out_dict, live_inputs, value, input_type, curr_prefix
    ):
        """Expand DynamicSlot into the slot and its associated inputs."""
        finalized_id = finalize_prefix(curr_prefix)
        if finalized_id in live_inputs:
            inputs = value[1]["inputs"]
            parse_class_inputs(out_dict, live_inputs, inputs, curr_prefix)
            out_dict[input_type][finalized_id] = value
            out_dict["dynamic_paths"][finalized_id] = finalize_prefix(
                curr_prefix, curr_prefix[-1]
            )


# ---------------------------------------------------------------------------
# Dynamic input lookup registry
# ---------------------------------------------------------------------------

DYNAMIC_INPUT_LOOKUP: dict[str, Callable] = {}


def register_dynamic_input_func(io_type: str, func: Callable):
    """Register *func* as the dynamic-schema expander for *io_type*."""
    DYNAMIC_INPUT_LOOKUP[io_type] = func


def get_dynamic_input_func(io_type: str) -> Callable:
    """Retrieve the dynamic-schema expander for *io_type*."""
    return DYNAMIC_INPUT_LOOKUP[io_type]


def setup_dynamic_input_funcs():
    """Register all built-in dynamic input expanders."""
    register_dynamic_input_func(Autogrow.io_type, Autogrow._expand_schema_for_dynamic)
    register_dynamic_input_func(
        DynamicCombo.io_type, DynamicCombo._expand_schema_for_dynamic
    )
    register_dynamic_input_func(
        DynamicSlot.io_type, DynamicSlot._expand_schema_for_dynamic
    )


if len(DYNAMIC_INPUT_LOOKUP) == 0:
    setup_dynamic_input_funcs()


# ---------------------------------------------------------------------------
# V3Data, HiddenHolder, Hidden
# ---------------------------------------------------------------------------


class V3Data(TypedDict):
    """Metadata that accompanies V3 inputs during execution."""

    hidden_inputs: dict[str, Any]
    dynamic_paths: dict[str, Any]
    dynamic_paths_default_value: dict[str, Any]
    create_dynamic_tuple: bool


class Hidden(str, Enum):
    """Enumerator for requesting hidden variables in nodes."""

    unique_id = "UNIQUE_ID"
    """UNIQUE_ID is the unique identifier of the node."""
    prompt = "PROMPT"
    """PROMPT is the complete prompt sent by the client to the server."""
    extra_pnginfo = "EXTRA_PNGINFO"
    """EXTRA_PNGINFO is a dictionary that will be copied into the metadata of any .png files saved."""
    dynprompt = "DYNPROMPT"
    """DYNPROMPT is an instance of comfy_execution.graph.DynamicPrompt."""
    auth_token_comfy_org = "AUTH_TOKEN_COMFY_ORG"
    """AUTH_TOKEN_COMFY_ORG is a token acquired from signing into a ComfyOrg account."""
    api_key_comfy_org = "API_KEY_COMFY_ORG"
    """API_KEY_COMFY_ORG is an API Key generated by ComfyOrg."""


class HiddenHolder:
    """Container for hidden input values, populated during execution."""

    def __init__(
        self,
        unique_id: str,
        prompt: Any,
        extra_pnginfo: Any,
        dynprompt: Any,
        auth_token_comfy_org: str,
        api_key_comfy_org: str,
        **kwargs,
    ):
        self.unique_id = unique_id
        self.prompt = prompt
        self.extra_pnginfo = extra_pnginfo
        self.dynprompt = dynprompt
        self.auth_token_comfy_org = auth_token_comfy_org
        self.api_key_comfy_org = api_key_comfy_org

    def __getattr__(self, key: str):
        """Return ``None`` for any hidden variable not found."""
        return None

    @classmethod
    def from_dict(cls, d: dict | None) -> "HiddenHolder":
        """Create a HiddenHolder from a plain dict of hidden values."""
        if d is None:
            d = {}
        return cls(
            unique_id=d.get(Hidden.unique_id, None),
            prompt=d.get(Hidden.prompt, None),
            extra_pnginfo=d.get(Hidden.extra_pnginfo, None),
            dynprompt=d.get(Hidden.dynprompt, None),
            auth_token_comfy_org=d.get(Hidden.auth_token_comfy_org, None),
            api_key_comfy_org=d.get(Hidden.api_key_comfy_org, None),
        )

    @classmethod
    def from_v3_data(cls, v3_data: V3Data | None) -> "HiddenHolder":
        """Create a HiddenHolder from V3Data."""
        return cls.from_dict(v3_data["hidden_inputs"] if v3_data else None)


# ---------------------------------------------------------------------------
# NodeInfoV1
# ---------------------------------------------------------------------------


@dataclass
class NodeInfoV1:
    """V1-compatible node information, used by the executor."""

    input: dict = None
    input_order: dict[str, list[str]] = None
    is_input_list: bool = None
    output: list[str] = None
    output_is_list: list[bool] = None
    output_name: list[str] = None
    output_tooltips: list[str] = None
    output_matchtypes: list[str] = None
    name: str = None
    display_name: str = None
    description: str = None
    python_module: Any = None
    category: str = None
    output_node: bool = None
    deprecated: bool = None
    experimental: bool = None
    dev_only: bool = None
    api_node: bool = None
    price_badge: dict | None = None
    search_aliases: list[str] = None
    essentials_category: str = None
    has_intermediate_output: bool = None


# ---------------------------------------------------------------------------
# PriceBadge
# ---------------------------------------------------------------------------


@dataclass
class PriceBadgeDepends:
    """Dependencies for client-evaluated pricing badge."""

    widgets: list[str] = field(default_factory=list)
    inputs: list[str] = field(default_factory=list)
    input_groups: list[str] = field(default_factory=list)

    def validate(self) -> None:
        """Validate that all fields are lists of strings."""
        if not isinstance(self.widgets, list) or any(
            not isinstance(x, str) for x in self.widgets
        ):
            raise ValueError("PriceBadgeDepends.widgets must be a list[str].")
        if not isinstance(self.inputs, list) or any(
            not isinstance(x, str) for x in self.inputs
        ):
            raise ValueError("PriceBadgeDepends.inputs must be a list[str].")
        if not isinstance(self.input_groups, list) or any(
            not isinstance(x, str) for x in self.input_groups
        ):
            raise ValueError("PriceBadgeDepends.input_groups must be a list[str].")

    def as_dict(self, schema_inputs: list[Input]) -> dict[str, Any]:
        """Serialize, enriching widget refs with type information."""
        input_types: dict[str, str] = {}
        for inp in schema_inputs:
            all_inputs = inp.get_all()
            input_types[inp.id] = inp.get_io_type()
            for nested_inp in all_inputs[1:]:
                prefixed_id = f"{inp.id}.{nested_inp.id}"
                input_types[prefixed_id] = nested_inp.get_io_type()

        widgets_data: list[dict[str, str]] = []
        for w in self.widgets:
            if w not in input_types:
                raise ValueError(
                    f"PriceBadge depends_on.widgets references unknown widget '{w}'. "
                    f"Available widgets: {list(input_types.keys())}"
                )
            widgets_data.append({"name": w, "type": input_types[w]})

        return {
            "widgets": widgets_data,
            "inputs": self.inputs,
            "input_groups": self.input_groups,
        }


@dataclass
class PriceBadge:
    """Client-evaluated pricing badge declaration."""

    expr: str
    depends_on: PriceBadgeDepends = field(default_factory=PriceBadgeDepends)
    engine: str = field(default="jsonata")

    def validate(self) -> None:
        """Validate the badge expression and its dependencies."""
        if self.engine != "jsonata":
            raise ValueError(
                f"Unsupported PriceBadge.engine '{self.engine}'. Only 'jsonata' is supported."
            )
        if not isinstance(self.expr, str) or not self.expr.strip():
            raise ValueError("PriceBadge.expr must be a non-empty string.")
        self.depends_on.validate()

    def as_dict(self, schema_inputs: list[Input]) -> dict[str, Any]:
        """Serialize the badge for the frontend."""
        return {
            "engine": self.engine,
            "depends_on": self.depends_on.as_dict(schema_inputs),
            "expr": self.expr,
        }


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@dataclass
class Schema:
    """Definition of V3 node properties."""

    node_id: str
    """ID of node -- should be globally unique."""
    display_name: str = None
    """Display name of node."""
    category: str = "sd"
    """The category of the node, as per the ``Add Node`` menu."""
    inputs: list[Input] = field(default_factory=list)
    outputs: list[Output] = field(default_factory=list)
    hidden: list[Hidden] = field(default_factory=list)
    description: str = ""
    """Node description, shown as a tooltip when hovering over the node."""
    search_aliases: list[str] = field(default_factory=list)
    """Alternative names for search."""
    is_input_list: bool = False
    """When True, all inputs become list[type]."""
    is_output_node: bool = False
    """Flags this node as an output node."""
    is_deprecated: bool = False
    """Flags a node as deprecated."""
    is_experimental: bool = False
    """Flags a node as experimental."""
    is_dev_only: bool = False
    """Flags a node as dev-only."""
    is_api_node: bool = False
    """Flags a node as an API node."""
    price_badge: PriceBadge | None = None
    """Optional client-evaluated pricing badge."""
    not_idempotent: bool = False
    """When True, the node will always re-execute."""
    enable_expand: bool = False
    """Flags a node as expandable."""
    accept_all_inputs: bool = False
    """When True, all inputs from the prompt are passed through."""
    essentials_category: str | None = None
    """Optional category for the Essentials tab."""
    has_intermediate_output: bool = False
    """Flags this node as having intermediate output."""

    def validate(self):
        """Validate the schema: verify IDs on inputs and outputs are unique."""
        nested_inputs: list[Input] = []
        for inp in self.inputs:
            if not isinstance(inp, DynamicInput):
                nested_inputs.extend(inp.get_all())
        input_ids = [i.id for i in nested_inputs]
        output_ids = [o.id for o in self.outputs]
        issues: list[str] = []
        if len(set(input_ids)) != len(input_ids):
            dupes = [item for item, count in Counter(input_ids).items() if count > 1]
            issues.append(f"Input ids must be unique, but {dupes} are not.")
        if len(set(output_ids)) != len(output_ids):
            dupes = [item for item, count in Counter(output_ids).items() if count > 1]
            issues.append(f"Output ids must be unique, but {dupes} are not.")
        if issues:
            raise ValueError("\n".join(issues))
        for inp in self.inputs:
            inp.validate()
        for out in self.outputs:
            out.validate()
        if self.price_badge is not None:
            self.price_badge.validate()

    def finalize(self):
        """Add hidden based on selected schema options, and default output IDs."""
        if self.inputs is None:
            self.inputs = []
        if self.outputs is None:
            self.outputs = []
        if self.hidden is None:
            self.hidden = []
        # API nodes need key-related hidden inputs
        if self.is_api_node:
            if Hidden.auth_token_comfy_org not in self.hidden:
                self.hidden.append(Hidden.auth_token_comfy_org)
            if Hidden.api_key_comfy_org not in self.hidden:
                self.hidden.append(Hidden.api_key_comfy_org)
        # Output nodes need prompt and extra_pnginfo
        if self.is_output_node:
            if Hidden.prompt not in self.hidden:
                self.hidden.append(Hidden.prompt)
            if Hidden.extra_pnginfo not in self.hidden:
                self.hidden.append(Hidden.extra_pnginfo)
        # Give outputs without IDs default IDs
        for i, output in enumerate(self.outputs):
            if output.id is None:
                output.id = f"_{i}_{output.io_type}_"

    def get_v1_info(self, cls) -> NodeInfoV1:
        """Build a V1-compatible NodeInfoV1 from this schema."""
        input_dict = create_input_dict_v1(self.inputs)
        if self.hidden:
            for hidden in self.hidden:
                input_dict.setdefault("hidden", {})[hidden.name] = (hidden.value,)

        output = []
        output_is_list = []
        output_name = []
        output_tooltips = []
        output_matchtypes = []
        any_matchtypes = False
        if self.outputs:
            for o in self.outputs:
                output.append(o.io_type)
                output_is_list.append(o.is_output_list)
                output_name.append(o.display_name if o.display_name else o.io_type)
                output_tooltips.append(o.tooltip if o.tooltip else None)
                if isinstance(o, MatchType.Output):
                    output_matchtypes.append(o.template.template_id)
                    any_matchtypes = True
                else:
                    output_matchtypes.append(None)

        if not any_matchtypes:
            output_matchtypes = None

        info = NodeInfoV1(
            input=input_dict,
            input_order={
                key: list(value.keys()) for (key, value) in input_dict.items()
            },
            is_input_list=self.is_input_list,
            output=output,
            output_is_list=output_is_list,
            output_name=output_name,
            output_tooltips=output_tooltips,
            output_matchtypes=output_matchtypes,
            name=self.node_id,
            display_name=self.display_name,
            category=self.category,
            description=self.description,
            output_node=self.is_output_node,
            has_intermediate_output=self.has_intermediate_output,
            deprecated=self.is_deprecated,
            experimental=self.is_experimental,
            dev_only=self.is_dev_only,
            api_node=self.is_api_node,
            python_module=getattr(cls, "RELATIVE_PYTHON_MODULE", "nodes"),
            price_badge=self.price_badge.as_dict(self.inputs)
            if self.price_badge is not None
            else None,
            search_aliases=self.search_aliases if self.search_aliases else None,
            essentials_category=self.essentials_category,
        )
        return info


# ---------------------------------------------------------------------------
# V1 serialisation helpers
# ---------------------------------------------------------------------------


def create_input_dict_v1(inputs: list[Input]) -> dict:
    """Create a V1-compatible ``INPUT_TYPES`` dict from a list of V3 inputs."""
    result = {"required": {}}
    for i in inputs:
        add_to_dict_v1(i, result)
    return result


def add_to_dict_v1(i: Input, d: dict):
    """Add a single V3 Input to a V1-compatible dict."""
    key = "optional" if i.optional else "required"
    as_dict = i.as_dict()
    as_dict.pop("optional", None)
    d.setdefault(key, {})[i.id] = (i.get_io_type(), as_dict)


class DynamicPathsDefaultValue:
    """Sentinel values for dynamic paths when no input was provided."""

    EMPTY_DICT = "empty_dict"


def get_finalized_class_inputs(
    d: dict[str, Any], live_inputs: dict[str, Any], include_hidden=False
):
    """Expand dynamic inputs and return (out_dict, hidden, v3_data)."""
    out_dict = {
        "required": {},
        "optional": {},
        "dynamic_paths": {},
        "dynamic_paths_default_value": {},
    }
    d = d.copy()
    hidden = d.pop("hidden", None)
    parse_class_inputs(out_dict, live_inputs, d)
    if hidden is not None and include_hidden:
        out_dict["hidden"] = hidden

    v3_data: dict[str, Any] = {}
    dynamic_paths = out_dict.pop("dynamic_paths", None)
    if dynamic_paths is not None and len(dynamic_paths) > 0:
        v3_data["dynamic_paths"] = dynamic_paths
    dynamic_paths_default_value = out_dict.pop("dynamic_paths_default_value", None)
    if dynamic_paths_default_value is not None and len(dynamic_paths_default_value) > 0:
        v3_data["dynamic_paths_default_value"] = dynamic_paths_default_value
    return out_dict, hidden, v3_data


def parse_class_inputs(
    out_dict: dict[str, Any],
    live_inputs: dict[str, Any],
    curr_dict: dict[str, Any],
    curr_prefix: list[str] | None = None,
) -> None:
    """Recursively expand inputs, handling dynamic types via lookup."""
    for input_type, inner_d in curr_dict.items():
        for id_, value in inner_d.items():
            io_type = value[0]
            if io_type in DYNAMIC_INPUT_LOOKUP:
                dynamic_input_func = get_dynamic_input_func(io_type)
                new_prefix = handle_prefix(curr_prefix, id_)
                dynamic_input_func(out_dict, live_inputs, value, input_type, new_prefix)
            else:
                finalized_id = finalize_prefix(curr_prefix, id_)
                out_dict[input_type][finalized_id] = value
                if curr_prefix:
                    out_dict["dynamic_paths"][finalized_id] = finalized_id


def build_nested_inputs(values: dict[str, Any], v3_data: V3Data) -> dict[str, Any]:
    """Reconstruct nested dicts from flat dynamic-path inputs."""
    paths = v3_data.get("dynamic_paths", None)
    default_value_dict = v3_data.get("dynamic_paths_default_value", {})
    if paths is None:
        return values
    values = values.copy()
    result: dict[str, Any] = {}
    create_tuple = v3_data.get("create_dynamic_tuple", False)

    for key, path in paths.items():
        parts = path.split(".")
        current = result
        for i, p in enumerate(parts):
            is_last = i == len(parts) - 1
            if is_last:
                value = values.pop(key, None)
                if value is None:
                    default_option = default_value_dict.get(key, None)
                    if default_option == DynamicPathsDefaultValue.EMPTY_DICT:
                        value = {}
                if create_tuple:
                    value = (value, key)
                current[p] = value
            else:
                current = current.setdefault(p, {})

    values.update(result)
    return values


# ---------------------------------------------------------------------------
# ExecutionBlocker
# ---------------------------------------------------------------------------


class ExecutionBlocker:
    """Return this from a node to block downstream execution.

    If *message* is ``None``, execution is blocked silently.
    """

    def __init__(self, message: str = None):
        self.message = message


# ---------------------------------------------------------------------------
# NodeOutput
# ---------------------------------------------------------------------------


class NodeOutput(_NodeOutputInternal):
    """Standardised output of a node.

    Pass any number of positional args (one per output slot) and/or
    keyword arguments for ``ui``, ``expand``, and ``block_execution``.
    """

    def __init__(
        self, *args: Any, ui=None, expand: dict = None, block_execution: str = None
    ):
        self.args = args
        self.ui = ui
        self.expand = expand
        self.block_execution = block_execution

    @property
    def result(self):
        """The output tuple, or ``None`` if there are no outputs."""
        return self.args if len(self.args) > 0 else None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NodeOutput":
        """Create a NodeOutput from a V1-style dict."""
        args = ()
        ui = None
        expand = None
        if "result" in data:
            result = data["result"]
            if isinstance(result, ExecutionBlocker):
                return cls(block_execution=result.message)
            args = result
        if "ui" in data:
            ui = data["ui"]
        if "expand" in data:
            expand = data["expand"]
        return cls(*args, ui=ui, expand=expand)

    def __getitem__(self, index) -> Any:
        return self.args[index]


# ---------------------------------------------------------------------------
# _UIOutput
# ---------------------------------------------------------------------------


class _UIOutput(ABC):
    """Abstract base for structured UI output objects."""

    def __init__(self):
        pass

    @abstractmethod
    def as_dict(self) -> dict:
        """Serialize to a dict suitable for the frontend."""
        ...


# ---------------------------------------------------------------------------
# _ComfyNodeBaseInternal
# ---------------------------------------------------------------------------


class _ComfyNodeBaseInternal(_ComfyNodeInternal):
    """Common base class for storing internal methods and properties.

    DO NOT USE directly for defining nodes -- use ``ComfyNode`` instead.
    """

    RELATIVE_PYTHON_MODULE = None
    SCHEMA = None

    # Filled in during execution
    hidden: HiddenHolder = None

    @classmethod
    @abstractmethod
    def define_schema(cls) -> Schema:
        """Override this function with one that returns a Schema instance."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def execute(cls, **kwargs) -> NodeOutput:
        """Override this function with one that performs node's actions."""
        raise NotImplementedError

    @classmethod
    def validate_inputs(cls, **kwargs) -> bool | str:
        """Optionally define to validate inputs (equivalent to V1 VALIDATE_INPUTS)."""
        raise NotImplementedError

    @classmethod
    def fingerprint_inputs(cls, **kwargs) -> Any:
        """Optionally define to fingerprint inputs (equivalent to V1 IS_CHANGED)."""
        raise NotImplementedError

    @classmethod
    def check_lazy_status(cls, **kwargs) -> list[str]:
        """Return a list of input names that should be evaluated.

        This basic mixin implementation requires all inputs.
        """
        return [name for name in kwargs if kwargs[name] is None]

    def __init__(self):
        self.__class__.VALIDATE_CLASS()

    @classmethod
    def GET_BASE_CLASS(cls):
        return _ComfyNodeBaseInternal

    @classmethod
    def VALIDATE_CLASS(cls):
        """Ensure required methods are defined."""
        if first_real_override(cls, "define_schema") is None:
            raise Exception(
                f"No define_schema function was defined for node class {cls.__name__}."
            )
        if first_real_override(cls, "execute") is None:
            raise Exception(
                f"No execute function was defined for node class {cls.__name__}."
            )

    @classproperty
    def FUNCTION(cls):  # noqa: N805
        """Return the executor function name based on sync/async."""
        if inspect.iscoroutinefunction(cls.execute):
            return "EXECUTE_NORMALIZED_ASYNC"
        return "EXECUTE_NORMALIZED"

    @classmethod
    def EXECUTE_NORMALIZED(cls, *args, **kwargs) -> NodeOutput:
        """Normalise the return value of ``execute`` into a ``NodeOutput``."""
        to_return = cls.execute(*args, **kwargs)
        if to_return is None:
            to_return = NodeOutput()
        elif isinstance(to_return, NodeOutput):
            pass
        elif isinstance(to_return, tuple):
            to_return = NodeOutput(*to_return)
        elif isinstance(to_return, dict):
            to_return = NodeOutput.from_dict(to_return)
        elif isinstance(to_return, ExecutionBlocker):
            to_return = NodeOutput(block_execution=to_return.message)
        else:
            raise Exception(f"Invalid return type from node: {type(to_return)}")
        if to_return.expand is not None and not cls.SCHEMA.enable_expand:
            raise Exception(
                f"Node {cls.__name__} is not expandable, but expand included in NodeOutput; "
                "developer should set enable_expand=True on node's Schema to allow this."
            )
        return to_return

    @classmethod
    async def EXECUTE_NORMALIZED_ASYNC(cls, *args, **kwargs) -> NodeOutput:
        """Async variant of ``EXECUTE_NORMALIZED``."""
        to_return = await cls.execute(*args, **kwargs)
        if to_return is None:
            to_return = NodeOutput()
        elif isinstance(to_return, NodeOutput):
            pass
        elif isinstance(to_return, tuple):
            to_return = NodeOutput(*to_return)
        elif isinstance(to_return, dict):
            to_return = NodeOutput.from_dict(to_return)
        elif isinstance(to_return, ExecutionBlocker):
            to_return = NodeOutput(block_execution=to_return.message)
        else:
            raise Exception(f"Invalid return type from node: {type(to_return)}")
        if to_return.expand is not None and not cls.SCHEMA.enable_expand:
            raise Exception(
                f"Node {cls.__name__} is not expandable, but expand included in NodeOutput; "
                "developer should set enable_expand=True on node's Schema to allow this."
            )
        return to_return

    @classmethod
    def PREPARE_CLASS_CLONE(cls, v3_data: V3Data | None):
        """Create clone of real node class to prevent monkey-patching."""
        c_type = cls if is_class(cls) else type(cls)
        type_clone = shallow_clone_class(c_type)
        type_clone.hidden = HiddenHolder.from_v3_data(v3_data)
        return type_clone

    # ----- V1 Backwards Compatibility -----

    @classmethod
    def GET_NODE_INFO_V1(cls) -> dict[str, Any]:
        """Return a V1-compatible info dict."""
        schema = cls.GET_SCHEMA()
        info = schema.get_v1_info(cls)
        return asdict(info)

    _DESCRIPTION = None

    @classproperty
    def DESCRIPTION(cls):  # noqa: N805
        if cls._DESCRIPTION is None:
            cls.GET_SCHEMA()
        return cls._DESCRIPTION

    _CATEGORY = None

    @classproperty
    def CATEGORY(cls):  # noqa: N805
        if cls._CATEGORY is None:
            cls.GET_SCHEMA()
        return cls._CATEGORY

    _EXPERIMENTAL = None

    @classproperty
    def EXPERIMENTAL(cls):  # noqa: N805
        if cls._EXPERIMENTAL is None:
            cls.GET_SCHEMA()
        return cls._EXPERIMENTAL

    _DEPRECATED = None

    @classproperty
    def DEPRECATED(cls):  # noqa: N805
        if cls._DEPRECATED is None:
            cls.GET_SCHEMA()
        return cls._DEPRECATED

    _DEV_ONLY = None

    @classproperty
    def DEV_ONLY(cls):  # noqa: N805
        if cls._DEV_ONLY is None:
            cls.GET_SCHEMA()
        return cls._DEV_ONLY

    _API_NODE = None

    @classproperty
    def API_NODE(cls):  # noqa: N805
        if cls._API_NODE is None:
            cls.GET_SCHEMA()
        return cls._API_NODE

    _OUTPUT_NODE = None

    @classproperty
    def OUTPUT_NODE(cls):  # noqa: N805
        if cls._OUTPUT_NODE is None:
            cls.GET_SCHEMA()
        return cls._OUTPUT_NODE

    _HAS_INTERMEDIATE_OUTPUT = None

    @classproperty
    def HAS_INTERMEDIATE_OUTPUT(cls):  # noqa: N805
        if cls._HAS_INTERMEDIATE_OUTPUT is None:
            cls.GET_SCHEMA()
        return cls._HAS_INTERMEDIATE_OUTPUT

    _INPUT_IS_LIST = None

    @classproperty
    def INPUT_IS_LIST(cls):  # noqa: N805
        if cls._INPUT_IS_LIST is None:
            cls.GET_SCHEMA()
        return cls._INPUT_IS_LIST

    _OUTPUT_IS_LIST = None

    @classproperty
    def OUTPUT_IS_LIST(cls):  # noqa: N805
        if cls._OUTPUT_IS_LIST is None:
            cls.GET_SCHEMA()
        return cls._OUTPUT_IS_LIST

    _RETURN_TYPES = None

    @classproperty
    def RETURN_TYPES(cls):  # noqa: N805
        if cls._RETURN_TYPES is None:
            cls.GET_SCHEMA()
        return cls._RETURN_TYPES

    _RETURN_NAMES = None

    @classproperty
    def RETURN_NAMES(cls):  # noqa: N805
        if cls._RETURN_NAMES is None:
            cls.GET_SCHEMA()
        return cls._RETURN_NAMES

    _OUTPUT_TOOLTIPS = None

    @classproperty
    def OUTPUT_TOOLTIPS(cls):  # noqa: N805
        if cls._OUTPUT_TOOLTIPS is None:
            cls.GET_SCHEMA()
        return cls._OUTPUT_TOOLTIPS

    _NOT_IDEMPOTENT = None

    @classproperty
    def NOT_IDEMPOTENT(cls):  # noqa: N805
        if cls._NOT_IDEMPOTENT is None:
            cls.GET_SCHEMA()
        return cls._NOT_IDEMPOTENT

    _ACCEPT_ALL_INPUTS = None

    @classproperty
    def ACCEPT_ALL_INPUTS(cls):  # noqa: N805
        if cls._ACCEPT_ALL_INPUTS is None:
            cls.GET_SCHEMA()
        return cls._ACCEPT_ALL_INPUTS

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict]:
        """V1-compatible INPUT_TYPES class method."""
        schema = cls.FINALIZE_SCHEMA()
        info = schema.get_v1_info(cls)
        return info.input

    @classmethod
    def FINALIZE_SCHEMA(cls):
        """Call define_schema and finalize it."""
        schema = cls.define_schema()
        schema.finalize()
        return schema

    @classmethod
    def GET_SCHEMA(cls) -> Schema:
        """Validate node class, finalize schema, validate schema, and set class properties."""
        cls.VALIDATE_CLASS()
        schema = cls.FINALIZE_SCHEMA()
        schema.validate()

        if cls._DESCRIPTION is None:
            cls._DESCRIPTION = schema.description
        if cls._CATEGORY is None:
            cls._CATEGORY = schema.category
        if cls._EXPERIMENTAL is None:
            cls._EXPERIMENTAL = schema.is_experimental
        if cls._DEPRECATED is None:
            cls._DEPRECATED = schema.is_deprecated
        if cls._DEV_ONLY is None:
            cls._DEV_ONLY = schema.is_dev_only
        if cls._API_NODE is None:
            cls._API_NODE = schema.is_api_node
        if cls._OUTPUT_NODE is None:
            cls._OUTPUT_NODE = schema.is_output_node
        if cls._HAS_INTERMEDIATE_OUTPUT is None:
            cls._HAS_INTERMEDIATE_OUTPUT = schema.has_intermediate_output
        if cls._INPUT_IS_LIST is None:
            cls._INPUT_IS_LIST = schema.is_input_list
        if cls._NOT_IDEMPOTENT is None:
            cls._NOT_IDEMPOTENT = schema.not_idempotent
        if cls._ACCEPT_ALL_INPUTS is None:
            cls._ACCEPT_ALL_INPUTS = schema.accept_all_inputs

        if cls._RETURN_TYPES is None:
            output = []
            output_name = []
            output_is_list = []
            output_tooltips = []
            if schema.outputs:
                for o in schema.outputs:
                    output.append(o.io_type)
                    output_name.append(o.display_name if o.display_name else o.io_type)
                    output_is_list.append(o.is_output_list)
                    output_tooltips.append(o.tooltip if o.tooltip else None)
            cls._RETURN_TYPES = output
            cls._RETURN_NAMES = output_name
            cls._OUTPUT_IS_LIST = output_is_list
            cls._OUTPUT_TOOLTIPS = output_tooltips

        cls.SCHEMA = schema
        return schema


# ---------------------------------------------------------------------------
# ComfyNode
# ---------------------------------------------------------------------------


class ComfyNode(_ComfyNodeBaseInternal):
    """Common base class for all V3 nodes.

    Subclass this and implement ``define_schema`` and ``execute``.
    """

    @classmethod
    @abstractmethod
    def define_schema(cls) -> Schema:
        """Override this function with one that returns a Schema instance."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def execute(cls, **kwargs) -> NodeOutput:
        """Override this function with one that performs node's actions."""
        raise NotImplementedError

    @classmethod
    def validate_inputs(cls, **kwargs) -> bool | str:
        """Optionally define to validate inputs (equivalent to V1 VALIDATE_INPUTS)."""
        raise NotImplementedError

    @classmethod
    def fingerprint_inputs(cls, **kwargs) -> Any:
        """Optionally define to fingerprint inputs (equivalent to V1 IS_CHANGED)."""
        raise NotImplementedError

    @classmethod
    def check_lazy_status(cls, **kwargs) -> list[str]:
        """Return a list of input names that should be evaluated.

        This basic mixin implementation requires all inputs.
        """
        return [name for name in kwargs if kwargs[name] is None]

    @classmethod
    def GET_BASE_CLASS(cls):
        """DO NOT override -- will break things in execution.py."""
        return ComfyNode


# ---------------------------------------------------------------------------
# NodeReplace
# ---------------------------------------------------------------------------


class InputMapOldId(TypedDict):
    """Map an old node input to a new node input by ID."""

    new_id: str
    old_id: str


class InputMapSetValue(TypedDict):
    """Set a specific value for a new node input."""

    new_id: str
    set_value: Any


InputMap = InputMapOldId | InputMapSetValue


class OutputMap(TypedDict):
    """Map outputs of node replacement via indexes."""

    new_idx: int
    old_idx: int


class NodeReplace:
    """Defines a possible node replacement, mapping inputs and outputs."""

    def __init__(
        self,
        new_node_id: str,
        old_node_id: str,
        old_widget_ids: list[str] | None = None,
        input_mapping: list[InputMap] | None = None,
        output_mapping: list[OutputMap] | None = None,
    ):
        self.new_node_id = new_node_id
        self.old_node_id = old_node_id
        self.old_widget_ids = old_widget_ids
        self.input_mapping = input_mapping
        self.output_mapping = output_mapping

    def as_dict(self) -> dict:
        """Serialize to a dict."""
        return {
            "new_node_id": self.new_node_id,
            "old_node_id": self.old_node_id,
            "old_widget_ids": self.old_widget_ids,
            "input_mapping": list(self.input_mapping) if self.input_mapping else None,
            "output_mapping": list(self.output_mapping)
            if self.output_mapping
            else None,
        }


# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

__all__ = [
    # Enums / options
    "FolderType",
    "UploadType",
    "RemoteOptions",
    "NumberDisplay",
    "ControlAfterGenerate",
    # Decorator / factory
    "comfytype",
    "Custom",
    # Base IO classes
    "Input",
    "WidgetInput",
    "Output",
    "ComfyTypeI",
    "ComfyTypeIO",
    # Concrete IO types
    "Boolean",
    "Int",
    "Float",
    "String",
    "Combo",
    "MultiCombo",
    "Image",
    "WanCameraEmbedding",
    "Webcam",
    "Mask",
    "Latent",
    "Conditioning",
    "Sampler",
    "Sigmas",
    "Noise",
    "Guider",
    "Clip",
    "ControlNet",
    "Vae",
    "Model",
    "ModelPatch",
    "ClipVision",
    "ClipVisionOutput",
    "AudioEncoder",
    "AudioEncoderOutput",
    "StyleModel",
    "Gligen",
    "UpscaleModel",
    "LatentUpscaleModel",
    "Audio",
    "Video",
    "SVG",
    "LoraModel",
    "LossMap",
    "Voxel",
    "Mesh",
    "File3DAny",
    "File3DGLB",
    "File3DGLTF",
    "File3DFBX",
    "File3DOBJ",
    "File3DSTL",
    "File3DUSDZ",
    "Hooks",
    "HookKeyframes",
    "TimestepsRange",
    "LatentOperation",
    "FlowControl",
    "Accumulation",
    "Load3DCamera",
    "Load3D",
    "Load3DAnimation",
    "Photomaker",
    "Point",
    "FaceAnalysis",
    "BBOX",
    "SEGS",
    "AnyType",
    "Tracks",
    "Color",
    "BoundingBox",
    "Curve",
    "Histogram",
    "ImageCompare",
    # Dynamic types
    "MultiType",
    "MatchType",
    "DynamicCombo",
    "DynamicSlot",
    "Autogrow",
    "DynamicInput",
    "DynamicOutput",
    # Hidden / execution infrastructure
    "HiddenHolder",
    "Hidden",
    "NodeInfoV1",
    "PriceBadgeDepends",
    "PriceBadge",
    "Schema",
    "ComfyNode",
    "NodeOutput",
    "ExecutionBlocker",
    "NodeReplace",
    # V1 helpers
    "V3Data",
    "add_to_dict_v1",
    "create_input_dict_v1",
    "get_finalized_class_inputs",
    "build_nested_inputs",
    "_UIOutput",
]
