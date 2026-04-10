"""Built-in node definitions for comfy_runtime.

Phase 1 of the MIT rewrite: CheckpointLoaderSimple, CLIPTextEncode,
KSampler, KSamplerAdvanced, VAEDecode, VAEEncode, EmptyLatentImage call
the MIT compat rewrites directly. LoraLoader, UNETLoader, CLIPLoader,
ControlNetLoader still delegate to the vendor bridge; those are
covered by Tasks 2.4 / 2.5.
"""

import json
import os
import time

import numpy as np
import torch
from PIL import Image
from PIL.PngImagePlugin import PngInfo

# Constants used by comfy_extras nodes
MAX_RESOLUTION = 16384


def _common_ksampler(
    model,
    seed: int,
    steps: int,
    cfg: float,
    sampler_name: str,
    scheduler: str,
    positive,
    negative,
    latent_image,
    denoise: float = 1.0,
    disable_noise: bool = False,
    start_step=None,
    last_step=None,
    force_full_denoise: bool = False,
):
    """Shared SD1.5 sampling path used by :class:`KSampler` and
    :class:`KSamplerAdvanced`.

    Backed by :mod:`comfy_runtime.compat.comfy.samplers`.  Honors a
    deterministic seed for reproducibility, uses the tiny fixture's noise
    shape, and returns a ComfyUI-shaped ``{"samples": ...}`` latent dict.
    """
    from comfy_runtime.compat.comfy import samplers

    latent_samples = latent_image["samples"]

    if disable_noise:
        noise = torch.zeros_like(latent_samples)
    else:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        noise = torch.randn(
            latent_samples.shape,
            generator=generator,
            dtype=latent_samples.dtype,
            device="cpu",
        )

    sigmas = samplers.calculate_sigmas(None, scheduler, steps)

    sampler = samplers.sampler_object(sampler_name)
    out_samples = sampler.sample(
        model=model,
        noise=noise,
        positive=positive,
        negative=negative,
        cfg=cfg,
        latent_image=latent_samples,
        sigmas=sigmas,
        disable_pbar=True,
        seed=seed,
    )

    out = latent_image.copy()
    out["samples"] = out_samples
    return (out,)


# ---------------------------------------------------------------------------
# Loader nodes — delegate to vendor bridge for actual model loading
# ---------------------------------------------------------------------------


class CheckpointLoaderSimple:
    @classmethod
    def INPUT_TYPES(s):
        import folder_paths

        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"
    CATEGORY = "loaders"

    def load_checkpoint(self, ckpt_name):
        import folder_paths
        from comfy_runtime.compat.comfy.sd import load_checkpoint_guess_config

        ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
        out = load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True)
        return out[:3]


class UNETLoader:
    @classmethod
    def INPUT_TYPES(s):
        import folder_paths

        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e5m2"],),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "advanced/loaders"

    def load_unet(self, unet_name, weight_dtype):
        import folder_paths
        from comfy_runtime.compat.comfy.sd import load_unet as _load_unet

        unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
        dtype_map = {
            "fp8_e4m3fn": torch.float8_e4m3fn,
            "fp8_e5m2": torch.float8_e5m2,
        }
        dtype = dtype_map.get(weight_dtype)
        model = _load_unet(unet_path, dtype=dtype)
        return (model,)


class CLIPLoader:
    @classmethod
    def INPUT_TYPES(s):
        import folder_paths

        return {
            "required": {
                "clip_name": (folder_paths.get_filename_list("text_encoders"),),
                "type": (
                    [
                        "stable_diffusion",
                        "stable_cascade",
                        "sd3",
                        "stable_audio",
                        "mochi",
                        "ltxv",
                        "pixart",
                        "cosmos",
                        "lumina2",
                        "wan",
                        "hidream",
                        "chroma",
                        "flux",
                        "flux2",
                        "hunyuan_video",
                        "long_clipl",
                    ],
                ),
            }
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"
    CATEGORY = "advanced/loaders"

    def load_clip(self, clip_name, type):
        import folder_paths
        from comfy_runtime.compat.comfy.sd import load_clip as _load_clip, CLIPType

        clip_path = folder_paths.get_full_path_or_raise("text_encoders", clip_name)

        type_map = {
            "stable_diffusion": CLIPType.SD1,
            "stable_cascade": CLIPType.STABLE_CASCADE,
            "sd3": CLIPType.SD3,
            "ltxv": CLIPType.LTXV,
            "pixart": CLIPType.PIXART,
            "lumina2": CLIPType.LUMINA2,
            "wan": CLIPType.WAN,
            "flux": CLIPType.FLUX,
            "hunyuan_video": CLIPType.HUNYUAN_VIDEO,
            "mochi": CLIPType.MOCHI,
        }
        clip_type = type_map.get(type, CLIPType.SD1)
        clip = _load_clip(clip_path, clip_type=clip_type)
        return (clip,)


class VAELoader:
    @classmethod
    def INPUT_TYPES(s):
        import folder_paths

        return {
            "required": {
                "vae_name": (folder_paths.get_filename_list("vae"),),
            }
        }

    RETURN_TYPES = ("VAE",)
    FUNCTION = "load_vae"
    CATEGORY = "loaders"

    def load_vae(self, vae_name):
        import folder_paths
        from comfy_runtime.compat.comfy.sd import load_vae as _load_vae

        vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)
        vae = _load_vae(vae_path)
        return (vae,)


class LoraLoader:
    @classmethod
    def INPUT_TYPES(s):
        import folder_paths

        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength_model": (
                    "FLOAT",
                    {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01},
                ),
                "strength_clip": (
                    "FLOAT",
                    {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "load_lora"
    CATEGORY = "loaders"

    def load_lora(self, model, clip, lora_name, strength_model, strength_clip):
        import folder_paths
        from comfy_runtime.compat.comfy.sd import load_lora_for_models
        from comfy_runtime.compat.comfy.utils import load_torch_file

        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora_sd = load_torch_file(lora_path)
        model_lora, clip_lora = load_lora_for_models(
            model, clip, lora_sd, strength_model, strength_clip
        )
        return (model_lora, clip_lora)


class ControlNetLoader:
    @classmethod
    def INPUT_TYPES(s):
        import folder_paths

        return {
            "required": {
                "control_net_name": (folder_paths.get_filename_list("controlnet"),),
            }
        }

    RETURN_TYPES = ("CONTROL_NET",)
    FUNCTION = "load_controlnet"
    CATEGORY = "loaders"

    def load_controlnet(self, control_net_name):
        import folder_paths
        from comfy_runtime.compat.comfy.controlnet import load_controlnet

        ckpt_path = folder_paths.get_full_path_or_raise(
            "controlnet", control_net_name
        )
        return (load_controlnet(ckpt_path),)


# ---------------------------------------------------------------------------
# Text encoding — delegates to vendor bridge
# ---------------------------------------------------------------------------


class CLIPTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "dynamic_prompts": True}),
                "clip": ("CLIP",),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "conditioning"

    def encode(self, clip, text):
        tokens = clip.tokenize(text)
        return (clip.encode_from_tokens_scheduled(tokens),)


# ---------------------------------------------------------------------------
# Sampling — delegates to vendor bridge
# ---------------------------------------------------------------------------


class KSampler:
    @classmethod
    def INPUT_TYPES(s):
        import comfy.samplers

        return {
            "required": {
                "model": ("MODEL",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": (
                    "FLOAT",
                    {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1},
                ),
                "sampler_name": (comfy.samplers.SAMPLER_NAMES,),
                "scheduler": (comfy.samplers.SCHEDULER_NAMES,),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "denoise": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling"

    def sample(
        self,
        model,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        denoise=1.0,
    ):
        return _common_ksampler(
            model=model,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            latent_image=latent_image,
            denoise=denoise,
        )


class KSamplerAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        import comfy.samplers

        return {
            "required": {
                "model": ("MODEL",),
                "add_noise": (["enable", "disable"],),
                "noise_seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF},
                ),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (comfy.samplers.SAMPLER_NAMES,),
                "scheduler": (comfy.samplers.SCHEDULER_NAMES,),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                "return_with_leftover_noise": (["disable", "enable"],),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling"

    def sample(
        self,
        model,
        add_noise,
        noise_seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        start_at_step,
        end_at_step,
        return_with_leftover_noise,
        denoise=1.0,
    ):
        disable_noise = add_noise != "enable"
        return _common_ksampler(
            model=model,
            seed=noise_seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            latent_image=latent_image,
            denoise=denoise,
            disable_noise=disable_noise,
        )


# ---------------------------------------------------------------------------
# Latent operations — pure MIT implementations
# ---------------------------------------------------------------------------


class EmptyLatentImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": (
                    "INT",
                    {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8},
                ),
                "height": (
                    "INT",
                    {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8},
                ),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"
    CATEGORY = "latent"

    def generate(self, width, height, batch_size=1):
        latent = torch.zeros([batch_size, 4, height // 8, width // 8], device="cpu")
        return ({"samples": latent},)


class VAEDecode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "latent"

    def decode(self, vae, samples):
        return (vae.decode(samples["samples"]),)


class VAEEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pixels": ("IMAGE",),
                "vae": ("VAE",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "encode"
    CATEGORY = "latent"

    def encode(self, vae, pixels):
        return ({"samples": vae.encode(pixels[:, :, :, :3])},)


class LatentUpscale:
    UPSCALE_METHODS = ["nearest-exact", "bilinear", "area", "bislerp", "lanczos"]
    CROP_METHODS = ["disabled", "center"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT",),
                "upscale_method": (s.UPSCALE_METHODS,),
                "width": (
                    "INT",
                    {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8},
                ),
                "height": (
                    "INT",
                    {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8},
                ),
                "crop": (s.CROP_METHODS,),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "upscale"
    CATEGORY = "latent"

    def upscale(self, samples, upscale_method, width, height, crop):
        import comfy.utils

        s = samples.copy()
        s_samples = samples["samples"]
        s["samples"] = comfy.utils.common_upscale(
            s_samples, width // 8, height // 8, upscale_method, crop
        )
        return (s,)


# ---------------------------------------------------------------------------
# Conditioning operations — pure MIT implementations
# ---------------------------------------------------------------------------


class ConditioningCombine:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning_1": ("CONDITIONING",),
                "conditioning_2": ("CONDITIONING",),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "combine"
    CATEGORY = "conditioning"

    def combine(self, conditioning_1, conditioning_2):
        return (conditioning_1 + conditioning_2,)


class ConditioningSetArea:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "width": (
                    "INT",
                    {"default": 64, "min": 64, "max": MAX_RESOLUTION, "step": 8},
                ),
                "height": (
                    "INT",
                    {"default": 64, "min": 64, "max": MAX_RESOLUTION, "step": 8},
                ),
                "x": (
                    "INT",
                    {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8},
                ),
                "y": (
                    "INT",
                    {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8},
                ),
                "strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "append"
    CATEGORY = "conditioning"

    def append(self, conditioning, width, height, x, y, strength):
        c = []
        for t in conditioning:
            n = [t[0], t[1].copy()]
            n[1]["area"] = (height // 8, width // 8, y // 8, x // 8)
            n[1]["strength"] = strength
            n[1]["set_area_to_bounds"] = False
            c.append(n)
        return (c,)


class ConditioningSetMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "mask": ("MASK",),
                "strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01},
                ),
                "set_cond_area": (["default", "mask bounds"],),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "append"
    CATEGORY = "conditioning"

    def append(self, conditioning, mask, set_cond_area, strength):
        set_area_to_bounds = set_cond_area != "default"
        c = []
        for t in conditioning:
            n = [t[0], t[1].copy()]
            n[1]["mask"] = mask
            n[1]["set_area_to_bounds"] = set_area_to_bounds
            n[1]["mask_strength"] = strength
            c.append(n)
        return (c,)


class ConditioningZeroOut:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning": ("CONDITIONING",)}}

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "zero_out"
    CATEGORY = "advanced/conditioning"

    def zero_out(self, conditioning):
        c = []
        for t in conditioning:
            d = t[1].copy()
            c.append([torch.zeros_like(t[0]), d])
        return (c,)


class SetLatentNoiseMask:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"samples": ("LATENT",), "mask": ("MASK",)}}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "set_mask"
    CATEGORY = "latent/inpaint"

    def set_mask(self, samples, mask):
        s = samples.copy()
        s["noise_mask"] = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))
        return (s,)


# ---------------------------------------------------------------------------
# Image operations — pure MIT implementations
# ---------------------------------------------------------------------------


class LoadImage:
    @classmethod
    def INPUT_TYPES(s):
        import folder_paths

        input_dir = folder_paths.get_input_directory()
        try:
            files = sorted(os.listdir(input_dir))
        except FileNotFoundError:
            files = []
        return {"required": {"image": (files, {"image_upload": True})}}

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"
    CATEGORY = "image"

    def load_image(self, image):
        import folder_paths

        image_path = folder_paths.get_annotated_filepath(image)
        img = Image.open(image_path)
        has_alpha = img.mode in ("RGBA", "LA") or (
            img.mode == "P" and "transparency" in img.info
        )
        if has_alpha:
            img = img.convert("RGBA")
            arr = np.asarray(img, dtype=np.uint8)
            rgb = arr[:, :, :3].astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(np.ascontiguousarray(rgb))[None, ...]
            alpha = arr[:, :, 3].astype(np.float32) / 255.0
            mask = 1.0 - torch.from_numpy(alpha)[None, ...]
        else:
            img = img.convert("RGB")
            arr = np.asarray(img, dtype=np.uint8)
            rgb = arr.astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(np.ascontiguousarray(rgb))[None, ...]
            mask = torch.zeros(
                (1, arr.shape[0], arr.shape[1]), dtype=torch.float32
            )
        return (image_tensor, mask)


class SaveImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "image"

    def save_images(
        self, images, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None
    ):
        import folder_paths

        full_output_folder, filename, counter, subfolder, filename_prefix = (
            folder_paths.get_save_image_path(
                filename_prefix,
                folder_paths.get_output_directory(),
                images.shape[2],
                images.shape[1],
            )
        )
        results = []
        # Batch-level .cpu() transfer: clamp/scale/cast happen as a single
        # fused op, then iterate numpy views for PIL conversion. Uses
        # non-in-place ops so the caller's tensor is untouched.
        batch_u8 = (
            images.detach()
            .clamp(0.0, 1.0)
            .mul(255.0)
            .to(dtype=torch.uint8, device="cpu")
        ).numpy()
        for batch_number in range(batch_u8.shape[0]):
            img = Image.fromarray(batch_u8[batch_number])
            metadata = None
            from comfy.cli_args import args

            if not args.disable_metadata:
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for k, v in extra_pnginfo.items():
                        metadata.add_text(k, json.dumps(v))
            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.png"
            img.save(
                os.path.join(full_output_folder, file),
                pnginfo=metadata,
                compress_level=4,
            )
            results.append({"filename": file, "subfolder": subfolder, "type": "output"})
            counter += 1
        return {"ui": {"images": results}}


class PreviewImage(SaveImage):
    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "image"


class ImageScale:
    UPSCALE_METHODS = ["nearest-exact", "bilinear", "area", "bislerp", "lanczos"]
    CROP_METHODS = ["disabled", "center"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "upscale_method": (s.UPSCALE_METHODS,),
                "width": (
                    "INT",
                    {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1},
                ),
                "height": (
                    "INT",
                    {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1},
                ),
                "crop": (s.CROP_METHODS,),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscaling"

    def upscale(self, image, upscale_method, width, height, crop):
        import comfy.utils

        if width == 0 and height == 0:
            return (image,)
        samples = image.movedim(-1, 1)
        s = comfy.utils.common_upscale(samples, width, height, upscale_method, crop)
        s = s.movedim(1, -1)
        return (s,)


class ImageBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image1": ("IMAGE",), "image2": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "batch"
    CATEGORY = "image"

    def batch(self, image1, image2):
        if image1.shape[1:] != image2.shape[1:]:
            import comfy.utils

            image2 = comfy.utils.common_upscale(
                image2.movedim(-1, 1),
                image1.shape[2],
                image1.shape[1],
                "bilinear",
                "center",
            ).movedim(1, -1)
        return (torch.cat((image1, image2), dim=0),)


class ImageScaleToTotalPixels:
    """Scale an image so its total pixel count matches the requested
    megapixel target.

    Built-in ComfyUI node from ``comfy_extras/nodes_post_processing.py``.
    Workflows (Flux2 in particular) reference it heavily.

    The optional ``resolution_steps`` argument quantizes both the new
    width and height to multiples of that value — newer Flux nodes
    pass ``resolution_steps=64`` so the latent shape is divisible
    by the patch size.
    """

    UPSCALE_METHODS = ["nearest-exact", "bilinear", "area", "bislerp", "lanczos"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "upscale_method": (s.UPSCALE_METHODS,),
                "megapixels": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.01, "max": 16.0, "step": 0.01},
                ),
            },
            "optional": {
                "resolution_steps": (
                    "INT",
                    {"default": 1, "min": 1, "max": 1024, "step": 1},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscaling"

    def upscale(self, image, upscale_method, megapixels, resolution_steps: int = 1):
        import math
        import comfy.utils

        samples = image.movedim(-1, 1)
        total_target = int(megapixels * 1024 * 1024)
        h, w = samples.shape[-2], samples.shape[-1]
        scale = math.sqrt(total_target / (h * w))
        new_h = max(1, round(h * scale))
        new_w = max(1, round(w * scale))

        # Quantize to a multiple of resolution_steps (e.g. 64 for Flux)
        step = max(1, int(resolution_steps))
        new_h = max(step, (new_h // step) * step)
        new_w = max(step, (new_w // step) * step)

        s = comfy.utils.common_upscale(
            samples, new_w, new_h, upscale_method, "disabled"
        )
        s = s.movedim(1, -1)
        return (s,)


# ---------------------------------------------------------------------------
# Register all built-in nodes
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "CheckpointLoaderSimple": CheckpointLoaderSimple,
    "UNETLoader": UNETLoader,
    "CLIPLoader": CLIPLoader,
    "VAELoader": VAELoader,
    "LoraLoader": LoraLoader,
    "ControlNetLoader": ControlNetLoader,
    "CLIPTextEncode": CLIPTextEncode,
    "KSampler": KSampler,
    "KSamplerAdvanced": KSamplerAdvanced,
    "EmptyLatentImage": EmptyLatentImage,
    "VAEDecode": VAEDecode,
    "VAEEncode": VAEEncode,
    "LatentUpscale": LatentUpscale,
    "ConditioningCombine": ConditioningCombine,
    "ConditioningSetArea": ConditioningSetArea,
    "ConditioningSetMask": ConditioningSetMask,
    "ConditioningZeroOut": ConditioningZeroOut,
    "SetLatentNoiseMask": SetLatentNoiseMask,
    "LoadImage": LoadImage,
    "SaveImage": SaveImage,
    "PreviewImage": PreviewImage,
    "ImageScale": ImageScale,
    "ImageScaleToTotalPixels": ImageScaleToTotalPixels,
    "ImageBatch": ImageBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {k: k for k in NODE_CLASS_MAPPINGS}
