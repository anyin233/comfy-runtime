"""One-shot A/B comparison script.

Runs the OPTIMIZED code path (what's in the tree now) against a NAIVE
reimplementation that mirrors the pre-optimization code. Produces a
markdown report contrasting the two so the wins are auditable.

Run: python -m benchmarks.compare_optimizations
"""

import os
import tempfile

import numpy as np
import torch
from PIL import Image as PILImage

import comfy_runtime  # noqa: F401 — triggers bootstrap
from comfy_runtime import config as cfg_mod
from benchmarks._harness import run_block


# ---------------------------------------------------------------------------
# Step 6 — SaveImage batched .cpu() transfer vs. per-image loop
# ---------------------------------------------------------------------------


def _save_image_naive(images, out_dir, prefix):
    """Pre-Step-6 implementation: per-image .cpu() + np.clip + uint8 cast."""
    for batch_number, image in enumerate(images):
        i = 255.0 * image.cpu().numpy()
        img = PILImage.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        img.save(os.path.join(out_dir, f"{prefix}_{batch_number:03d}.png"))


def _save_image_optimized(images, out_dir, prefix):
    """Step-6 implementation."""
    batch_u8 = (
        images.detach()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .to(dtype=torch.uint8, device="cpu")
    ).numpy()
    for batch_number in range(batch_u8.shape[0]):
        img = PILImage.fromarray(batch_u8[batch_number])
        img.save(os.path.join(out_dir, f"{prefix}_{batch_number:03d}.png"))


# ---------------------------------------------------------------------------
# Step 7 — LoadImage with/without forced RGBA conversion
# ---------------------------------------------------------------------------


def _load_image_naive(path):
    """Pre-Step-7: always convert("RGBA")."""
    img = PILImage.open(path).convert("RGBA")
    image_np = np.array(img).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np)[None,]
    mask = torch.zeros(
        (1, image_np.shape[0], image_np.shape[1]), dtype=torch.float32
    )
    if image_np.shape[2] == 4:
        mask = 1.0 - torch.from_numpy(image_np[:, :, 3])[None,]
    return (image_tensor[:, :, :, :3], mask)


def _load_image_optimized(path):
    """Step-7: conditional RGBA conversion."""
    img = PILImage.open(path)
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


# ---------------------------------------------------------------------------
# Step 8 — Lanczos PIL per-channel loop vs. torch bicubic-antialias
# ---------------------------------------------------------------------------


def _lanczos_naive(samples, width, height):
    """Pre-Step-8: per-sample, per-channel PIL resize."""
    result = []
    for sample in samples:
        channels = []
        for c in range(sample.shape[0]):
            arr = sample[c].cpu().numpy()
            img = PILImage.fromarray(arr.astype(np.float32), mode="F")
            img = img.resize((width, height), PILImage.LANCZOS)
            channels.append(torch.from_numpy(np.array(img)))
        result.append(torch.stack(channels))
    return torch.stack(result).to(samples.device)


def _lanczos_optimized(samples, width, height):
    """Step-8: torch.nn.functional.interpolate bicubic+antialias."""
    from torch.nn.functional import interpolate

    return interpolate(
        samples,
        size=(height, width),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _compare_dispatch():
    """Steps 1 (introspection memoize) + 3 (V1 instance pool) combined.

    A/B by toggling _V3_CHECK_CACHE + _COROUTINE_CACHE + _V1_INSTANCE_POOL
    between normal operation and a forced-miss mode.
    """
    from comfy_runtime import executor

    print("\n## execute_node dispatch (EmptyLatentImage, 512x512)")

    def warm():
        executor.execute_node(
            "EmptyLatentImage", width=512, height=512, batch_size=1
        )

    # Optimized: caches + pool intact.
    dispatch_opt = run_block("dispatch.optimized", warm, warmup=20, iters=2000)

    # Naive: blow away caches on every iteration so each call hits the
    # unoptimized path (re-runs introspection + re-instantiates V1 class).
    cls = executor.get_node_class("EmptyLatentImage")

    def cold():
        executor._V3_CHECK_CACHE.pop(cls, None)
        stale = [k for k in executor._COROUTINE_CACHE if k[0] is cls]
        for k in stale:
            executor._COROUTINE_CACHE.pop(k, None)
        executor._V1_INSTANCE_POOL.pop(cls, None)
        executor.execute_node(
            "EmptyLatentImage", width=512, height=512, batch_size=1
        )

    dispatch_naive = run_block("dispatch.naive", cold, warmup=20, iters=2000)
    _print_row("execute_node dispatch", dispatch_naive, dispatch_opt)


def main():
    tmp = tempfile.mkdtemp(prefix="compare_opt_")
    cfg_mod._LAST_CONFIG = None
    comfy_runtime.configure(output_dir=tmp, input_dir=tmp)

    _compare_dispatch()

    # Step 6's win comes from collapsing N GPU→CPU syncs into one.
    # Force the batch onto CUDA if available so the benchmark reflects
    # the long-running-service scenario (images coming from VAEDecode).
    device = "cuda" if torch.cuda.is_available() else "cpu"
    imgs = torch.rand(4, 128, 128, 3, device=device)

    print(f"## SaveImage (batch=4, 128x128, device={device})")
    save_naive = run_block(
        "save_image.naive",
        lambda: _save_image_naive(imgs, tmp, "naive"),
        warmup=3,
        iters=50,
    )
    save_opt = run_block(
        "save_image.optimized",
        lambda: _save_image_optimized(imgs, tmp, "opt"),
        warmup=3,
        iters=50,
    )
    _print_row("SaveImage (full, incl. PNG)", save_naive, save_opt)

    # Conversion-only micro-bench: isolate the tensor-preparation path
    # that Step 6 actually changed, excluding PNG encoding / disk I/O
    # which are unaffected and drown out the signal.
    print(f"\n## SaveImage tensor conversion ONLY (batch=8, 512x512, device={device})")
    big = torch.rand(8, 512, 512, 3, device=device)

    def _naive_convert():
        out = []
        for image in big:
            i = 255.0 * image.cpu().numpy()
            out.append(np.clip(i, 0, 255).astype(np.uint8))
        return out

    def _opt_convert():
        batch_u8 = (
            big.detach()
            .clamp(0.0, 1.0)
            .mul(255.0)
            .to(dtype=torch.uint8, device="cpu")
        ).numpy()
        return [batch_u8[i] for i in range(batch_u8.shape[0])]

    conv_naive = run_block("save.convert.naive", _naive_convert, warmup=3, iters=50)
    conv_opt = run_block("save.convert.optimized", _opt_convert, warmup=3, iters=50)
    _print_row("SaveImage conversion", conv_naive, conv_opt)

    # Write a pure-RGB PNG for LoadImage
    rgb_arr = (np.random.rand(256, 256, 3) * 255).astype("uint8")
    rgb_path = os.path.join(tmp, "cmp_rgb.png")
    PILImage.fromarray(rgb_arr, "RGB").save(rgb_path)

    print("\n## LoadImage (RGB, 256x256)")
    load_rgb_naive = run_block(
        "load_image.rgb.naive",
        lambda: _load_image_naive(rgb_path),
        warmup=3,
        iters=200,
    )
    load_rgb_opt = run_block(
        "load_image.rgb.optimized",
        lambda: _load_image_optimized(rgb_path),
        warmup=3,
        iters=200,
    )
    _print_row("LoadImage RGB", load_rgb_naive, load_rgb_opt)

    print("\n## Lanczos upscale (latent 64->128, 4 channels)")
    latent = torch.rand(1, 4, 64, 64)
    lanczos_naive = run_block(
        "lanczos.latent.naive",
        lambda: _lanczos_naive(latent, 128, 128),
        warmup=3,
        iters=100,
    )
    lanczos_opt = run_block(
        "lanczos.latent.optimized",
        lambda: _lanczos_optimized(latent, 128, 128),
        warmup=3,
        iters=100,
    )
    _print_row("Lanczos latent 64->128", lanczos_naive, lanczos_opt)

    print("\n## Lanczos upscale (image 3ch 256->512)")
    image = torch.rand(1, 3, 256, 256)
    lanczos_img_naive = run_block(
        "lanczos.image.naive",
        lambda: _lanczos_naive(image, 512, 512),
        warmup=3,
        iters=20,
    )
    lanczos_img_opt = run_block(
        "lanczos.image.optimized",
        lambda: _lanczos_optimized(image, 512, 512),
        warmup=3,
        iters=20,
    )
    _print_row("Lanczos image 256->512", lanczos_img_naive, lanczos_img_opt)


def _print_row(label, naive, opt):
    naive_us = naive["median_ns"] / 1000.0
    opt_us = opt["median_ns"] / 1000.0
    speedup = naive_us / opt_us if opt_us else float("inf")
    print(
        f"  {label:30s}  naive={naive_us:10.2f}us  optimized={opt_us:10.2f}us  "
        f"speedup={speedup:6.2f}x"
    )


if __name__ == "__main__":
    main()
