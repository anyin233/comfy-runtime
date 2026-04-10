# Slim Vendor Plan Progress

Tracks implementation of `docs/superpowers/plans/2026-04-10-mit-rewrite-vendor.md`.

## Phase 0 Audit — 2026-04-10

### `_vendor` touchpoints (sources that must be rewritten/removed)

| File | Lines | Purpose |
|---|---|---|
| `comfy_runtime/config.py` | 10, 15, 17, 129 | Calls `activate_vendor_bridge` at configure() time |
| `comfy_runtime/compat/comfy/_vendor_bridge.py` | whole file (359 LoC) | The bridge itself |
| `comfy_runtime/compat/nodes.py` | 14 import sites | Imports bridge functions from every node that does inference |

### Bridge API surface (what `compat/nodes.py` currently calls)

From `_vendor_bridge.py`:
- `_ensure_vendor_imports()` — swaps compat for vendor `sys.modules`
- `activate_vendor_bridge()` — public entry point
- `load_checkpoint_guess_config(ckpt_path, ...)`
- `load_unet(unet_path, dtype=None)`
- `load_clip(clip_paths, clip_type=None, model_options=None)`
- `load_vae(vae_path)`
- `encode_clip_text(clip, text)`
- `ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, ...)`
- `vae_decode(vae, samples)`
- `vae_encode(vae, pixels)`

These nine functions are the complete list of Phase-3 replacements needed.

### Documentation-only mentions (no code link)
- `comfy_runtime/compat/comfy_api/latest/_io.py:5` — just a comment asserting this
  file stays MIT-pure.

### Plan exit criterion
All `_vendor` references removed except the docstring comment in `_io.py`, which
can be updated to past-tense at the end of Phase 4.

---
