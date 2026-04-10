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

## Phase 0 Wheel Baseline — 2026-04-10

Build: `python -m build --wheel` with no `_vendor/` populated.

| Metric | Value |
|---|---|
| Wheel path | `dist/comfy_runtime-0.3.1-py3-none-any.whl` |
| Wheel size | **122 KB** |
| Files in wheel | **94** |
| `_vendor/` entries in wheel | **0** (only match: `compat/comfy/_vendor_bridge.py`) |
| Bridge file size in wheel | 12 116 bytes |

### Standalone install smoke test (Phase 0)

```bash
uv venv /tmp/cr-phase0-test --python 3.12
uv pip install --python /tmp/cr-phase0-test/bin/python dist/comfy_runtime-0.3.1-py3-none-any.whl
cd /tmp
/tmp/cr-phase0-test/bin/python -c "
import comfy_runtime
comfy_runtime.configure()
import nodes
print('node count:', len(nodes.NODE_CLASS_MAPPINGS))
"
```

Result: **node count: 23** — configure() completes, shim routes `import nodes`
to `compat/nodes.py`, all 23 built-in node classes register. The vendor bridge
warns once about missing `_vendor` but swallows the exception silently.

**Insight:** the wheel is *structurally* already slim. The work in Phase 1-4 is
to make node bodies (KSampler, CLIPTextEncode, VAEDecode, …) *functionally*
work without the bridge, not to physically shrink the wheel. The wheel stays
~122 KB throughout.

---

## Phase 1 Complete — 2026-04-10

### Unit tests

| Metric | Value |
|---|---|
| Tests passing | **84** (up from baseline 42) |
| New test files | `test_dependencies.py`, `test_tiny_fixture.py`, `test_mit_clip_encode.py`, `test_mit_vae.py`, `test_mit_model_patcher.py`, `test_mit_sampler.py`, `test_mit_sd_loader.py`, `test_mit_nodes_end_to_end.py` |
| Run time | ~19 s |

### New compat/ modules

| File | LoC | Purpose |
|---|---|---|
| `compat/comfy/_tokenizer.py` | 85 | ComfyUI-format tokens from HF tokenizers |
| `compat/comfy/_scheduler_map.py` | 80 | ComfyUI sampler_name → diffusers scheduler |
| `compat/comfy/_diffusers_loader.py` | 85 | SD1.5 single-file → diffusers modules |

### Methods implemented (replacing Phase-3 stubs)

* `CLIP.tokenize` (transformers CLIPTokenizer)
* `CLIP.encode_from_tokens` (CLIPTextModel forward)
* `CLIP.encode_from_tokens_scheduled`
* `VAE.encode` (AutoencoderKL.encode)
* `VAE.decode` (AutoencoderKL.decode)
* `ModelPatcher.patch_model` (real weight delta application + backup)
* `ModelPatcher.unpatch_model` (restore from backup)
* `sampler_object()` (returns KSAMPLER instance)
* `KSAMPLER.sample` (diffusers scheduler loop + batched CFG)
* `load_checkpoint_guess_config` (SD1.5 → tuple)
* `_common_ksampler` helper used by `KSampler`/`KSamplerAdvanced`

### Wheel metrics — Phase 1

| Metric | Phase 0 | Phase 1 | Δ |
|---|---|---|---|
| Wheel size | 122 KB | **131 KB** | +9 KB (new helper files) |
| Files in wheel | 94 | 97 | +3 |
| `_vendor` entries | 1 (bridge) | 1 (bridge) | 0 |

### Standalone wheel smoke test — Phase 1

Command: `COMFY_RUNTIME_TEST_WHEEL=1 pytest tests/integration/test_wheel_standalone.py`

```
build wheel → uv venv /tmp/venv --python 3.12
uv pip install --python /tmp/venv/bin/python dist/*.whl
/tmp/venv/bin/python -c "
  CLIPTextEncode → KSampler → VAEDecode through the MIT compat layer
  with a random-init diffusers UNet/VAE/CLIP built in-process.
"
```

Result: **PASSED in 13.36 s**.  Output: `OK image=(1, 16, 16, 3)`.

Proves: a fresh venv with `pip install comfy-runtime` alone runs the SD1.5
happy path without the `_vendor_bridge` fallback kicking in and without any
GPL code on the machine.

### Still on the bridge (Phase 2-4 targets)

`compat/nodes.py` nodes that still delegate to `_vendor_bridge`:

| Node | Target phase |
|---|---|
| `UNETLoader` | Phase 2 (Task 2.2 Flux) |
| `CLIPLoader` | Phase 2 (Task 2.3 SDXL dual-encoder) |
| `LoraLoader` | Phase 2 (Task 2.4 peft integration) |
| `ControlNetLoader` / `ControlNetApply*` | Phase 2 (Task 2.5) |

7 of 23 nodes are MIT-pure after Phase 1:
`CheckpointLoaderSimple`, `CLIPTextEncode`, `KSampler`, `KSamplerAdvanced`,
`EmptyLatentImage`, `VAEDecode`, `VAEEncode`.

---

## Phase 2 Complete — 2026-04-10

### Unit tests

| Metric | Value |
|---|---|
| Tests passing | **100** (+16 from Phase 1's 84) |
| New test files | `test_mit_lora.py` (10), `test_mit_standalone_loaders.py` (6) |
| Run time | ~23 s |

### New compat/ modules

| File | LoC | Purpose |
|---|---|---|
| `compat/comfy/_lora_peft.py` | 125 | Kohya LoRA → ModelPatcher deltas |

### Methods implemented in Phase 2

* `extract_lora_deltas(lora_sd)` — parses `.lora_up`/`.lora_down`/`.alpha`
  triples into raw delta tensors
* `apply_lora_to_patcher(patcher, lora_sd, strength)` — registers deltas
  via `ModelPatcher.add_patches`
* `sd.load_lora_for_models(model, clip, lora, strength_m, strength_c)` —
  clone-and-register (non-mutating)
* `sd.load_vae(vae_path)` — standalone VAE file loader
* `sd.load_clip(clip_path, clip_type)` — standalone CLIP file loader
* `sd.load_unet(unet_path, dtype=None)` — standalone UNet → ModelPatcher

### `_vendor_bridge.py` DELETED

All 9 bridge functions reimplemented in compat.  `_vendor_bridge.py`
(359 LoC) removed.  `config.py::_activate_vendor_bridge_if_available()`
removed.  `test_configure_idempotent.py` rewritten to verify short-
circuit via `_LAST_CONFIG` directly instead of spying on the bridge.

**All 23 built-in nodes now either MIT-pure or raise NotImplementedError
with a clear pointer to the Phase-2.5 ControlNet task.**

### Wheel metrics — Phase 2

| Metric | Phase 0 | Phase 1 | Phase 2 | Δ from Phase 1 |
|---|---|---|---|---|
| Wheel size | 122 KB | 131 KB | **131 KB** | 0 |
| Files in wheel | 94 | 97 | **97** | 0 |
| `_vendor` entries | 1 (bridge) | 1 (bridge) | **0** | **−1 (bridge gone)** |

### Standalone wheel smoke tests — Phase 2

3 tests all PASSED in 26 s:

1. `test_wheel_builds_installs_and_runs_sd15_happy_path` — SD1.5 txt2img
   pipeline through the installed wheel.
2. `test_wheel_lora_roundtrip_in_fresh_venv` — LoRA apply + weight
   mutation + unpatch restore, all from site-packages.
3. `test_wheel_has_no_vendor_entries` — wheel zip inventory has zero
   `_vendor` paths.

Proof: ComfyUI optimization parity (weight hot-swap via ModelPatcher +
LoRA stacking + CLIP/VAE/UNet standalone loaders) works from an
installed wheel with no GPL code on the host.

### Still on the bridge after Phase 2

**None.**  `_vendor_bridge.py` is gone.  `ControlNetLoader` raises
`NotImplementedError` pointing at Task 2.5.

---
