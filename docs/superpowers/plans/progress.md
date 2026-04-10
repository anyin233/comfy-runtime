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
