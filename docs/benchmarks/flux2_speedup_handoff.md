# Flux2 Speedup Recovery — Handoff Notes

**Branch:** `feature/flux2-investigation` (pushed to `origin`)
**Fork off point:** `main` at commit `d2a76fa` — "Merge branch 'feature/e2e-benchmark': e2e benchmark vs upstream ComfyUI"
**Branch HEAD at handoff:** `c8eaf97` — "wip(flux2): add unload_all_models API + handoff doc (currently a net loss)"
**Commits on branch beyond `main`:** just `c8eaf97` (the WIP). Everything else in this investigation — the dual-import bugfix `fcbc98e` and the Flux2-included benchmark rerun `0bc51f6` — is already merged into `main` at `d2a76fa`.

## Environment assumptions

This document is path-agnostic — the investigation was done on one machine and is being
resumed on another. Where needed, the following placeholders are used:

| Placeholder | Meaning |
|---|---|
| `$REPO` | Absolute path to the `comfy_runtime` repository clone on the current machine |
| `$COMFYUI_PATH` | Absolute path to a sibling clone of upstream `ComfyUI` at tag/commit `0.18.1` (used for the benchmark's upstream comparison) |

The original investigation ran on an Ubuntu host with an RTX 4090 24 GB. Behaviour may
differ on other GPUs, particularly on smaller-VRAM cards where Flux2 may never fit at
1024×1024 without tiling regardless of the fix.

## Status: Stalled — naive `unload_all_models()` is a net LOSS

The primary Flux2 bug (dual-import CLIPType enum → wrong text encoder → matmul shape error)
is **fully fixed and shipped** on commit `fcbc98e` and `0bc51f6`. Flux2 Klein runs end-to-end
and is already included in the published benchmark at `docs/benchmarks/`.

This handoff document is about the **secondary Flux2 performance gap**: in the
published benchmark numbers, comfy_runtime is 1419 ms slower than upstream ComfyUI on
Flux2 (runtime 16696 ms vs comfyui 15277 ms, speedup 0.92x). All other 6 workflows
already show comfy_runtime 1.03x–1.09x faster.

## Observed numbers (from the most recent published benchmark)

| Flux2 stage | runtime | comfyui | gap |
|---|---|---|---|
| model_load   | 2178 ms | 2122 ms | +56 ms |
| text_encode  | 1470 ms | 1548 ms | -78 ms (runtime faster) |
| sample       | 11905 ms | 11196 ms | +709 ms (+6.3%) |
| **decode**   | **971 ms** | **256 ms** | **+715 ms (+279%)** |
| save         | 144 ms | 148 ms | -4 ms |
| **total**    | **16696 ms** | **15277 ms** | **+1419 ms (+9.3%)** |
| GPU peak allocated | 23384 MB | 17514 MB | +5870 MB |

The VAEDecode +279% slowdown is the smoking gun: runtime hits a **23.4 GB GPU peak**
(vs ComfyUI's 17.5 GB), triggering `comfy.sd.VAE.decode`'s **tiled-decoding fallback**.
Log line: `Warning: Ran out of memory when regular VAE decoding, retrying with tiled VAE
decoding.` Tiled decode processes overlapping tiles and is ~4x slower than single-pass.

## Root cause (confirmed via investigation)

ComfyUI's `PromptExecutor` walks the prompt graph via an `execution_list` that calls
`complete_node_execution` for each finished node. That pops the node's entry from
`execution_cache`, which in turn releases Python strong references to the node's intermediate
tensors. When the next node's `load_models_gpu([next_model])` fires, `comfy.model_management`
sees there are no strong refs pinning the upstream model's weights and auto-evicts.

comfy_runtime's direct-call API (`main.py` calling `execute_node` sequentially) has no such
walker. The workflow's `main.py` retains `model`, `clip`, `vae`, `positive`, `negative`,
`guider` etc. as local variables, which keep the upstream ModelPatchers alive via:

```
current_loaded_models[i] (strong) → LoadedModel.model (strong) →
ModelPatcher.model (strong) → Flux2 base model (strong) → layer weights (strong)
```

For SD1.5 (860 M params, ~2 GB peak), this is a cosmetic memory artifact. For Flux2
(4B diffusion + 4B Qwen text encoder ≈ 16 GB of weights alone), it pushes the GPU past
the single-pass VAE decode threshold.

**Verification**: An isolated probe that calls `execute_node` sequentially (no benchmark
harness) shows BOTH runtime and upstream ComfyUI hit exactly `23361 MB peak` for Flux2.
The only reason upstream's BENCHMARK result is 17.5 GB is that the benchmark harness
runs upstream through `PromptExecutor.execute_async`, which has the auto-eviction walker.

Scripts used (all committed to `benchmarks/e2e/profiling/` — no more ephemeral `/tmp/` paths):
- `probe_runtime.py` / `probe_comfyui.py` — step-by-step memory for sd15 (used earlier)
- `flux2_upstream.py` — runs Flux2 entirely through upstream ComfyUI (shows it works and produces correct output)
- `flux2_shapes.py` — dumps conditioning/latent shapes on both sides; showed runtime returns `ZImageTEModel_` with `(1, 16, 2560)` while upstream returns `Flux2TEModel_` with `(1, 512, 7680)`
- `flux2_enum.py` — verifies `CLIPType.FLUX2` value, `detect_te_model` result, and the `==` comparison all agree on both sides at the value level
- `flux2_dual_import.py` — **proves the identity mismatch bug**: shows `from comfy.sd import CLIPType` and `from comfy_runtime._vendor.comfy.sd import CLIPType` return two different enum classes that are not `==`-equal
- `flux2_cliploader_test.py` — verifies the `_vendor_import` fix works: `CLIPLoader(type="flux2")` now returns `Flux2TEModel_`
- `flux2_mem_probe.py` — walks the Flux2 workflow step-by-step on both sides, prints GPU alloc/peak/reserved and `current_loaded_models` after every node. **Key finding:** both sides hit exactly `23361 MB` peak when run via direct `execute_node` calls outside any harness.
- `flux2_fix_test.py` — proves that two `mm.free_memory(1e30, cuda)` calls bring peak down to `11848 MB` and drop decode time from 971 ms to 274 ms
- `flux2_unload_timing.py` — measures each `unload_all_models()` call in isolation (3.3 s each)
- `flux2_unload_profile.py` — **profiles where the 3.3 s goes**: `free_memory(1e30)` takes the whole 3278.8 ms, `empty_cache` alone takes 3.3 ms
- `flux2_fast_unload.py` — the unverified sketch of the fast-unload helper that should replace the current naive path (the investigation was interrupted right as it was about to run)

All scripts now use `Path(__file__).resolve().parents[3]` to find the repo root and honour `COMFYUI_PATH` / `WORKFLOW_MODELS` env vars for paths outside the repo. Run any of them from the repo root with:

```bash
benchmarks/e2e/runtime-env/.venv/bin/python benchmarks/e2e/profiling/flux2_<name>.py
# or, if the probe has a "runtime"/"comfyui" arg:
benchmarks/e2e/runtime-env/.venv/bin/python benchmarks/e2e/profiling/flux2_mem_probe.py runtime
```

## The WIP fix (committed on this branch, NOT WORKING YET)

Attempt: add a public `comfy_runtime.unload_all_models()` helper and call it twice in
`workflows/flux2_klein_text_to_image/main.py` — once after text_encode, once after sampling.

Files on branch:

- `comfy_runtime/memory.py` (new) — exposes `free_memory()` and `unload_all_models()`
  as thin wrappers around `comfy.model_management`
- `comfy_runtime/__init__.py` — exports the new helpers
- `workflows/flux2_klein_text_to_image/main.py` — inserts two `unload_all_models()`
  calls at the boundaries

## Why the WIP fix is a NET LOSS

Direct timing of each unload call:

```
before unload: alloc=7735MB
unload_all_models #1:  3292.0 ms   ← evicting Flux2TEModel_
after unload:  alloc=8MB
sampler:       11854.3 ms
unload_all_models #2:  3102.8 ms   ← evicting Flux2 diffusion model
decode:        274.6 ms            ← GREAT! single-pass, matches upstream
```

Benchmark result with the WIP fix:

```
workflow                  runtime     comfyui    speedup
flux2_klein_text_to_image  23096 ms   15246 ms   0.66x   ← WORSE than before
```

- VAEDecode savings: ~705 ms (971 → 266 ms) ✅
- unload_all_models cost: ~6400 ms (two 3.3s calls) ❌
- Net: **-5700 ms** (worse than before the fix)

## Where the 3.3 seconds per unload goes

Profiled via `benchmarks/e2e/profiling/flux2_unload_profile.py`:

```
A) empty_cache only:                   3.3 ms
B) free_memory(1e30) [moves to CPU]:   3278.8 ms  ← THE PROBLEM
C) empty_cache post-unload:            0.0 ms
```

Traced into `comfy/model_management.py`:
- `free_memory` calls `LoadedModel.model_unload(memory_to_free)`
- `model_unload` calls `self.model.detach(unpatch_weights=True)` (in `model_patcher.py`)
- `detach(unpatch_all=True)` calls `unpatch_model(self.offload_device, unpatch_weights=True)`
- `unpatch_model` at line 937-939 of `model_patcher.py`:

```python
if device_to is not None:
    self.model.to(device_to)       # <-- 3.3 seconds for 8 GB GPU → CPU copy
    self.model.device = device_to
```

The cost is a full **GPU → CPU weight transfer** for the entire 7-8 GB model, plus some
unpatching overhead. For a workflow where we're DONE with the model (will never use it
again this run), the CPU-ward copy is pure waste — we just want to drop the GPU refs
and let `empty_cache` release the pool.

## The next step: fast-unload that skips the CPU copy

The key insight is that `ModelPatcher.detach()` has an `unpatch_all` parameter:

```python
def detach(self, unpatch_all=True):
    self.eject_model()
    self.model_patches_to(self.offload_device)
    if unpatch_all:
        self.unpatch_model(self.offload_device, unpatch_weights=unpatch_all)
    ...
```

With `unpatch_all=False`, the entire expensive `unpatch_model → self.model.to(offload)`
branch is skipped. The weights are left on GPU (not moved to CPU), and the caller is
expected to drop references so Python's GC can free them.

A "fast unload" helper should:

1. Iterate `current_loaded_models` in reverse
2. For each entry matching `device`:
   - Call `lm.model.detach(unpatch_all=False)`  ← SKIPS the 3.3s `.to(cpu)`
   - `lm.model_finalizer.detach(); lm.model_finalizer = None; lm.real_model = None`
3. Pop the entry from `current_loaded_models`
4. Run `gc.collect()` (so the ModelPatcher's refcount can drop if user has already deleted)
5. `torch.cuda.empty_cache()` to release the freed blocks back to the driver

The user-level `main.py` should then `del model; del clip` etc. after calling the helper,
so no strong refs remain and the weights can be reclaimed.

## Test that was about to run

`benchmarks/e2e/profiling/flux2_fast_unload.py` — the unverified sketch of the
fast-unload approach. Was about to execute for the first time when the session
got interrupted. The script is committed but has NOT been run against real
hardware yet. Implementation sketch inside the script:

```python
import comfy.model_management as mm

def fast_unload(device):
    to_pop = []
    for i, lm in enumerate(mm.current_loaded_models):
        if lm.device == device:
            to_pop.append(i)
    for i in sorted(to_pop, reverse=True):
        lm = mm.current_loaded_models.pop(i)
        lm.model.detach(unpatch_all=False)  # KEY: avoid .to(cpu)
        if lm.model_finalizer is not None:
            lm.model_finalizer.detach()
        lm.model_finalizer = None
        lm.real_model = None
```

Expected behaviour: fast_unload should complete in <50 ms (vs 3.3s for unload_all_models).
If that works, plumb it into `comfy_runtime.free_memory()` as the default path, add `del`
statements to the flux2 `main.py`, re-benchmark.

Expected final Flux2 numbers if the fast-unload works as predicted:
- Previously-saved decode time: ~265 ms (vs 971 ms stale)
- Expected unload overhead: ~100 ms total (vs 6400 ms stale)
- Expected total: ~16700 - 705 (decode saved) - 5700 (unload cost eliminated) ≈
  **10200 ms** ← would be 1.50x FASTER than upstream ComfyUI's 15277 ms

Or more conservatively, matching the measured single-pass decode (256 ms), the total
should land somewhere between **14500 and 15000 ms**, making Flux2 roughly
tie with or slightly beat upstream.

## How to resume from scratch (new machine)

Set `$REPO` to wherever you have (or want to clone) `comfy_runtime` on the new machine,
and `$COMFYUI_PATH` to wherever upstream ComfyUI is (or will be) cloned. The benchmark
harness uses `$COMFYUI_PATH` only to run the upstream comparison; if you just want to
iterate on the fix without re-running the full benchmark, you can skip setting it up.

```bash
# 1. Clone comfy_runtime if you don't have it yet
git clone https://github.com/anyin233/comfy-runtime.git $REPO
cd $REPO
git fetch origin
git checkout main

# 2. Create an isolated worktree for this investigation
git worktree add .worktrees/flux2-investigation feature/flux2-investigation
cd .worktrees/flux2-investigation
```

### Vendored comfy tree

`comfy_runtime/_vendor/` is gitignored — it contains a snapshot of upstream ComfyUI's
`comfy/` tree that comfy_runtime ships internally. On the previous machine it existed
under `$REPO/comfy_runtime/_vendor` and the worktree used a symlink to share it.

On the new machine, you need `_vendor/` populated before anything inference-related
works. The three options, in order of ease:

1. **Copy from a working checkout** (fastest if available):
   ```bash
   # If you have another clone that already has _vendor populated (e.g. the parent
   # repo on this machine or a cached copy from the previous machine):
   cp -r /path/to/known-good/comfy_runtime/_vendor comfy_runtime/_vendor
   ```

2. **Symlink from the parent repo** (fastest if you keep multiple worktrees):
   ```bash
   # From inside the worktree:
   ln -s $REPO/comfy_runtime/_vendor comfy_runtime/_vendor
   # Remember to unignore symlinks in .gitignore if they show up in git status —
   # there's already a `_vendor` (no trailing slash) entry in .gitignore for this.
   ```

3. **Regenerate from upstream ComfyUI** (if neither of the above applies): the repo's
   build infrastructure should document how `_vendor` is produced. Look for a
   Makefile, `scripts/sync_vendor.py`, or similar. As of commit `d2a76fa` on main I
   didn't inspect this path closely — check `comfy_runtime/__init__.py` and
   `comfy_runtime/bootstrap.py` for import paths to figure out which upstream version
   was vendored.

### Python environment

```bash
# Rebuild the runtime-env venv for this worktree (editable install of this worktree)
cd benchmarks/e2e/runtime-env && uv sync && cd ../../..

# (Optional) rebuild the comfyui-env venv only if you want to re-run the full
# cross-framework benchmark. For pure fast-unload iteration you don't need it.
# cd benchmarks/e2e/comfyui-env && uv sync && cd ../../..
```

### Model files (required for any real run)

The workflows under `workflows/*/models/` are gitignored. You need to either (a) let
each workflow's `workflow_utils/download_models.py` fetch them from Hugging Face on
first run, or (b) bring them across from the previous machine. Flux2 Klein specifically
needs:

- `workflows/flux2_klein_text_to_image/models/diffusion_models/flux-2-klein-base-4b.safetensors` (~7.7 GB)
- `workflows/flux2_klein_text_to_image/models/text_encoders/qwen_3_4b.safetensors` (~7.5 GB)
- `workflows/flux2_klein_text_to_image/models/vae/flux2-vae.safetensors` (~0.3 GB)

Run `python workflows/flux2_klein_text_to_image/workflow_utils/download_models.py` to
auto-download.

For the other 6 workflows (all SD1.5-based plus ESRGAN), the downloaded files can be
shared across workflows — the benchmark harness historically symlinked each
`workflows/<name>/models/` to a single canonical location, but this is optional.


## Other branches / state to be aware of

- **Main branch** has the first half of the Flux2 work:
  - `fcbc98e` — the dual-import `_vendor_import` fix (the real bug)
  - `0bc51f6` — benchmark report including Flux2 with 0.92x speedup
  These are **already merged and pushed** to `origin/main` (via the earlier session).

- **`feature/flux2-investigation` branch** (pushed to origin at handoff) has:
  - Everything on main, plus
  - `comfy_runtime/memory.py` — public helpers (currently doing the SLOW path)
  - `comfy_runtime/__init__.py` — exports them
  - `workflows/flux2_klein_text_to_image/main.py` — calls `unload_all_models()` twice (currently a net loss)
  - This handoff document

- No benchmark run has been published for the WIP fix — only the single-workflow
  flux2 re-benchmark in `results/latest/` inside the worktree (now removed).

## One-line summary for the next agent

> **Change `comfy_runtime/memory.py` to use `detach(unpatch_all=False)` + pop from
> `current_loaded_models` instead of the stock `mm.free_memory(1e30)` path, to avoid the
> 3.3s-per-call GPU→CPU weight copy that makes the current `unload_all_models()` a net
> loss for Flux2.**
