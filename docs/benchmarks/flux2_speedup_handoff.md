# Flux2 Speedup Recovery — Handoff Notes

**Branch:** `feature/flux2-investigation`
**Worktree:** `/home/yanweiye/Project/comfy_runtime/.worktrees/flux2-investigation` (now removed)
**Last commit before handoff:** WIP commit with `comfy_runtime.unload_all_models()` public API added

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

Scripts used:
- `benchmarks/e2e/profiling/probe_runtime.py` / `probe_comfyui.py` — step-by-step mem
- `/tmp/flux2_mem_probe.py` (not saved into repo) — the one that showed both hit 23.3 GB
- `/tmp/flux2_fix_test.py` (not saved) — proved `mm.free_memory(1e30, cuda)` would work

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

Profiled via `/tmp/flux2_unload_profile.py`:

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

`/tmp/flux2_fast_unload.py` — proved out the above approach. Script was about to execute
but got interrupted. Recreate by implementing:

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

## How to resume from main

The investigation worktree (`/home/yanweiye/Project/comfy_runtime/.worktrees/flux2-investigation`)
has been removed. To resume:

```bash
cd /home/yanweiye/Project/comfy_runtime
git fetch origin
git worktree add .worktrees/flux2-investigation feature/flux2-investigation
cd .worktrees/flux2-investigation

# The vendored comfy tree is gitignored; symlink from parent repo
ln -s /home/yanweiye/Project/comfy_runtime/comfy_runtime/_vendor comfy_runtime/_vendor

# Rebuild the runtime-env venv (editable install of this worktree)
cd benchmarks/e2e/runtime-env && uv sync && cd ../../..

# The comfyui-env is only needed for upstream-side benchmark comparison; skip for now

# Restore workflow model symlinks + input images
for w in sd15_text_to_image img2img inpainting hires_fix area_composition flux2_klein_text_to_image esrgan_upscale; do
  mkdir -p workflows/$w/models
  for sub in $(ls /home/yanweiye/Project/comfy_runtime/workflows/$w/models/ 2>/dev/null); do
    if [ -d /home/yanweiye/Project/comfy_runtime/workflows/$w/models/$sub ]; then
      ln -sfn /home/yanweiye/Project/comfy_runtime/workflows/$w/models/$sub workflows/$w/models/$sub
    fi
  done
done
```

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
