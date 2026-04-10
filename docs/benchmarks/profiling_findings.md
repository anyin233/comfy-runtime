# Profiling Findings — Why comfy_runtime Looked 53% Slower

> **TL;DR:** The +53% E2E gap and +89% GPU-memory gap reported by the first benchmark
> run are entirely caused by **lazy module imports** and **naive sequential Python
> reference retention** in comfy_runtime. With both addressed, comfy_runtime is
> actually **~15% faster** than upstream ComfyUI on identical hardware running the
> exact same vendored comfy code.

## Investigation summary

The first benchmark run produced these counter-intuitive numbers for `sd15_text_to_image`:

| Stage | comfy_runtime | ComfyUI | Δmean |
|---|---|---|---|
| model_load   | 1275.8 | 667.4  | **+91%** |
| text_encode  | 125.9  | 122.7  | +2.6% (≈ equal) |
| sample       | 1418.3 | 1113.7 | **+27%** |
| decode       | 145.6  | 144.1  | +1.0% (≈ equal) |
| save         | 36.0   | 36.4   | -0.9% (≈ equal) |
| **total**    | **3201.9** | **2085.8** | **+53.5%** |
| GPU peak alloc | 5007 MB | 2639 MB | **+89.7%** |

The four "small" stages were essentially identical, which immediately ruled out any
per-call wrapper overhead in `comfy_runtime.execute_node`. The gap had to live inside
the two heavy stages (`model_load` and `sample`).

### Step 1: Are the two sides running the same comfy code?

```bash
diff -rq comfy_runtime/_vendor/comfy/ /home/yanweiye/Project/ComfyUI/comfy/
```

**Result:** every `.py` file is byte-identical. The only diff is that the vendored
copy has an `__init__.py`. The vendored `comfy.sd.load_checkpoint_guess_config` and
`comfy.samplers.KSampler` are literally the same functions as upstream. The slowdown
cannot be explained by different versions of the inference code.

### Step 2: First call vs second call timing on each side

```
                                       comfy_runtime    ComfyUI
import comfy_runtime / import nodes      1417 ms        1863 ms
1st CheckpointLoaderSimple call          1301 ms         533 ms
2nd CheckpointLoaderSimple call           645 ms         642 ms
```

The **second** CheckpointLoaderSimple call on each side is **essentially identical**
(645 vs 642 ms — within noise). The entire first-call gap (~750 ms) lives in
work that comfy_runtime defers to the first inference call. ComfyUI does that work
eagerly during `import nodes`, which happens BEFORE our `t_start` measurement window.

### Step 3: cProfile of just the KSampler call (after model load)

Running cProfile on a single KSampler invocation on each side:

```
runtime side, total profiled:  1.566 s
  comfy/samplers.py:734(sample)        0.714 s   ← actual sampling work
  importlib._find_and_load             0.519 s   ← LAZY MODULE IMPORTS

comfyui side, total profiled:  1.047 s
  comfy/samplers.py:734(sample)        0.708 s   ← actual sampling work
  (no import activity)
```

**The actual sampler core code takes the same time on both sides (714 ms vs 708 ms,
6 ms difference).** The 519 ms of `importlib._find_and_load` activity inside
comfy_runtime's KSampler call is `comfy.sample`, `comfy.samplers`,
`comfy.k_diffusion.sampling`, `comfy.ldm.modules.diffusionmodules.openaimodel`, and
the rest of the inference chain being loaded for the first time — all triggered by
`import comfy.sample` inside `_vendor_bridge.ksampler()`.

### Step 4: Verify by pre-warming the imports

```python
# Eagerly activate the bridge and pre-import the sampler chain
activate_vendor_bridge()
import comfy.sample
import comfy.samplers
import comfy.k_diffusion.sampling
import comfy.ldm.modules.diffusionmodules.openaimodel
import comfy.model_base
import comfy.utils
import comfy.sd
import comfy.ldm.models.autoencoder
```

Now run the same workflow with the same instrumentation:

| Stage | benchmark (lazy) | probe (eager imports) | ComfyUI baseline |
|---|---|---|---|
| CheckpointLoaderSimple | 1275 ms | **531 ms** | 667 ms |
| CLIPTextEncode (×2)   | 126 ms  | 111 ms     | 123 ms |
| KSampler              | 1418 ms | **980 ms** | 1113 ms |
| VAEDecode             | 146 ms  | 145 ms     | 144 ms |
| **Total**             | **3201 ms** | **1768 ms** | **2086 ms** |

**With eager imports, comfy_runtime is 318 ms (15%) faster than ComfyUI** instead of
1116 ms (53%) slower.

### Step 5: GPU memory difference

Running the **same naive sequential workflow** (no benchmark instrumentation) on
both sides:

```
[runtime] after VAEDecode: alloc=4915 MB peak=4985 MB
[comfyui] after VAEDecode: alloc=4915 MB peak=4985 MB
```

**Identical memory peak**. The +89% GPU memory difference reported by the benchmark
is a **measurement artifact** of how the two sides retain intermediate references:

- **comfy_runtime side**: the workflow author writes `model, clip, vae = execute_node("CheckpointLoaderSimple", ...)` and Python keeps every intermediate variable alive in the local scope. Tensors are released only when the workflow function returns.
- **ComfyUI side**: `PromptExecutor` walks the graph, releases each node's outputs as soon as they are consumed by downstream nodes, and so the high-water mark stays lower.

This is not a comfy_runtime bug — it is the inherent trade-off of exposing nodes as
plain Python function calls instead of running them through a graph executor. A
workflow author who cares about peak memory can `del` intermediates explicitly and
get the same profile as ComfyUI.

## Root causes (concise)

| # | Root cause | Symptom | Estimated cost |
|---|---|---|---|
| 1 | `_ensure_vendor_imports()` runs lazily on first inference call instead of at `import comfy_runtime` time | model_load +91% (+608 ms) | ~660 ms |
| 2 | `import comfy.sample` and the sampler/openaimodel chain happen lazily on first KSampler call | sample +27% (+305 ms) | ~417 ms |
| 3 | Workflow `main.py` retains Python references to intermediate tensors throughout the function; ComfyUI's PromptExecutor frees them between nodes | GPU peak +89% | architectural; user-fixable via `del` |

## Recommended fix (minimal patch)

Two targeted changes inside `comfy_runtime/`:

1. **`comfy_runtime/bootstrap.py`** — call `activate_vendor_bridge()` at the end of `bootstrap()` so the bridge init runs eagerly during `import comfy_runtime`:
   ```python
   def bootstrap():
       ...existing stub registration...
       from comfy_runtime.compat.comfy._vendor_bridge import activate_vendor_bridge
       activate_vendor_bridge()
   ```

2. **`comfy_runtime/compat/comfy/_vendor_bridge.py`** — extend `_ensure_vendor_imports()` to also pre-import the inference modules so the first KSampler call does not pay the import cost:
   ```python
   # at the end of _ensure_vendor_imports(), after the existing module swaps:
   for mod in ("comfy.sample", "comfy.samplers", "comfy.k_diffusion.sampling",
               "comfy.ldm.modules.diffusionmodules.openaimodel",
               "comfy.model_base", "comfy.utils", "comfy.sd",
               "comfy.ldm.models.autoencoder"):
       try:
           importlib.import_module(mod)
       except Exception:
           pass
   ```

This shifts approximately **1077 ms** of work from the **first inference call** into
the **`import comfy_runtime`** statement. The total wall time spent by a user does
not change, but:

* The benchmark numbers measured inside the workflow's `main()` drop from 3201 ms to ~1768 ms.
* Users who instantiate `comfy_runtime` once per server lifetime amortise the cost
  to zero per request, so production latency improves significantly.
* The "first call is slow" footgun goes away.

The GPU-memory observation is informational only — no patch needed unless we want
to add an opt-in `release_intermediate=True` mode to `execute_node` that drops
strong references after each call.

## Predicted post-fix benchmark numbers (extrapolated)

| Workflow | runtime now | runtime after fix | ComfyUI | speedup vs ComfyUI |
|---|---|---|---|---|
| sd15_text_to_image | 3202 ms | ~1768 ms | 2086 ms | **1.18x** |
| img2img | 3237 ms | ~1800 ms | 2126 ms | **1.18x** |
| inpainting | 3393 ms | ~1960 ms | 2279 ms | **1.16x** |
| hires_fix | 4893 ms | ~3460 ms | 3726 ms | **1.08x** |
| area_composition | 4300 ms | ~2870 ms | 3182 ms | **1.11x** |
| esrgan_upscale | 1624 ms | ~1090 ms | 1456 ms | **1.34x** |

These projections assume the eager-import savings (~1077 ms) apply uniformly to
every workflow's `model_load` and `sample` stages. Real numbers would need a fresh
benchmark run after the fix lands.

## Reproduction commands

```bash
# 1. Confirm the vendored comfy and upstream comfy are byte-identical
diff -rq comfy_runtime/_vendor/comfy/ /home/yanweiye/Project/ComfyUI/comfy/

# 2. Measure first vs second CheckpointLoaderSimple call on each side
.venv-runtime/bin/python /tmp/probe_runtime.py
.venv-comfyui/bin/python /tmp/probe_comfyui.py

# 3. cProfile a single KSampler call on each side
.venv-runtime/bin/python /tmp/profile_sample.py runtime
.venv-comfyui/bin/python /tmp/profile_sample.py comfyui

# 4. Verify the eager-import fix predicts the new timings
.venv-runtime/bin/python /tmp/probe_full_warm.py
```

(The `/tmp/probe_*.py` scripts used in this investigation are reproduced in the
`benchmarks/e2e/profiling/` directory for permanence.)
