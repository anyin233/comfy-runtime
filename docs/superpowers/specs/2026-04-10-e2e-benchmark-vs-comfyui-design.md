# End-to-End Benchmark: comfy_runtime vs ComfyUI — Design

**Status:** Draft
**Date:** 2026-04-10
**Author:** cydia2001

## Goal

Produce a reproducible, scientifically defensible benchmark that quantifies
`comfy_runtime`'s execution-efficiency advantage over upstream ComfyUI for the
seven reference workflows shipped in `workflows/`. The output is a full report
under `docs/benchmarks/` intended for external readers (blog/paper/README link),
covering end-to-end wall time, per-stage and per-node breakdowns, and memory
footprint (GPU + host).

## Non-goals

- Benchmarking ComfyUI's HTTP server / queue / websocket layers (we measure the
  engine, not the service).
- Benchmarking custom-node packs other than the ones each workflow already uses.
- Measuring image quality or numerical equivalence beyond a correctness gate.
- Micro-benchmarks of individual helpers (those already exist under
  `benchmarks/bench_*.py`; this work is orthogonal and lives under
  `benchmarks/e2e/`).

## Key design decisions (confirmed with user)

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | Run ComfyUI **in-process via `execution.PromptExecutor`** with a JSON prompt graph, not via HTTP. | Measures the full engine overhead (graph validation, hierarchical cache, input signatures) that `comfy_runtime` is designed to skip, without mixing in HTTP/serialization noise. |
| 2 | **Two independent uv-managed venvs** (`.venv-runtime`, `.venv-comfyui`). Benchmark driver dispatches via `subprocess`; each side writes a JSON and the parent aggregates. | ComfyUI 0.18.1 pins its own torch; isolation removes dependency conflicts. Startup/import cost is part of what `comfy_runtime` optimizes, so two fresh processes are fair. |
| 3 | **Dual-granularity instrumentation**: stage view (for humans) + per-node view (for validation and appendix). | Stage view is the figure in the report; per-node view supports "model inference time is the same, framework overhead is where we win". |
| 4 | Memory: **GPU peak allocated + reserved** (via `torch.cuda.max_memory_*`) **+ host VmHWM** (via `/proc/self/status`). | Peak-only is sufficient for "which framework is leaner". VmHWM is kernel-maintained, zero-overhead, no sampling thread to perturb timings. |
| 5 | **Run protocol:** 1 warmup + 3 timed runs per `(workflow, side)`, each run in a fresh subprocess, `torch.cuda.empty_cache()` + `synchronize()` at run boundaries, fixed seed (42). Report min / mean / median / stddev. | Cold-start subprocesses include the import/configure cost that `comfy_runtime` optimizes. Warmup absorbs first-time OS page cache pain. Stddev proves the gap is not noise. |
| 6 | **Output**: full report (Markdown + JSON raw + matplotlib figures + per-node appendix + environment metadata). Top-level `docs/benchmarks/README.md` + one sub-page per workflow under `docs/benchmarks/workflows/`. | User will publish this externally; per-workflow sub-pages allow deep dives without cluttering the summary. |

## Architecture

### Directory layout

```
benchmarks/e2e/
├── pyproject.runtime.toml          # uv project for comfy_runtime side
├── pyproject.comfyui.toml          # uv project for ComfyUI side
├── .venv-runtime/                  # created via `uv sync --project pyproject.runtime.toml`
├── .venv-comfyui/                  # created via `uv sync --project pyproject.comfyui.toml`
├── _harness/
│   ├── __init__.py
│   ├── timing.py                   # StageRecorder, NodeRecorder, perf_counter + cuda.synchronize wrappers
│   ├── memory.py                   # GPU peak reader, VmHWM reader
│   ├── env.py                      # GPU/driver/CUDA/torch/uv.lock metadata collector
│   └── result_schema.py            # shared dataclass / TypedDict for one run's JSON
├── runners/
│   ├── runtime_runner.py           # subprocess entrypoint for comfy_runtime side
│   └── comfyui_runner.py           # subprocess entrypoint for ComfyUI side
├── workflows/                      # benchmark metadata per workflow (no per-workflow code)
│   ├── sd15_text_to_image/
│   │   ├── comfyui_prompt.json     # equivalent ComfyUI API-format prompt graph
│   │   └── stages.yaml             # class_type → stage mapping (shared by both sides)
│   ├── flux2_klein_text_to_image/
│   ├── img2img/
│   ├── inpainting/
│   ├── hires_fix/
│   ├── area_composition/
│   └── esrgan_upscale/
├── run_all.py                      # top-level driver: workflows × sides × (1 warmup + 3 timed) subprocesses
├── aggregate.py                    # results/ → docs/benchmarks/ + figures/
├── verify.py                       # correctness gate: each workflow runs once per side, compares output image stats
└── results/                        # raw JSON, gitignored except docs/benchmarks/data copy
    └── {timestamp}/
        └── {workflow}_{side}_{run_idx}.json

docs/benchmarks/
├── README.md                       # overview, environment, summary table, e2e + memory figures, links to per-workflow pages
├── workflows/
│   ├── sd15_text_to_image.md
│   ├── flux2_klein_text_to_image.md
│   ├── img2img.md
│   ├── inpainting.md
│   ├── hires_fix.md
│   ├── area_composition.md
│   └── esrgan_upscale.md
├── data/                           # versioned copy of the "official" run's JSON
└── figures/
    ├── e2e_comparison.png          # 7 workflows × 2 sides, bars with error bars
    ├── stage_breakdown_{workflow}.png
    └── memory_comparison.png
```

### Run orchestration (`run_all.py`)

```python
RUNS_PER_SIDE = 4   # 1 warmup (run_idx=0, discarded at aggregation) + 3 timed
WORKFLOWS = ["sd15_text_to_image", "flux2_klein_text_to_image",
             "img2img", "inpainting", "hires_fix",
             "area_composition", "esrgan_upscale"]
SIDES = [
    ("runtime", ".venv-runtime/bin/python", "runners/runtime_runner.py"),
    ("comfyui", ".venv-comfyui/bin/python", "runners/comfyui_runner.py"),
]

for workflow in WORKFLOWS:
    for side_name, python_bin, runner_py in SIDES:
        for run_idx in range(RUNS_PER_SIDE):
            out = f"results/{timestamp}/{workflow}_{side_name}_{run_idx}.json"
            subprocess.run(
                [python_bin, runner_py,
                 "--workflow", workflow,
                 "--run-idx", str(run_idx),
                 "--out", out],
                check=False,            # failures are captured in the JSON
                env=clean_env(),        # CUDA_VISIBLE_DEVICES=0, PYTHONHASHSEED=0, ...
            )
```

- **Execution is strictly serial** (no parallelism across workflows or sides).
  Two workflows contending for the GPU would destroy measurement validity.
- **Failures do not abort the batch**; the runner catches exceptions and writes
  a JSON with `{"status": "failed", "error": "..."}`. Aggregation surfaces
  failures as explicit cells in the report.
- **`clean_env()`** whitelists environment variables to reduce drift across
  runs: fixed `CUDA_VISIBLE_DEVICES=0`, `PYTHONHASHSEED=0`,
  `CUBLAS_WORKSPACE_CONFIG=:4096:8` (deterministic), and strips
  `PYTHONPATH` so each venv resolves cleanly.

### Instrumentation: comfy_runtime side (`runners/runtime_runner.py`)

1. Parse CLI args (`--workflow`, `--run-idx`, `--out`).
2. **Seed determinism**: call `random.seed(42)` and `torch.manual_seed(42)`
   **before** loading the workflow module. Six of the seven workflows compute
   `SEED = random.randint(0, 2**63)` at module import time; seeding Python's
   `random` first locks that to a deterministic value.
3. `import comfy_runtime` and **monkey-patch `comfy_runtime.execute_node`** with
   a wrapper that:
   - records `class_type` and elapsed ns (after `torch.cuda.synchronize()`)
   - attributes the elapsed time to the current stage via the workflow's
     `stages.yaml` mapping
4. `torch.cuda.reset_peak_memory_stats()`.
5. `importlib.util`-load the workflow's existing `workflows/{name}/main.py`.
   **Before** calling `module.main()`, overwrite `module.SEED = 42` so the
   sampler call passes seed=42, matching the `comfyui_prompt.json` on the other
   side. This keeps the workflow source files untouched and gives both sides
   the exact same seed for comparable sampler output.
6. Call `module.main()`. No other changes to the workflow source —
   monkey-patching alone produces the timing data.
7. `torch.cuda.synchronize()`, capture total wall time, peak GPU, VmHWM.
8. Serialize to JSON via the shared schema and write to `--out`.

### Instrumentation: ComfyUI side (`runners/comfyui_runner.py`)

1. Parse CLI args.
2. `sys.path.insert(0, "/home/yanweiye/Project/ComfyUI")`, then `import nodes,
   execution, folder_paths`.
3. Point `folder_paths` at the workflow's `models/` tree for each category
   (`checkpoints`, `diffusion_models`, `text_encoders`, `vae`, `upscale_models`,
   `controlnet`, `clip_vision`, `loras`). We add paths via
   `folder_paths.add_model_folder_path(category, path)` — same mechanism
   ComfyUI uses for extra_model_paths.yaml.
4. Load the workflow's custom nodes directory (`workflows/{name}/nodes/`) via
   `nodes.load_custom_node(path)` if it exists.
5. **Monkey-patch `execution.execute`** (the node-execution callsite used by
   `PromptExecutor`) with a wrapper using `*args, **kwargs` passthrough.
   Detect the `class_type` from the `dynprompt` argument and record
   `(class_type, elapsed_ns)` after `torch.cuda.synchronize()`.
6. Load `workflows/{name}/comfyui_prompt.json`, instantiate
   `execution.PromptExecutor(server=MockServer(), lru_size=0)` (LRU disabled to
   avoid any cross-node caching that `comfy_runtime` does not have).
7. `torch.cuda.reset_peak_memory_stats()`, call `executor.execute(prompt,
   prompt_id="bench", extra_data={}, execute_outputs=[...])`.
8. `torch.cuda.synchronize()`, capture total wall time, peak GPU, VmHWM.
9. Aggregate per-node records into stages via `stages.yaml`.
10. Serialize to JSON, write to `--out`.

**`MockServer`** is a minimal stub exposing the methods `PromptExecutor`
touches (`send_sync`, `last_node_id`, `last_prompt_id`, `client_id`,
`prompt_queue`) — all no-ops, so server-side telemetry/events are dropped.

### Per-node-sync trade-off

Calling `torch.cuda.synchronize()` after every node forces serialization,
making both sides pay the sync cost. The E2E number therefore includes this
overhead on both sides equally — the comparison is fair, though the absolute
numbers will be slightly higher than production throughput. This is documented
in the methodology section of the final report.

### Stage definition

Each workflow has a `stages.yaml`:

```yaml
# sd15_text_to_image/stages.yaml
stages:
  model_load:  [CheckpointLoaderSimple]
  text_encode: [CLIPTextEncode]
  latent_init: [EmptyLatentImage]
  sample:      [KSampler]
  decode:      [VAEDecode]
  save:        [SaveImage]
```

Stages are workflow-specific (Flux2 has different node types), but the
harness groups by `class_type` identically on both sides, guaranteeing aligned
labels in the output table.

If the same `class_type` is called more than once in a workflow (e.g.,
`CLIPTextEncode` for positive + negative prompts), all calls land in the same
stage bucket — this matches the semantic intent and keeps both sides aligned.

### Memory capture

```python
# both sides, same helper
torch.cuda.reset_peak_memory_stats()   # before workflow
# ... workflow runs ...
gpu_allocated = torch.cuda.max_memory_allocated()
gpu_reserved = torch.cuda.max_memory_reserved()

def read_vmhwm_peak() -> int:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024  # kB → bytes
    raise RuntimeError("VmHWM not found in /proc/self/status")
```

No background sampling thread. Peak only. Zero perturbation of timing.

### Environment capture (`_harness/env.py`)

Each result JSON embeds:

```json
{
  "env": {
    "hostname": "...",
    "timestamp_utc": "2026-04-10T...",
    "python_version": "3.13.7",
    "torch_version": "2.11.0+cu130",
    "cuda_version": "13.0",
    "gpu_name": "NVIDIA GeForce RTX 4090",
    "gpu_memory_total_mb": 24564,
    "driver_version": "...",
    "comfy_runtime_version": "0.3.1",
    "comfyui_commit": "b615af1c",
    "runtime_uv_lock_sha256": "...",
    "comfyui_uv_lock_sha256": "..."
  }
}
```

All values read once in the runner startup, before instrumentation begins,
so they don't affect timing. The summary `docs/benchmarks/README.md`
collapses this into a single "Environment" section at the top.

### Result JSON schema

```python
# _harness/result_schema.py
@dataclass
class NodeRecord:
    class_type: str
    call_index: int                 # 0 for first call, 1 for second call of same class_type
    elapsed_ns: int

@dataclass
class StageRecord:
    name: str                       # "model_load", "text_encode", ...
    elapsed_ns: int
    node_count: int

@dataclass
class RunResult:
    workflow: str
    side: str                       # "runtime" | "comfyui"
    run_idx: int                    # 0 = warmup, 1..3 = timed
    status: str                     # "ok" | "failed"
    error: str | None
    total_ns: int
    stages: list[StageRecord]
    nodes: list[NodeRecord]
    gpu_peak_allocated_bytes: int
    gpu_peak_reserved_bytes: int
    host_vmhwm_bytes: int
    env: dict
```

### Correctness verification (`verify.py`)

Runs once before the first benchmark batch (and can be re-run on demand).

For each workflow:
1. Run comfy_runtime side once, save output image.
2. Run ComfyUI side once, save output image.
3. Compare: shape, dtype, `abs(mean_a - mean_b) / mean_a < 0.01`, same for stddev.
4. If mismatch, print which stage/node is suspicious (by comparing per-node
   times — if KSampler time differs by 50%, likely a parameter mismatch).

**Verification failing aborts the benchmark**. The intent is to catch
JSON-prompt authoring errors (wrong sampler name, off-by-one width) before
wasting 30+ minutes on a full run.

### Aggregation (`aggregate.py`)

Inputs: `results/{timestamp}/` (or `results/latest` symlink).
Outputs: `docs/benchmarks/README.md`, `docs/benchmarks/workflows/*.md`,
`docs/benchmarks/data/*.json`, `docs/benchmarks/figures/*.png`.

Steps:
1. Glob all JSONs under `results/{timestamp}/`.
2. Group by `(workflow, side)`, discard `run_idx == 0`.
3. Per group, compute {min, mean, median, stddev, p95} over {total, each
   stage, each node, gpu_peak_allocated, gpu_peak_reserved, host_vmhwm}.
4. Load Jinja2 templates (`_harness/templates/*.md.j2`) and render.
5. Emit matplotlib figures (see §Figures).
6. Copy the raw JSONs into `docs/benchmarks/data/` for versioned archiving.

### Figures

All generated via matplotlib, saved as PNG at 150 DPI.

1. **`e2e_comparison.png`** — grouped bar chart: x = 7 workflows,
   y = mean total time (ms), two bars per workflow (runtime, comfyui), error
   bars = stddev. Title includes the speedup range as a subtitle.
2. **`stage_breakdown_{workflow}.png`** — stacked bar: two bars (runtime vs
   comfyui), each stacked by stage. Inline labels show each stage's ms and
   percentage. Makes the "ComfyUI overhead lives in these stages" story
   visually obvious.
3. **`memory_comparison.png`** — grouped bar: x = 7 workflows, three metric
   groups (gpu_allocated, gpu_reserved, host_vmhwm), runtime vs comfyui side
   by side. Log y-axis because flux2 dominates.

### Report templates (sketch)

**`docs/benchmarks/README.md`**:

- Title, TL;DR (one sentence + headline figure)
- Methodology (protocol, hardware, software versions, limitations — including
  per-node sync trade-off)
- Environment table (auto-generated from `env` field)
- Summary table: 7 rows × {E2E runtime mean, E2E comfyui mean, speedup,
  GPU peak Δ, host VmHWM Δ}
- `e2e_comparison.png`
- `memory_comparison.png`
- Links to per-workflow pages
- "Reproducing" section with 3-line bootstrap + run command

**`docs/benchmarks/workflows/{name}.md`** (one per workflow):

- Pipeline description (one line)
- Stage breakdown table (runtime min/mean/median/stddev vs comfyui
  min/mean/median/stddev vs Δmean%)
- `stage_breakdown_{name}.png`
- Memory table (GPU allocated, GPU reserved, host VmHWM, Δ)
- Per-node appendix table
- Links to raw data JSONs

## One-time bootstrap work

1. `uv sync --project benchmarks/e2e/pyproject.runtime.toml` — creates
   `.venv-runtime` with the local editable `comfy_runtime` + deps.
2. `uv sync --project benchmarks/e2e/pyproject.comfyui.toml` — creates
   `.venv-comfyui` with ComfyUI 0.18.1 (editable install from
   `/home/yanweiye/Project/ComfyUI`) + its deps. Pin torch to match
   `comfy_runtime`'s torch 2.11.0+cu130 if possible, else let ComfyUI pick.
3. **Hand-author 7 `comfyui_prompt.json` files** — one per workflow, each
   mirroring the corresponding `workflows/{name}/main.py` exactly (same node
   order, parameters, seed=42, prompt text).
4. **Hand-author 7 `stages.yaml` files** — one per workflow, mapping node
   `class_type` to stage name.
5. Run `verify.py` → must pass before benchmark runs.
6. Run `run_all.py` → produces `results/{timestamp}/`.
7. Run `aggregate.py results/{timestamp}` → produces `docs/benchmarks/`.

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| ComfyUI's `execution.execute` signature differs from expected at patch time | Use `inspect.signature` to introspect and patch with `*args, **kwargs` passthrough; fallback to patching at the `PromptExecutor.execute` level if needed. |
| ComfyUI 0.18.1 requires a torch version incompatible with `comfy_runtime`'s | Two separate venvs already isolate this. Worst case: ComfyUI venv uses a different torch minor, disclosed in the report's environment section. |
| A workflow's `comfyui_prompt.json` is not truly equivalent to its `main.py` (wrong sampler, missing node) | `verify.py` catches this before we spend time on a full benchmark run. |
| Output image is nondeterministic (cudnn nondeterministic kernels) | `verify.py` uses statistical comparison (mean/stddev within 1%), not pixel-exact. Seed fixed at 42. `CUBLAS_WORKSPACE_CONFIG=:4096:8` for deterministic workspace. |
| Six workflows use `SEED = random.randint(0, 2**63)` at module import, so naive re-execution would pick a different seed every run | `runtime_runner.py` seeds `random`/`torch` before importing the workflow module, then overwrites `module.SEED = 42` before calling `main()`. `comfyui_prompt.json` hard-codes `seed: 42` on the matching sampler node. Both sides provably pass the same seed. |
| ComfyUI's `PromptExecutor` LRU cache hits across re-runs, skewing measurements | Each run is a fresh subprocess, so cache is empty. Additionally `lru_size=0` passed explicitly. |
| `torch.cuda.synchronize()` per node adds overhead | Both sides pay it equally, documented in methodology section. Absolute numbers will be slightly higher than production throughput, but the *comparison* is valid. |
| `VmHWM` includes memory held by libraries loaded during Python startup — could dominate for small workflows | That's actually the point: comfy_runtime's lean import is a measured advantage. Reported as-is. |
| One workflow failing takes down the whole batch | Runner catches exceptions, writes `{"status": "failed", ...}` JSON; aggregation surfaces failures as explicit cells. |

## Out of scope for this spec

The following are deliberately excluded:

- Running the same benchmark on a second hardware target (CPU, different GPU).
- Benchmarking `comfy_runtime` against other runtimes (only vs upstream ComfyUI).
- Profiling at the CUDA kernel level (Nsight, `torch.profiler`).
- Tracking benchmark regressions over time (no CI integration).

These are natural follow-ups but each doubles the scope.

## Success criteria

- All 7 workflows produce valid JSONs on both sides (no failures).
- `verify.py` passes for all 7 workflows.
- `docs/benchmarks/README.md` renders with populated tables and figures.
- Each per-workflow page links to its raw data and figures.
- A reader new to the project can run `uv sync` + `python run_all.py` +
  `python aggregate.py` and reproduce similar numbers on comparable hardware
  (within stddev).
