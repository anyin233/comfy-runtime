# hires_fix

[← Back to summary](../README.md)

## Stage breakdown (mean +/- stddev, ms)

| Stage | comfy_runtime min | mean | median | stddev | ComfyUI min | mean | median | stddev | Δmean |
|---|---|---|---|---|---|---|---|---|---|
| model_load | 528.8 | 531.3 | 532.2 | 1.8 | 660.6 | 666.2 | 669.0 | 3.9 | -20.3% |
| text_encode | 123.1 | 123.9 | 123.3 | 1.0 | 120.0 | 120.6 | 120.4 | 0.5 | +2.7% |
| latent_init | 0.1 | 0.1 | 0.1 | 0.0 | 0.2 | 0.2 | 0.2 | 0.0 | -70.7% |
| sample | 2590.4 | 2594.6 | 2596.4 | 2.9 | 2522.1 | 2536.5 | 2542.5 | 10.2 | +2.3% |
| upscale_latent | 0.2 | 0.2 | 0.2 | 0.0 | 0.2 | 0.2 | 0.2 | 0.0 | -14.7% |
| decode | 272.4 | 272.9 | 273.0 | 0.4 | 260.7 | 262.5 | 261.5 | 1.9 | +4.0% |
| save | 132.7 | 134.8 | 135.0 | 1.7 | 136.4 | 138.1 | 138.8 | 1.2 | -2.4% |

| **total** | 3656.3 | 3661.3 | 3660.8 | 4.4 | 3714.0 | 3726.0 | 3728.7 | 8.9 | **-1.7%** |

![Stage breakdown](../figures/stage_breakdown_hires_fix.png)

## Memory

| Metric | comfy_runtime (MB) | ComfyUI (MB) | Δ |
|---|---|---|---|
| GPU max allocated | 13759.8 | 4283.7 | +221.2% |
| GPU max reserved  | 14174.0 | 5278.0 | +168.5% |
| Host VmHWM        | 6959.6 | 7015.9 | -0.8% |

## Per-node breakdown (mean, ms)

| Node | Call index | comfy_runtime | ComfyUI | Δ |
|---|---|---|---|---|
| CheckpointLoaderSimple | 0 | 531.3 | 666.2 | -20.3% |
| CLIPTextEncode | 0 | 110.0 | 107.2 | +2.6% |
| CLIPTextEncode | 1 | 13.9 | 13.4 | +3.6% |
| EmptyLatentImage | 0 | 0.1 | 0.2 | -70.7% |
| KSampler | 0 | 1145.0 | 1089.7 | +5.1% |
| LatentUpscale | 0 | 0.2 | 0.2 | -14.7% |
| KSampler | 1 | 1449.5 | 1446.7 | +0.2% |
| VAEDecode | 0 | 272.9 | 262.5 | +4.0% |
| SaveImage | 0 | 134.8 | 138.1 | -2.4% |


## Raw data

- [hires_fix_comfyui_0.json](../data/hires_fix_comfyui_0.json)
- [hires_fix_comfyui_1.json](../data/hires_fix_comfyui_1.json)
- [hires_fix_comfyui_2.json](../data/hires_fix_comfyui_2.json)
- [hires_fix_comfyui_3.json](../data/hires_fix_comfyui_3.json)
- [hires_fix_runtime_0.json](../data/hires_fix_runtime_0.json)
- [hires_fix_runtime_1.json](../data/hires_fix_runtime_1.json)
- [hires_fix_runtime_2.json](../data/hires_fix_runtime_2.json)
- [hires_fix_runtime_3.json](../data/hires_fix_runtime_3.json)
