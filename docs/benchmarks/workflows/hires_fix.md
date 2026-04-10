# hires_fix

[← Back to summary](../README.md)

## Stage breakdown (mean +/- stddev, ms)

| Stage | comfy_runtime min | mean | median | stddev | ComfyUI min | mean | median | stddev | Δmean |
|---|---|---|---|---|---|---|---|---|---|
| model_load | 533.6 | 540.9 | 543.4 | 5.2 | 684.8 | 688.6 | 688.6 | 3.1 | -21.5% |
| text_encode | 122.4 | 123.3 | 123.4 | 0.7 | 119.0 | 124.0 | 125.3 | 3.7 | -0.5% |
| latent_init | 0.1 | 0.1 | 0.1 | 0.0 | 0.2 | 0.2 | 0.2 | 0.0 | -70.5% |
| sample | 2570.0 | 2587.4 | 2572.6 | 22.7 | 2539.0 | 2555.0 | 2558.2 | 12.0 | +1.3% |
| upscale_latent | 0.2 | 0.2 | 0.2 | 0.0 | 0.2 | 0.3 | 0.3 | 0.0 | -26.2% |
| decode | 271.4 | 272.1 | 272.1 | 0.6 | 260.7 | 262.7 | 261.8 | 2.1 | +3.6% |
| save | 132.8 | 135.0 | 135.7 | 1.6 | 135.3 | 139.3 | 138.6 | 3.6 | -3.1% |

| **total** | 3638.3 | 3662.5 | 3648.3 | 27.5 | 3757.3 | 3771.8 | 3773.8 | 11.1 | **-2.9%** |

![Stage breakdown](../figures/stage_breakdown_hires_fix.png)

## Memory

| Metric | comfy_runtime (MB) | ComfyUI (MB) | Δ |
|---|---|---|---|
| GPU max allocated | 13759.8 | 4283.7 | +221.2% |
| GPU max reserved  | 14174.0 | 5278.0 | +168.5% |
| Host VmHWM        | 6959.4 | 7015.4 | -0.8% |

## Per-node breakdown (mean, ms)

| Node | Call index | comfy_runtime | ComfyUI | Δ |
|---|---|---|---|---|
| CheckpointLoaderSimple | 0 | 540.9 | 688.6 | -21.5% |
| CLIPTextEncode | 0 | 109.5 | 110.7 | -1.1% |
| CLIPTextEncode | 1 | 13.8 | 13.3 | +4.0% |
| EmptyLatentImage | 0 | 0.1 | 0.2 | -70.5% |
| KSampler | 0 | 1138.3 | 1108.9 | +2.7% |
| LatentUpscale | 0 | 0.2 | 0.3 | -26.2% |
| KSampler | 1 | 1449.0 | 1446.1 | +0.2% |
| VAEDecode | 0 | 272.1 | 262.7 | +3.6% |
| SaveImage | 0 | 135.0 | 139.3 | -3.1% |


## Raw data

- [hires_fix_comfyui_0.json](../data/hires_fix_comfyui_0.json)
- [hires_fix_comfyui_1.json](../data/hires_fix_comfyui_1.json)
- [hires_fix_comfyui_2.json](../data/hires_fix_comfyui_2.json)
- [hires_fix_comfyui_3.json](../data/hires_fix_comfyui_3.json)
- [hires_fix_runtime_0.json](../data/hires_fix_runtime_0.json)
- [hires_fix_runtime_1.json](../data/hires_fix_runtime_1.json)
- [hires_fix_runtime_2.json](../data/hires_fix_runtime_2.json)
- [hires_fix_runtime_3.json](../data/hires_fix_runtime_3.json)
