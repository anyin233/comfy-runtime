# esrgan_upscale

[← Back to summary](../README.md)

## Stage breakdown (mean +/- stddev, ms)

| Stage | comfy_runtime min | mean | median | stddev | ComfyUI min | mean | median | stddev | Δmean |
|---|---|---|---|---|---|---|---|---|---|
| load_input | 9.7 | 9.7 | 9.7 | 0.1 | 10.0 | 10.2 | 10.1 | 0.2 | -4.8% |
| model_load | 168.7 | 170.4 | 171.2 | 1.2 | 299.4 | 310.3 | 315.4 | 7.7 | -45.1% |
| upscale | 615.5 | 626.6 | 631.2 | 7.9 | 611.0 | 613.9 | 612.1 | 3.3 | +2.1% |
| save | 487.8 | 488.2 | 488.3 | 0.3 | 515.6 | 520.1 | 520.7 | 3.4 | -6.1% |

| **total** | 1320.9 | 1330.8 | 1333.3 | 7.3 | 1438.5 | 1455.9 | 1461.9 | 12.5 | **-8.6%** |

![Stage breakdown](../figures/stage_breakdown_esrgan_upscale.png)

## Memory

| Metric | comfy_runtime (MB) | ComfyUI (MB) | Δ |
|---|---|---|---|
| GPU max allocated | 3139.1 | 3139.1 | +0.0% |
| GPU max reserved  | 5252.0 | 5252.0 | +0.0% |
| Host VmHWM        | 1230.6 | 1311.1 | -6.1% |

## Per-node breakdown (mean, ms)

| Node | Call index | comfy_runtime | ComfyUI | Δ |
|---|---|---|---|---|
| LoadImage | 0 | 9.7 | 10.2 | -4.8% |
| UpscaleModelLoader | 0 | 170.4 | 310.3 | -45.1% |
| ImageUpscaleWithModel | 0 | 626.6 | 613.9 | +2.1% |
| SaveImage | 0 | 488.2 | 520.1 | -6.1% |


## Raw data

- [esrgan_upscale_comfyui_0.json](../data/esrgan_upscale_comfyui_0.json)
- [esrgan_upscale_comfyui_1.json](../data/esrgan_upscale_comfyui_1.json)
- [esrgan_upscale_comfyui_2.json](../data/esrgan_upscale_comfyui_2.json)
- [esrgan_upscale_comfyui_3.json](../data/esrgan_upscale_comfyui_3.json)
- [esrgan_upscale_runtime_0.json](../data/esrgan_upscale_runtime_0.json)
- [esrgan_upscale_runtime_1.json](../data/esrgan_upscale_runtime_1.json)
- [esrgan_upscale_runtime_2.json](../data/esrgan_upscale_runtime_2.json)
- [esrgan_upscale_runtime_3.json](../data/esrgan_upscale_runtime_3.json)
