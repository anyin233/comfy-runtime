# esrgan_upscale

[← Back to summary](../README.md)

## Stage breakdown (mean +/- stddev, ms)

| Stage | comfy_runtime min | mean | median | stddev | ComfyUI min | mean | median | stddev | Δmean |
|---|---|---|---|---|---|---|---|---|---|
| load_input | 10.7 | 16.2 | 10.8 | 7.7 | 10.0 | 10.1 | 10.0 | 0.1 | +60.8% |
| model_load | 262.1 | 264.3 | 262.6 | 2.8 | 306.3 | 311.5 | 313.5 | 3.7 | -15.1% |
| upscale | 613.5 | 615.0 | 613.6 | 2.0 | 610.4 | 613.7 | 613.2 | 2.9 | +0.2% |
| save | 490.0 | 491.9 | 491.3 | 1.9 | 513.5 | 518.9 | 520.1 | 4.0 | -5.2% |

| **total** | 1613.7 | 1623.9 | 1621.3 | 9.5 | 1449.0 | 1455.6 | 1455.4 | 5.5 | **+11.6%** |

![Stage breakdown](../figures/stage_breakdown_esrgan_upscale.png)

## Memory

| Metric | comfy_runtime (MB) | ComfyUI (MB) | Δ |
|---|---|---|---|
| GPU max allocated | 3139.1 | 3139.1 | +0.0% |
| GPU max reserved  | 5252.0 | 5252.0 | +0.0% |
| Host VmHWM        | 1159.3 | 1311.4 | -11.6% |

## Per-node breakdown (mean, ms)

| Node | Call index | comfy_runtime | ComfyUI | Δ |
|---|---|---|---|---|
| LoadImage | 0 | 16.2 | 10.1 | +60.8% |
| UpscaleModelLoader | 0 | 264.3 | 311.5 | -15.1% |
| ImageUpscaleWithModel | 0 | 615.0 | 613.7 | +0.2% |
| SaveImage | 0 | 491.9 | 518.9 | -5.2% |


## Raw data

- [esrgan_upscale_comfyui_0.json](../data/esrgan_upscale_comfyui_0.json)
- [esrgan_upscale_comfyui_1.json](../data/esrgan_upscale_comfyui_1.json)
- [esrgan_upscale_comfyui_2.json](../data/esrgan_upscale_comfyui_2.json)
- [esrgan_upscale_comfyui_3.json](../data/esrgan_upscale_comfyui_3.json)
- [esrgan_upscale_runtime_0.json](../data/esrgan_upscale_runtime_0.json)
- [esrgan_upscale_runtime_1.json](../data/esrgan_upscale_runtime_1.json)
- [esrgan_upscale_runtime_2.json](../data/esrgan_upscale_runtime_2.json)
- [esrgan_upscale_runtime_3.json](../data/esrgan_upscale_runtime_3.json)
