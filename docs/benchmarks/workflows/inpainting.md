# inpainting

[← Back to summary](../README.md)

## Stage breakdown (mean +/- stddev, ms)

| Stage | comfy_runtime min | mean | median | stddev | ComfyUI min | mean | median | stddev | Δmean |
|---|---|---|---|---|---|---|---|---|---|
| model_load | 536.1 | 541.3 | 541.9 | 4.0 | 675.9 | 702.6 | 685.8 | 31.1 | -23.0% |
| load_input | 7.5 | 7.7 | 7.7 | 0.2 | 8.7 | 8.8 | 8.9 | 0.1 | -12.9% |
| vae_encode | 248.7 | 254.1 | 254.9 | 4.1 | 250.4 | 252.1 | 252.8 | 1.2 | +0.8% |
| noise_mask | 0.0 | 0.1 | 0.1 | 0.0 | 0.1 | 0.1 | 0.1 | 0.0 | -46.8% |
| text_encode | 111.6 | 114.9 | 115.6 | 2.4 | 116.6 | 118.0 | 117.8 | 1.2 | -2.6% |
| sample | 1143.3 | 1187.7 | 1190.3 | 35.3 | 1154.1 | 1158.3 | 1154.9 | 5.4 | +2.5% |
| decode | 42.6 | 43.0 | 42.7 | 0.4 | 40.7 | 40.9 | 40.8 | 0.1 | +5.1% |
| save | 33.9 | 34.9 | 34.4 | 1.0 | 35.5 | 36.0 | 36.1 | 0.3 | -3.1% |

| **total** | 2127.5 | 2187.4 | 2191.3 | 47.3 | 2286.2 | 2318.8 | 2297.5 | 38.4 | **-5.7%** |

![Stage breakdown](../figures/stage_breakdown_inpainting.png)

## Memory

| Metric | comfy_runtime (MB) | ComfyUI (MB) | Δ |
|---|---|---|---|
| GPU max allocated | 6565.6 | 2645.5 | +148.2% |
| GPU max reserved  | 6760.0 | 2908.0 | +132.5% |
| Host VmHWM        | 6959.6 | 7016.5 | -0.8% |

## Per-node breakdown (mean, ms)

| Node | Call index | comfy_runtime | ComfyUI | Δ |
|---|---|---|---|---|
| CheckpointLoaderSimple | 0 | 541.3 | 702.6 | -23.0% |
| LoadImage | 0 | 7.7 | 8.8 | -12.9% |
| VAEEncode | 0 | 254.1 | 252.1 | +0.8% |
| SetLatentNoiseMask | 0 | 0.1 | 0.1 | -46.8% |
| CLIPTextEncode | 0 | 100.7 | 104.6 | -3.7% |
| CLIPTextEncode | 1 | 14.2 | 13.4 | +6.0% |
| KSampler | 0 | 1187.7 | 1158.3 | +2.5% |
| VAEDecode | 0 | 43.0 | 40.9 | +5.1% |
| SaveImage | 0 | 34.9 | 36.0 | -3.1% |


## Raw data

- [inpainting_comfyui_0.json](../data/inpainting_comfyui_0.json)
- [inpainting_comfyui_1.json](../data/inpainting_comfyui_1.json)
- [inpainting_comfyui_2.json](../data/inpainting_comfyui_2.json)
- [inpainting_comfyui_3.json](../data/inpainting_comfyui_3.json)
- [inpainting_runtime_0.json](../data/inpainting_runtime_0.json)
- [inpainting_runtime_1.json](../data/inpainting_runtime_1.json)
- [inpainting_runtime_2.json](../data/inpainting_runtime_2.json)
- [inpainting_runtime_3.json](../data/inpainting_runtime_3.json)
