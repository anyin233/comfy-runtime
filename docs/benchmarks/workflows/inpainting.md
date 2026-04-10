# inpainting

[← Back to summary](../README.md)

## Stage breakdown (mean +/- stddev, ms)

| Stage | comfy_runtime min | mean | median | stddev | ComfyUI min | mean | median | stddev | Δmean |
|---|---|---|---|---|---|---|---|---|---|
| model_load | 540.8 | 551.0 | 545.1 | 11.6 | 647.2 | 655.2 | 658.7 | 5.7 | -15.9% |
| load_input | 7.9 | 7.9 | 7.9 | 0.0 | 8.8 | 8.9 | 8.9 | 0.1 | -11.5% |
| vae_encode | 256.4 | 259.3 | 259.4 | 2.3 | 252.1 | 252.9 | 252.4 | 1.0 | +2.5% |
| noise_mask | 0.0 | 0.1 | 0.1 | 0.0 | 0.1 | 0.1 | 0.1 | 0.0 | -41.9% |
| text_encode | 115.5 | 116.3 | 116.1 | 0.7 | 116.1 | 117.6 | 117.8 | 1.2 | -1.2% |
| sample | 1198.3 | 1201.3 | 1200.2 | 3.0 | 1110.3 | 1127.9 | 1134.7 | 12.5 | +6.5% |
| decode | 42.8 | 43.0 | 43.0 | 0.2 | 40.8 | 41.2 | 40.8 | 0.6 | +4.3% |
| save | 35.6 | 35.9 | 35.7 | 0.4 | 34.8 | 35.3 | 34.9 | 0.5 | +1.9% |

| **total** | 2207.0 | 2218.5 | 2212.6 | 12.5 | 2212.1 | 2241.1 | 2253.7 | 20.6 | **-1.0%** |

![Stage breakdown](../figures/stage_breakdown_inpainting.png)

## Memory

| Metric | comfy_runtime (MB) | ComfyUI (MB) | Δ |
|---|---|---|---|
| GPU max allocated | 6565.6 | 2645.5 | +148.2% |
| GPU max reserved  | 6760.0 | 2908.0 | +132.5% |
| Host VmHWM        | 6957.8 | 7016.6 | -0.8% |

## Per-node breakdown (mean, ms)

| Node | Call index | comfy_runtime | ComfyUI | Δ |
|---|---|---|---|---|
| CheckpointLoaderSimple | 0 | 551.0 | 655.2 | -15.9% |
| LoadImage | 0 | 7.9 | 8.9 | -11.5% |
| VAEEncode | 0 | 259.3 | 252.9 | +2.5% |
| SetLatentNoiseMask | 0 | 0.1 | 0.1 | -41.9% |
| CLIPTextEncode | 0 | 101.9 | 104.2 | -2.2% |
| CLIPTextEncode | 1 | 14.3 | 13.5 | +6.5% |
| KSampler | 0 | 1201.3 | 1127.9 | +6.5% |
| VAEDecode | 0 | 43.0 | 41.2 | +4.3% |
| SaveImage | 0 | 35.9 | 35.3 | +1.9% |


## Raw data

- [inpainting_comfyui_0.json](../data/inpainting_comfyui_0.json)
- [inpainting_comfyui_1.json](../data/inpainting_comfyui_1.json)
- [inpainting_comfyui_2.json](../data/inpainting_comfyui_2.json)
- [inpainting_comfyui_3.json](../data/inpainting_comfyui_3.json)
- [inpainting_runtime_0.json](../data/inpainting_runtime_0.json)
- [inpainting_runtime_1.json](../data/inpainting_runtime_1.json)
- [inpainting_runtime_2.json](../data/inpainting_runtime_2.json)
- [inpainting_runtime_3.json](../data/inpainting_runtime_3.json)
