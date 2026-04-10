# sd15_text_to_image

[← Back to summary](../README.md)

## Stage breakdown (mean +/- stddev, ms)

| Stage | comfy_runtime min | mean | median | stddev | ComfyUI min | mean | median | stddev | Δmean |
|---|---|---|---|---|---|---|---|---|---|
| model_load | 532.8 | 545.1 | 549.1 | 8.8 | 664.1 | 679.0 | 681.5 | 11.2 | -19.7% |
| text_encode | 118.7 | 124.6 | 122.7 | 5.7 | 119.5 | 121.9 | 122.8 | 1.7 | +2.2% |
| latent_init | 0.0 | 0.1 | 0.1 | 0.0 | 0.2 | 0.2 | 0.2 | 0.0 | -73.7% |
| sample | 1100.8 | 1127.0 | 1136.7 | 18.7 | 1087.4 | 1096.5 | 1099.9 | 6.5 | +2.8% |
| decode | 143.9 | 145.2 | 144.7 | 1.3 | 139.8 | 141.2 | 141.1 | 1.2 | +2.8% |
| save | 34.6 | 35.9 | 36.6 | 0.9 | 36.0 | 36.6 | 36.8 | 0.5 | -1.9% |

| **total** | 1934.2 | 1981.2 | 1997.5 | 33.8 | 2051.2 | 2076.8 | 2083.4 | 18.8 | **-4.6%** |

![Stage breakdown](../figures/stage_breakdown_sd15_text_to_image.png)

## Memory

| Metric | comfy_runtime (MB) | ComfyUI (MB) | Δ |
|---|---|---|---|
| GPU max allocated | 5007.0 | 2639.5 | +89.7% |
| GPU max reserved  | 5202.0 | 2896.0 | +79.6% |
| Host VmHWM        | 6958.2 | 7016.3 | -0.8% |

## Per-node breakdown (mean, ms)

| Node | Call index | comfy_runtime | ComfyUI | Δ |
|---|---|---|---|---|
| CheckpointLoaderSimple | 0 | 545.1 | 679.0 | -19.7% |
| CLIPTextEncode | 0 | 111.0 | 108.4 | +2.4% |
| CLIPTextEncode | 1 | 13.6 | 13.5 | +0.3% |
| EmptyLatentImage | 0 | 0.1 | 0.2 | -73.7% |
| KSampler | 0 | 1127.0 | 1096.5 | +2.8% |
| VAEDecode | 0 | 145.2 | 141.2 | +2.8% |
| SaveImage | 0 | 35.9 | 36.6 | -1.9% |


## Raw data

- [sd15_text_to_image_comfyui_0.json](../data/sd15_text_to_image_comfyui_0.json)
- [sd15_text_to_image_comfyui_1.json](../data/sd15_text_to_image_comfyui_1.json)
- [sd15_text_to_image_comfyui_2.json](../data/sd15_text_to_image_comfyui_2.json)
- [sd15_text_to_image_comfyui_3.json](../data/sd15_text_to_image_comfyui_3.json)
- [sd15_text_to_image_runtime_0.json](../data/sd15_text_to_image_runtime_0.json)
- [sd15_text_to_image_runtime_1.json](../data/sd15_text_to_image_runtime_1.json)
- [sd15_text_to_image_runtime_2.json](../data/sd15_text_to_image_runtime_2.json)
- [sd15_text_to_image_runtime_3.json](../data/sd15_text_to_image_runtime_3.json)
