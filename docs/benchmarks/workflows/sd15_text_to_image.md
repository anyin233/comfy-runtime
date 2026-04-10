# sd15_text_to_image

[← Back to summary](../README.md)

## Stage breakdown (mean +/- stddev, ms)

| Stage | comfy_runtime min | mean | median | stddev | ComfyUI min | mean | median | stddev | Δmean |
|---|---|---|---|---|---|---|---|---|---|
| model_load | 1262.6 | 1275.8 | 1278.1 | 10.0 | 665.2 | 667.4 | 665.8 | 2.7 | +91.2% |
| text_encode | 121.8 | 125.9 | 123.4 | 4.7 | 120.4 | 122.7 | 123.5 | 1.6 | +2.6% |
| latent_init | 0.0 | 0.1 | 0.1 | 0.0 | 0.2 | 0.2 | 0.2 | 0.0 | -73.6% |
| sample | 1407.5 | 1418.3 | 1422.0 | 7.8 | 1098.1 | 1113.7 | 1116.4 | 11.8 | +27.3% |
| decode | 145.2 | 145.6 | 145.7 | 0.2 | 143.3 | 144.1 | 143.3 | 1.0 | +1.0% |
| save | 35.6 | 36.0 | 35.8 | 0.4 | 36.3 | 36.4 | 36.4 | 0.0 | -0.9% |

| **total** | 3188.0 | 3201.9 | 3196.1 | 14.4 | 2068.0 | 2085.8 | 2094.1 | 12.6 | **+53.5%** |

![Stage breakdown](../figures/stage_breakdown_sd15_text_to_image.png)

## Memory

| Metric | comfy_runtime (MB) | ComfyUI (MB) | Δ |
|---|---|---|---|
| GPU max allocated | 5007.0 | 2639.5 | +89.7% |
| GPU max reserved  | 5202.0 | 2896.0 | +79.6% |
| Host VmHWM        | 6915.6 | 7016.8 | -1.4% |

## Per-node breakdown (mean, ms)

| Node | Call index | comfy_runtime | ComfyUI | Δ |
|---|---|---|---|---|
| CheckpointLoaderSimple | 0 | 1275.8 | 667.4 | +91.2% |
| CLIPTextEncode | 0 | 112.1 | 109.2 | +2.6% |
| CLIPTextEncode | 1 | 13.8 | 13.5 | +2.2% |
| EmptyLatentImage | 0 | 0.1 | 0.2 | -73.6% |
| KSampler | 0 | 1418.3 | 1113.7 | +27.3% |
| VAEDecode | 0 | 145.6 | 144.1 | +1.0% |
| SaveImage | 0 | 36.0 | 36.4 | -0.9% |


## Raw data

- [sd15_text_to_image_comfyui_0.json](../data/sd15_text_to_image_comfyui_0.json)
- [sd15_text_to_image_comfyui_1.json](../data/sd15_text_to_image_comfyui_1.json)
- [sd15_text_to_image_comfyui_2.json](../data/sd15_text_to_image_comfyui_2.json)
- [sd15_text_to_image_comfyui_3.json](../data/sd15_text_to_image_comfyui_3.json)
- [sd15_text_to_image_runtime_0.json](../data/sd15_text_to_image_runtime_0.json)
- [sd15_text_to_image_runtime_1.json](../data/sd15_text_to_image_runtime_1.json)
- [sd15_text_to_image_runtime_2.json](../data/sd15_text_to_image_runtime_2.json)
- [sd15_text_to_image_runtime_3.json](../data/sd15_text_to_image_runtime_3.json)
