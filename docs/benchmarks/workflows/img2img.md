# img2img

[← Back to summary](../README.md)

## Stage breakdown (mean +/- stddev, ms)

| Stage | comfy_runtime min | mean | median | stddev | ComfyUI min | mean | median | stddev | Δmean |
|---|---|---|---|---|---|---|---|---|---|
| model_load | 547.0 | 549.6 | 549.7 | 2.1 | 676.1 | 679.5 | 678.1 | 3.5 | -19.1% |
| load_input | 7.8 | 8.0 | 7.9 | 0.2 | 8.9 | 9.1 | 9.0 | 0.2 | -12.1% |
| text_encode | 114.3 | 115.3 | 114.7 | 1.2 | 119.2 | 120.3 | 119.7 | 1.1 | -4.1% |
| vae_encode | 252.2 | 255.7 | 255.0 | 3.2 | 252.7 | 254.6 | 255.3 | 1.4 | +0.5% |
| sample | 1058.8 | 1068.5 | 1067.7 | 8.2 | 1020.8 | 1028.4 | 1031.2 | 5.4 | +3.9% |
| decode | 44.3 | 44.9 | 45.0 | 0.4 | 41.1 | 41.8 | 41.9 | 0.6 | +7.3% |
| save | 38.2 | 38.6 | 38.7 | 0.3 | 39.0 | 39.1 | 39.1 | 0.1 | -1.4% |

| **total** | 2076.1 | 2084.1 | 2081.4 | 8.0 | 2163.1 | 2174.7 | 2176.1 | 8.9 | **-4.2%** |

![Stage breakdown](../figures/stage_breakdown_img2img.png)

## Memory

| Metric | comfy_runtime (MB) | ComfyUI (MB) | Δ |
|---|---|---|---|
| GPU max allocated | 6565.6 | 2645.5 | +148.2% |
| GPU max reserved  | 6760.0 | 2908.0 | +132.5% |
| Host VmHWM        | 6958.4 | 7016.5 | -0.8% |

## Per-node breakdown (mean, ms)

| Node | Call index | comfy_runtime | ComfyUI | Δ |
|---|---|---|---|---|
| CheckpointLoaderSimple | 0 | 549.6 | 679.5 | -19.1% |
| LoadImage | 0 | 8.0 | 9.1 | -12.1% |
| VAEEncode | 0 | 255.7 | 254.6 | +0.5% |
| CLIPTextEncode | 0 | 100.9 | 105.8 | -4.6% |
| CLIPTextEncode | 1 | 14.4 | 14.5 | -0.5% |
| KSampler | 0 | 1068.5 | 1028.4 | +3.9% |
| VAEDecode | 0 | 44.9 | 41.8 | +7.3% |
| SaveImage | 0 | 38.6 | 39.1 | -1.4% |


## Raw data

- [img2img_comfyui_0.json](../data/img2img_comfyui_0.json)
- [img2img_comfyui_1.json](../data/img2img_comfyui_1.json)
- [img2img_comfyui_2.json](../data/img2img_comfyui_2.json)
- [img2img_comfyui_3.json](../data/img2img_comfyui_3.json)
- [img2img_runtime_0.json](../data/img2img_runtime_0.json)
- [img2img_runtime_1.json](../data/img2img_runtime_1.json)
- [img2img_runtime_2.json](../data/img2img_runtime_2.json)
- [img2img_runtime_3.json](../data/img2img_runtime_3.json)
