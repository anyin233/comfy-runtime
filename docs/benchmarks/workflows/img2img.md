# img2img

[← Back to summary](../README.md)

## Stage breakdown (mean +/- stddev, ms)

| Stage | comfy_runtime min | mean | median | stddev | ComfyUI min | mean | median | stddev | Δmean |
|---|---|---|---|---|---|---|---|---|---|
| model_load | 1325.8 | 1331.1 | 1328.0 | 6.0 | 644.9 | 650.5 | 651.0 | 4.4 | +104.6% |
| load_input | 7.3 | 7.7 | 7.8 | 0.4 | 8.9 | 9.0 | 9.0 | 0.1 | -14.4% |
| text_encode | 111.0 | 114.2 | 115.8 | 2.3 | 117.1 | 117.9 | 117.8 | 0.6 | -3.1% |
| vae_encode | 245.4 | 253.1 | 255.6 | 5.6 | 248.6 | 252.6 | 253.9 | 2.9 | +0.2% |
| sample | 1274.6 | 1303.7 | 1313.4 | 20.9 | 994.4 | 1014.1 | 1021.8 | 14.0 | +28.6% |
| decode | 42.8 | 43.1 | 43.1 | 0.2 | 41.1 | 41.1 | 41.1 | 0.1 | +4.7% |
| save | 36.8 | 38.1 | 38.5 | 0.9 | 38.1 | 38.5 | 38.6 | 0.3 | -1.0% |

| **total** | 3193.6 | 3237.4 | 3253.2 | 31.4 | 2108.6 | 2125.5 | 2128.3 | 12.9 | **+52.3%** |

![Stage breakdown](../figures/stage_breakdown_img2img.png)

## Memory

| Metric | comfy_runtime (MB) | ComfyUI (MB) | Δ |
|---|---|---|---|
| GPU max allocated | 6565.6 | 2645.5 | +148.2% |
| GPU max reserved  | 6760.0 | 2908.0 | +132.5% |
| Host VmHWM        | 6914.8 | 7016.0 | -1.4% |

## Per-node breakdown (mean, ms)

| Node | Call index | comfy_runtime | ComfyUI | Δ |
|---|---|---|---|---|
| CheckpointLoaderSimple | 0 | 1331.1 | 650.5 | +104.6% |
| LoadImage | 0 | 7.7 | 9.0 | -14.4% |
| VAEEncode | 0 | 253.1 | 252.6 | +0.2% |
| CLIPTextEncode | 0 | 100.1 | 104.3 | -4.1% |
| CLIPTextEncode | 1 | 14.2 | 13.6 | +4.5% |
| KSampler | 0 | 1303.7 | 1014.1 | +28.6% |
| VAEDecode | 0 | 43.1 | 41.1 | +4.7% |
| SaveImage | 0 | 38.1 | 38.5 | -1.0% |


## Raw data

- [img2img_comfyui_0.json](../data/img2img_comfyui_0.json)
- [img2img_comfyui_1.json](../data/img2img_comfyui_1.json)
- [img2img_comfyui_2.json](../data/img2img_comfyui_2.json)
- [img2img_comfyui_3.json](../data/img2img_comfyui_3.json)
- [img2img_runtime_0.json](../data/img2img_runtime_0.json)
- [img2img_runtime_1.json](../data/img2img_runtime_1.json)
- [img2img_runtime_2.json](../data/img2img_runtime_2.json)
- [img2img_runtime_3.json](../data/img2img_runtime_3.json)
