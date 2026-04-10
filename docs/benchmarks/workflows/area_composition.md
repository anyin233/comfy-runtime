# area_composition

[← Back to summary](../README.md)

## Stage breakdown (mean +/- stddev, ms)

| Stage | comfy_runtime min | mean | median | stddev | ComfyUI min | mean | median | stddev | Δmean |
|---|---|---|---|---|---|---|---|---|---|
| model_load | 1332.9 | 1341.0 | 1344.4 | 5.7 | 655.7 | 662.5 | 661.7 | 6.0 | +102.4% |
| text_encode | 149.1 | 152.1 | 149.9 | 3.6 | 145.3 | 148.7 | 146.5 | 4.1 | +2.2% |
| conditioning | 0.1 | 0.1 | 0.1 | 0.0 | 0.2 | 0.2 | 0.2 | 0.0 | -75.8% |
| latent_init | 0.1 | 0.1 | 0.1 | 0.0 | 0.2 | 0.2 | 0.2 | 0.0 | -72.9% |
| sample | 2447.6 | 2448.8 | 2448.8 | 1.0 | 2154.0 | 2157.2 | 2154.4 | 4.3 | +13.5% |
| decode | 164.3 | 167.3 | 167.2 | 2.5 | 160.5 | 161.7 | 160.9 | 1.4 | +3.5% |
| save | 49.3 | 49.4 | 49.4 | 0.1 | 49.0 | 49.2 | 49.2 | 0.1 | +0.4% |

| **total** | 4283.7 | 4300.1 | 4307.2 | 11.6 | 3172.8 | 3181.7 | 3177.0 | 9.7 | **+35.2%** |

![Stage breakdown](../figures/stage_breakdown_area_composition.png)

## Memory

| Metric | comfy_runtime (MB) | ComfyUI (MB) | Δ |
|---|---|---|---|
| GPU max allocated | 6465.8 | 2913.5 | +121.9% |
| GPU max reserved  | 6692.0 | 3326.0 | +101.2% |
| Host VmHWM        | 6916.2 | 7016.5 | -1.4% |

## Per-node breakdown (mean, ms)

| Node | Call index | comfy_runtime | ComfyUI | Δ |
|---|---|---|---|---|
| CheckpointLoaderSimple | 0 | 1341.0 | 662.5 | +102.4% |
| CLIPTextEncode | 0 | 111.4 | 110.8 | +0.6% |
| CLIPTextEncode | 1 | 14.2 | 13.1 | +7.9% |
| CLIPTextEncode | 2 | 13.2 | 12.4 | +6.8% |
| CLIPTextEncode | 3 | 13.3 | 12.4 | +6.6% |
| ConditioningSetArea | 0 | 0.0 | 0.1 | -57.9% |
| ConditioningSetArea | 1 | 0.0 | 0.1 | -85.0% |
| ConditioningCombine | 0 | 0.0 | 0.1 | -78.9% |
| ConditioningCombine | 1 | 0.0 | 0.0 | -85.8% |
| EmptyLatentImage | 0 | 0.1 | 0.2 | -72.9% |
| KSampler | 0 | 2448.8 | 2157.2 | +13.5% |
| VAEDecode | 0 | 167.3 | 161.7 | +3.5% |
| SaveImage | 0 | 49.4 | 49.2 | +0.4% |


## Raw data

- [area_composition_comfyui_0.json](../data/area_composition_comfyui_0.json)
- [area_composition_comfyui_1.json](../data/area_composition_comfyui_1.json)
- [area_composition_comfyui_2.json](../data/area_composition_comfyui_2.json)
- [area_composition_comfyui_3.json](../data/area_composition_comfyui_3.json)
- [area_composition_runtime_0.json](../data/area_composition_runtime_0.json)
- [area_composition_runtime_1.json](../data/area_composition_runtime_1.json)
- [area_composition_runtime_2.json](../data/area_composition_runtime_2.json)
- [area_composition_runtime_3.json](../data/area_composition_runtime_3.json)
