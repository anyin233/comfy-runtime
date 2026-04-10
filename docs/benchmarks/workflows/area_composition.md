# area_composition

[← Back to summary](../README.md)

## Stage breakdown (mean +/- stddev, ms)

| Stage | comfy_runtime min | mean | median | stddev | ComfyUI min | mean | median | stddev | Δmean |
|---|---|---|---|---|---|---|---|---|---|
| model_load | 540.4 | 551.0 | 556.2 | 7.6 | 665.7 | 673.9 | 668.4 | 9.8 | -18.2% |
| text_encode | 145.1 | 148.5 | 149.0 | 2.6 | 145.0 | 146.6 | 146.7 | 1.3 | +1.3% |
| conditioning | 0.1 | 0.1 | 0.1 | 0.0 | 0.2 | 0.3 | 0.2 | 0.0 | -76.6% |
| latent_init | 0.1 | 0.1 | 0.1 | 0.0 | 0.2 | 0.2 | 0.2 | 0.0 | -73.4% |
| sample | 2152.1 | 2165.2 | 2168.7 | 9.6 | 2153.4 | 2164.4 | 2166.1 | 8.4 | +0.0% |
| decode | 165.5 | 177.9 | 166.2 | 17.1 | 161.4 | 163.3 | 162.9 | 1.7 | +9.0% |
| save | 47.2 | 48.3 | 48.9 | 0.8 | 49.7 | 50.2 | 49.8 | 0.7 | -3.8% |

| **total** | 3078.4 | 3094.6 | 3097.4 | 12.2 | 3185.9 | 3200.8 | 3193.5 | 15.9 | **-3.3%** |

![Stage breakdown](../figures/stage_breakdown_area_composition.png)

## Memory

| Metric | comfy_runtime (MB) | ComfyUI (MB) | Δ |
|---|---|---|---|
| GPU max allocated | 6465.8 | 2913.5 | +121.9% |
| GPU max reserved  | 6692.0 | 3326.0 | +101.2% |
| Host VmHWM        | 6958.6 | 7016.2 | -0.8% |

## Per-node breakdown (mean, ms)

| Node | Call index | comfy_runtime | ComfyUI | Δ |
|---|---|---|---|---|
| CheckpointLoaderSimple | 0 | 551.0 | 673.9 | -18.2% |
| CLIPTextEncode | 0 | 108.3 | 108.1 | +0.2% |
| CLIPTextEncode | 1 | 13.9 | 13.3 | +4.2% |
| CLIPTextEncode | 2 | 13.2 | 12.6 | +4.6% |
| CLIPTextEncode | 3 | 13.1 | 12.5 | +4.8% |
| ConditioningSetArea | 0 | 0.0 | 0.1 | -57.9% |
| ConditioningSetArea | 1 | 0.0 | 0.1 | -86.0% |
| ConditioningCombine | 0 | 0.0 | 0.1 | -78.8% |
| ConditioningCombine | 1 | 0.0 | 0.1 | -88.3% |
| EmptyLatentImage | 0 | 0.1 | 0.2 | -73.4% |
| KSampler | 0 | 2165.2 | 2164.4 | +0.0% |
| VAEDecode | 0 | 177.9 | 163.3 | +9.0% |
| SaveImage | 0 | 48.3 | 50.2 | -3.8% |


## Raw data

- [area_composition_comfyui_0.json](../data/area_composition_comfyui_0.json)
- [area_composition_comfyui_1.json](../data/area_composition_comfyui_1.json)
- [area_composition_comfyui_2.json](../data/area_composition_comfyui_2.json)
- [area_composition_comfyui_3.json](../data/area_composition_comfyui_3.json)
- [area_composition_runtime_0.json](../data/area_composition_runtime_0.json)
- [area_composition_runtime_1.json](../data/area_composition_runtime_1.json)
- [area_composition_runtime_2.json](../data/area_composition_runtime_2.json)
- [area_composition_runtime_3.json](../data/area_composition_runtime_3.json)
