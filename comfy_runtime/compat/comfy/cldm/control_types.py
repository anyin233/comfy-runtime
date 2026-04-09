"""Control type definitions for Union ControlNet.

MIT reimplementation of comfy.cldm.control_types — provides the
UNION_CONTROLNET_TYPES mapping used by ControlNet nodes to identify
different control modalities.
"""

# ---------------------------------------------------------------------------
# Union ControlNet type registry
# ---------------------------------------------------------------------------

UNION_CONTROLNET_TYPES: dict = {
    "openpose": {
        "index": 0,
        "name": "OpenPose",
        "description": "Human pose estimation",
    },
    "depth": {
        "index": 1,
        "name": "Depth",
        "description": "Depth map conditioning",
    },
    "hed": {
        "index": 2,
        "name": "HED/Scribble/PIDI",
        "description": "Soft edge detection",
    },
    "canny": {
        "index": 3,
        "name": "Canny",
        "description": "Canny edge detection",
    },
    "normal": {
        "index": 4,
        "name": "Normal Map",
        "description": "Surface normal conditioning",
    },
    "segment": {
        "index": 5,
        "name": "Segmentation",
        "description": "Semantic segmentation map",
    },
    "tile": {
        "index": 6,
        "name": "Tile",
        "description": "Tile/upscale conditioning",
    },
    "repaint": {
        "index": 7,
        "name": "Repaint/Inpaint",
        "description": "Inpainting conditioning",
    },
}
