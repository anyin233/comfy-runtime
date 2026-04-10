"""Stub for ``comfy.cldm.cldm`` (legacy ControlNet UNet wrapper)."""
import torch.nn as nn


class ControlNet(nn.Module):
    """Import-compat stub for the legacy CLDM ControlNet class."""

    def __init__(self, *args, **kwargs):
        super().__init__()


class ControlledUnetModel(nn.Module):
    """Import-compat stub for the controlled-UNet wrapper."""

    def __init__(self, *args, **kwargs):
        super().__init__()
