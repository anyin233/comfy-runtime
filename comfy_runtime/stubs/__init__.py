"""Bootstrap stubs for comfy_runtime.

These stubs replace heavy ComfyUI dependencies (server, comfy_aimdo,
latent_preview) with lightweight no-op implementations so that node
code can be imported without pulling in aiohttp, torch, etc.
"""
