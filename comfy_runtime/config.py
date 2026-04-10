"""Runtime configuration API for comfy-runtime."""


# Snapshot of the most recent successful configure() call. Used to
# short-circuit identical re-configuration (common in long-running
# services that re-initialize on every request).
_LAST_CONFIG: tuple | None = None


def configure(
    models_dir=None,
    output_dir=None,
    input_dir=None,
    temp_dir=None,
    vram_mode=None,
    device=None,
    **kwargs,
):
    """Configure model paths and runtime settings.

    Args:
        models_dir: Base directory that contains Comfy model subdirectories.
        output_dir: Output directory for generated files.
        input_dir: Input directory for source files.
        temp_dir: Temporary directory for intermediate files.
        vram_mode: VRAM mode flag to enable on comfy.cli_args.args.
        device: CUDA device index or ``"cpu"`` to force CPU mode.
        **kwargs: Extra attributes to apply directly to comfy.cli_args.args.

    Returns:
        None.
    """
    global _LAST_CONFIG
    snapshot = (
        models_dir,
        output_dir,
        input_dir,
        temp_dir,
        vram_mode,
        device,
        tuple(sorted(kwargs.items())),
    )
    if _LAST_CONFIG is not None and _LAST_CONFIG == snapshot:
        return

    from comfy.cli_args import args  # type: ignore[import-not-found]

    if vram_mode == "highvram":
        args.highvram = True
    elif vram_mode == "normalvram":
        args.normalvram = True
    elif vram_mode == "lowvram":
        args.lowvram = True
    elif vram_mode == "novram":
        args.novram = True
    elif vram_mode == "cpu":
        args.cpu = True

    if device is not None:
        if isinstance(device, int):
            args.cuda_device = device
        elif device == "cpu":
            args.cpu = True

    for key, val in kwargs.items():
        setattr(args, key, val)

    import folder_paths  # type: ignore[import-not-found]

    if output_dir is not None:
        folder_paths.output_directory = output_dir
    if input_dir is not None:
        folder_paths.input_directory = input_dir
    if temp_dir is not None:
        folder_paths.temp_directory = temp_dir

    if models_dir is not None:
        import os

        model_categories = {
            "checkpoints": {".ckpt", ".safetensors", ".pt", ".bin", ".pth"},
            "configs": {".yaml", ".yml", ".json"},
            "loras": {".safetensors", ".ckpt", ".pt", ".bin"},
            "vae": {".safetensors", ".pt", ".bin", ".ckpt"},
            "text_encoders": {".safetensors", ".pt", ".bin", ".ckpt"},
            "diffusion_models": {".safetensors", ".pt", ".bin", ".ckpt"},
            "clip_vision": {".safetensors", ".pt", ".bin", ".ckpt"},
            "style_models": {".safetensors", ".pt", ".bin", ".ckpt"},
            "embeddings": {".safetensors", ".pt", ".bin", ".ckpt"},
            "controlnet": {".safetensors", ".pt", ".bin", ".ckpt"},
            "unet": {".safetensors", ".pt", ".bin", ".ckpt"},
            "upscale_models": {".safetensors", ".pt", ".bin", ".ckpt"},
        }
        for category, extensions in model_categories.items():
            cat_dir = os.path.join(models_dir, category)
            if category in folder_paths.folder_names_and_paths:
                existing_paths, existing_exts = folder_paths.folder_names_and_paths[
                    category
                ]
                # Dedup guard: don't prepend the same path twice.
                if cat_dir in existing_paths:
                    merged_exts = set(existing_exts) | extensions
                    if merged_exts != set(existing_exts):
                        folder_paths.folder_names_and_paths[category] = (
                            list(existing_paths),
                            merged_exts,
                        )
                    continue
                folder_paths.folder_names_and_paths[category] = (
                    [cat_dir] + list(existing_paths),
                    set(existing_exts) | extensions,
                )
            else:
                folder_paths.folder_names_and_paths[category] = ([cat_dir], extensions)

    _LAST_CONFIG = snapshot


def get_config():
    """Return current configuration as a dictionary.

    Returns:
        dict: Selected runtime directories and device-related flags.
    """
    from comfy.cli_args import args  # type: ignore[import-not-found]
    import folder_paths  # type: ignore[import-not-found]

    return {
        "output_directory": getattr(folder_paths, "output_directory", None),
        "input_directory": getattr(folder_paths, "input_directory", None),
        "temp_directory": getattr(folder_paths, "temp_directory", None),
        "cpu": getattr(args, "cpu", False),
        "highvram": getattr(args, "highvram", False),
        "cuda_device": getattr(args, "cuda_device", 0),
    }
