"""Minimal options stub for comfy_runtime.

Disables argparse so that importing comfy.cli_args does not
attempt to parse the process command line.
"""

args_parsing = False


def enable_args_parsing(enable=True):
    """No-op — comfy_runtime never uses argparse."""
    pass
