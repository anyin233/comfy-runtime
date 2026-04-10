"""Import-compat stub for ``comfy_extras.nodes_hooks``.

ComfyUI's nodes_hooks module exposes the public node classes for
hook creation and binding (CreateHookLora, CreateHookModelAsLora,
SetClipHooks, ConditioningSetProperties, ...).  Custom nodes import
these for hook plumbing.

Real hook execution is handled by
:mod:`comfy_runtime.compat.comfy.hooks` (see Task 3.5).
"""


class CreateHookLora:
    """Stub for the CreateHookLora node."""

    pass


class CreateHookModelAsLora:
    pass


class CreateHookModelAsLoraTest:
    pass


class SetHookKeyframes:
    pass


class CreateHookKeyframe:
    pass


class CreateHookKeyframesInterpolated:
    pass


class CreateHookKeyframesFromFloats:
    pass


class CombineHooks:
    pass


class CombineHooksFour:
    pass


class CombineHooksEight:
    pass


class SetClipHooks:
    pass


class ConditioningSetProperties:
    pass


class ConditioningSetPropertiesAndCombine:
    pass


class PairConditioningSetProperties:
    pass


class PairConditioningSetPropertiesAndCombine:
    pass
