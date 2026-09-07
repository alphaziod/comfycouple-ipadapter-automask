"""Standalone left/right (or top/bottom) ownership-mask helper.

Used by callers that only need a hard split mask outside the couple node's own
mask geometry.
"""

from __future__ import annotations

from typing import Any

try:
    import torch
except Exception:  # pragma: no cover - torch is always present under ComfyUI
    torch = None

_MIN_SIDE = 8
_DEFAULT_SIDE = 1024


def create_region_mask(
    width: int,
    height: int,
    orientation: str = "horizontal",
    center: float = 0.5,
    side: int = 0,
) -> Any:
    """Return a ``[1, height, width]`` 0/1 mask for one side of a split.

    ``orientation`` is ``"horizontal"`` (split on ``width``) or ``"vertical"``
    (split on ``height``); ``center`` in ``[0, 1]`` is the split position;
    ``side`` selects the first (``0``) or second (non-zero) region. Returns
    ``None`` when torch is unavailable.
    """
    if torch is None:
        return None

    mask_w = max(int(width or _DEFAULT_SIDE), _MIN_SIDE)
    mask_h = max(int(height or _DEFAULT_SIDE), _MIN_SIDE)
    split = min(max(float(center if center is not None else 0.5), 0.0), 1.0)

    mask = torch.zeros((1, mask_h, mask_w), dtype=torch.float32)
    if str(orientation) == "vertical":
        cut = int(mask_h * split)
        if side == 0:
            mask[:, :cut, :] = 1.0
        else:
            mask[:, cut:, :] = 1.0
    else:
        cut = int(mask_w * split)
        if side == 0:
            mask[:, :, :cut] = 1.0
        else:
            mask[:, :, cut:] = 1.0
    return mask
