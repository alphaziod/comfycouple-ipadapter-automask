"""Low-level attention execution helpers for the regional-attention patches.

Two small helpers used by the cross-attention patch:

* :func:`try_standard_attention` runs ComfyUI's unmodified attention for one CFG
  branch. Its result is blended with the regional output as a safety fallback.
* :func:`calculate_regional_blend` picks how strongly the regional output should
  replace the standard one for a given mode.
"""

from __future__ import annotations

import torch
from comfy.ldm.modules.attention import optimized_attention

from .context import attention_debug_enabled, resolve_attention_mode

_LOG_PREFIX = "[ComfyCouple SayaPatch v3.3]"


def try_standard_attention(
    query: torch.Tensor,
    original_keys: torch.Tensor | None,
    original_values: torch.Tensor | None,
    chunks: int,
    branch_index: int,
    n_heads: int,
) -> torch.Tensor | None:
    """Run ComfyUI's native attention for one CFG branch, or ``None`` on any issue.

    ``original_keys`` / ``original_values`` are the full (all-branch) projected
    K/V tensors handed to the patch; they are split into ``chunks`` along the
    batch axis and only ``branch_index`` is used. K/V are broadcast/repeated to
    the query batch size and cast to the query dtype when needed.

    Returns ``None`` (rather than raising) whenever the fallback cannot be
    computed, so the caller can simply skip the blend.
    """
    try:
        if original_keys is None or original_values is None:
            return None

        chunks = max(1, int(chunks))
        branch_index = int(branch_index)
        key_parts = original_keys.chunk(chunks, dim=0)
        value_parts = original_values.chunk(chunks, dim=0)
        if branch_index >= len(key_parts) or branch_index >= len(value_parts):
            return None

        keys = key_parts[branch_index]
        values = value_parts[branch_index]

        target_batch = query.shape[0]
        if keys.shape[0] != target_batch:
            if keys.shape[0] == 1:
                keys = keys.expand(target_batch, -1, -1)
                values = values.expand(target_batch, -1, -1)
            else:
                reps = (target_batch + keys.shape[0] - 1) // keys.shape[0]
                keys = keys.repeat(reps, 1, 1)[:target_batch]
                values = values.repeat(reps, 1, 1)[:target_batch]

        if keys.dtype != query.dtype:
            keys = keys.to(query.dtype)
        if values.dtype != query.dtype:
            values = values.to(query.dtype)

        return optimized_attention(query, keys, values, n_heads)
    except Exception as error:
        if attention_debug_enabled():
            print(f"{_LOG_PREFIX} normal attention fallback unavailable: {error}")
        return None


def calculate_regional_blend(
    mode: str,
    masks_aligned: bool = True,
    custom_strength: float | None = None,  # noqa: ARG001 - kept for call-site compatibility
) -> float:
    """Return the blend factor in ``[0, 1]`` for regional vs. standard attention.

    ``1.0`` means "use the regional output only". ``REFINER`` and ``DETAILER``
    keep a small amount of the standard output; a non-aligned mask forces at
    least ``0.98`` regional.
    """
    mode = resolve_attention_mode(mode)
    if mode == "REFINER":
        base = 0.92
    elif mode == "DETAILER":
        base = 0.99
    else:
        base = 1.0
    if not masks_aligned:
        base = max(base, 0.98)
    return max(0.0, min(1.0, float(base)))
