"""Conditioning-context helpers for the regional-attention patches.

This module groups the small, pure helpers shared by the cross-attention and
self-attention patches:

* token-length padding so several regional prompt banks can be concatenated,
* attention-mode resolution (``GEN`` / ``REFINER`` / ``DETAILER`` / ``AUTO``),
* the "listening strength" curve that softens ownership masks in ambiguous areas.

Nothing here touches ComfyUI state; every function is deterministic and safe to
unit-test in isolation.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any

import torch

_DEBUG_ENV_VAR = "SAYA_COUPLE_DEBUG"
_LOG_PREFIX = "[ComfyCouple SayaPatch v3.3]"

#: Valid values for the regional-attention "mode".
ATTENTION_MODES = ("GEN", "REFINER", "DETAILER", "AUTO")

#: Default listening strength per resolved mode.
_DEFAULT_REGION_STRENGTH = {"GEN": 1.0, "REFINER": 0.8, "DETAILER": 0.94, "AUTO": 1.0}

#: A pass whose largest side is at or below this many pixels is treated as a
#: detailer crop when the mode is ``AUTO``.
_AUTO_DETAILER_MAX_SIDE = 768


def pad_context_token_lengths(tensors: Iterable[torch.Tensor]) -> list[torch.Tensor]:
    """Right-pad rank>=3 context tensors so they share the longest token length.

    Padding repeats each tensor's last token (``t[:, -1:, ...]``). Tensors that
    are not rank-3 embeddings, or that already have the maximum length, are
    returned untouched. With fewer than two tensors there is nothing to align.
    """
    tensors = list(tensors)
    if len(tensors) < 2:
        return tensors

    max_tokens = max(
        (t.shape[1] for t in tensors if hasattr(t, "ndim") and t.ndim >= 3),
        default=None,
    )
    if max_tokens is None:
        return tensors

    padded: list[torch.Tensor] = []
    for tensor in tensors:
        if hasattr(tensor, "ndim") and tensor.ndim >= 3 and tensor.shape[1] < max_tokens:
            pad_len = max_tokens - tensor.shape[1]
            pad = tensor[:, -1:, ...].expand(tensor.shape[0], pad_len, *tensor.shape[2:]).clone()
            tensor = torch.cat((tensor, pad), dim=1)
        padded.append(tensor)
    return padded


def concatenate_context_tensors(tensors: Iterable[torch.Tensor], dim: int = 0) -> torch.Tensor:
    """Concatenate context tensors along ``dim`` after equalizing their token length."""
    return torch.cat(pad_context_token_lengths(tensors), dim=dim)


def attention_debug_enabled(local_debug: bool = False) -> bool:
    """Return whether regional-attention debug logging should be emitted.

    Enabled either by the per-node ``saya_debug`` flag (passed in as
    ``local_debug``) or by the ``SAYA_COUPLE_DEBUG=1`` environment variable.
    """
    if local_debug:
        return True
    return os.environ.get(_DEBUG_ENV_VAR, "0") == "1"


def resolve_attention_mode(
    mode: str,
    original_shape: Any = None,
    qkv: Any = None,  # noqa: ARG001 - reserved for call-site compatibility
) -> str:
    """Normalize ``mode`` and, for ``AUTO``, infer ``DETAILER`` from the pass size.

    ``qkv`` is accepted (and ignored) so historical call sites that pass it
    positionally keep working.
    """
    mode = str(mode or "GEN").upper()
    if mode not in ATTENTION_MODES:
        mode = "GEN"
    if mode != "AUTO":
        return mode

    try:
        if original_shape is not None:
            height = int(original_shape[-2])
            width = int(original_shape[-1])
            if max(height, width) <= _AUTO_DETAILER_MAX_SIDE:
                return "DETAILER"
    except Exception:
        pass
    return "GEN"


def default_region_strength(mode: str) -> float:
    """Return the default regional listening strength for ``mode``."""
    return _DEFAULT_REGION_STRENGTH.get(resolve_attention_mode(mode), 1.0)


def resolve_region_strength(mode: str, custom_strength: float | None = None) -> float:
    """Clamp a caller-supplied listening strength to ``[0, 1.5]`` or use the default.

    A negative ``custom_strength`` (the node's "use the default" sentinel) falls
    back to :func:`default_region_strength`.
    """
    try:
        if custom_strength is not None and float(custom_strength) >= 0:
            return max(0.0, min(1.5, float(custom_strength)))
    except Exception:
        pass
    return default_region_strength(mode)


def apply_region_strength(masks_v: torch.Tensor, strength: float) -> torch.Tensor:
    """Re-shape normalized ownership masks according to the listening ``strength``.

    Input/output shape is ``[regions, batch, tokens, channels]`` and each column
    stays a probability distribution over regions.

    * ``strength == 1``  -> masks are only renormalized (identity behaviour).
    * ``strength < 1``   -> masks are blended toward the uniform distribution,
      more so where they are already ambiguous (low peak probability). This
      "opens up" uncertain areas without leaking prompt influence into regions
      that are clearly owned.
    * ``strength > 1``   -> masks are sharpened by raising them to a power.

    On any error the input is returned unchanged (this runs in the attention hot
    path and must never raise).
    """
    try:
        if not isinstance(masks_v, torch.Tensor) or masks_v.ndim != 4:
            return masks_v
        region_count = max(1, int(masks_v.shape[0]))
        strength = max(0.0, min(1.5, float(strength)))

        masks_v = masks_v / masks_v.sum(dim=0, keepdim=True).clamp_min(1e-06)
        if region_count <= 1 or abs(strength - 1.0) < 1e-06:
            return masks_v

        neutral = torch.ones_like(masks_v) / region_count
        if strength < 1.0:
            peak_probability = masks_v.max(dim=0, keepdim=True).values
            neutral_peak = 1.0 / region_count
            denom = max(1e-06, 1.0 - neutral_peak)
            ambiguity = ((1.0 - peak_probability) / denom).clamp(0.0, 1.0)
            edge_mix = (1.0 - strength) * ambiguity
            out = masks_v * (1.0 - edge_mix) + neutral * edge_mix
        else:
            power = 1.0 + (strength - 1.0) * 2.0
            out = masks_v.clamp_min(1e-06).pow(power)
        return out / out.sum(dim=0, keepdim=True).clamp_min(1e-06)
    except Exception as error:
        print(f"{_LOG_PREFIX} strength apply failed: {error}")
        return masks_v
