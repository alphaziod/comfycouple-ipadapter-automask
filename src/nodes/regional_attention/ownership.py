"""Shared tri-state ownership utilities for Saya Comfy Couple v21.7E.

The v21.7E experiment deliberately separates spatial ownership from private-token
injection.  Ownership has three explicit channels: Person 1, Person 2 and Unknown.
Unknown is not interpreted as Person 2 and never becomes a hidden complement rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


DEFAULT_UNKNOWN_BAND = 0.15
CROSS_WRONG_OWNER_STRENGTH = 8.0
CROSS_UNKNOWN_PENALTY = 3.0


@dataclass(frozen=True, slots=True)
class OwnershipState:
    """Tri-state ownership probabilities for flattened spatial query positions."""

    p1: torch.Tensor
    p2: torch.Tensor
    unknown: torch.Tensor

    @property
    def confidence(self) -> torch.Tensor:
        """Return the known-owner confidence ``1 - unknown``."""
        return (1.0 - self.unknown).clamp(0.0, 1.0)


def project_ownership_mask(
    mask: torch.Tensor,
    *,
    batch: int,
    grid_h: int,
    grid_w: int,
    device: torch.device,
) -> torch.Tensor:
    """Project one source ownership mask to a flattened attention query grid."""
    if not isinstance(mask, torch.Tensor):
        raise TypeError("ownership mask must be a tensor")
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim != 3:
        raise ValueError(f"unexpected ownership mask shape: {tuple(mask.shape)}")
    if batch <= 0 or grid_h <= 0 or grid_w <= 0:
        raise ValueError("batch and target grid dimensions must be positive")

    mask = mask.to(device=device, dtype=torch.float32)
    if int(mask.shape[0]) == 1 and batch > 1:
        mask = mask.expand(batch, -1, -1)
    elif int(mask.shape[0]) != batch:
        repeats = (batch + int(mask.shape[0]) - 1) // int(mask.shape[0])
        mask = mask.repeat(repeats, 1, 1)[:batch]

    resized = F.interpolate(
        mask.unsqueeze(1),
        size=(grid_h, grid_w),
        mode="bilinear",
        align_corners=False,
    )[:, 0]
    return resized.reshape(batch, grid_h * grid_w).clamp(0.0, 1.0)


def build_tristate_ownership(
    score_p1: torch.Tensor,
    score_p2: torch.Tensor,
    *,
    unknown_band: float = DEFAULT_UNKNOWN_BAND,
) -> OwnershipState:
    """Convert two independent spatial supports into ``P1/P2/Unknown`` channels.

    The two incoming supports are intentionally treated independently.  No
    ``P2 = 1 - P1`` operation appears here.  Ambiguous, weak or overlapping
    evidence is moved into ``unknown`` instead of being handed to the other owner.
    """
    if score_p1.shape != score_p2.shape:
        raise ValueError("P1 and P2 ownership scores must have identical shapes")
    if score_p1.ndim != 2:
        raise ValueError("ownership scores must have shape [batch, query_tokens]")

    band = max(0.0, min(0.95, float(unknown_band)))
    p1_raw = torch.nan_to_num(score_p1.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    p2_raw = torch.nan_to_num(score_p2.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

    support = p1_raw + p2_raw
    safe_support = support.clamp_min(1e-6)
    share_p1 = p1_raw / safe_support
    share_p2 = p2_raw / safe_support

    dominance = (p1_raw - p2_raw).abs() / safe_support
    certainty = ((dominance - band) / max(1e-6, 1.0 - band)).clamp(0.0, 1.0)
    # Weak total support is itself uncertain.  Complementary masks have support=1,
    # while background-like or partially missing support remains explicitly unknown.
    certainty = certainty * support.clamp(0.0, 1.0)

    p1 = (share_p1 * certainty).clamp(0.0, 1.0)
    p2 = (share_p2 * certainty).clamp(0.0, 1.0)
    unknown = (1.0 - p1 - p2).clamp(0.0, 1.0)

    # Keep the simplex numerically exact after fp32 arithmetic.
    total = (p1 + p2 + unknown).clamp_min(1e-6)
    return OwnershipState(p1=p1 / total, p2=p2 / total, unknown=unknown / total)


def build_cross_attention_bias(
    ownership: OwnershipState,
    *,
    base_tokens: int,
    p1_tokens: int,
    p2_tokens: int,
    dtype: torch.dtype,
    wrong_owner_strength: float = CROSS_WRONG_OWNER_STRENGTH,
    unknown_penalty: float = CROSS_UNKNOWN_PENALTY,
) -> torch.Tensor:
    """Build finite Base/P1/P2 pre-softmax bias from tri-state ownership.

    Confirmed opposite ownership receives the strongest penalty.  Unknown regions
    penalise both private banks only moderately, leaving the Base bank at zero.
    """
    if p1_tokens <= 0 or p2_tokens <= 0:
        raise ValueError("private token groups cannot be empty")
    if ownership.p1.shape != ownership.p2.shape or ownership.p1.shape != ownership.unknown.shape:
        raise ValueError("tri-state ownership tensors must have identical shapes")

    strength = max(0.0, float(wrong_owner_strength))
    unknown_strength = max(0.0, float(unknown_penalty))
    confidence = ownership.confidence

    penalty_p1 = -(strength * confidence * ownership.p2 + unknown_strength * ownership.unknown)
    penalty_p2 = -(strength * confidence * ownership.p1 + unknown_strength * ownership.unknown)

    batch, query_tokens = ownership.p1.shape
    parts: list[torch.Tensor] = []
    if base_tokens > 0:
        parts.append(
            torch.zeros(
                (batch, query_tokens, base_tokens),
                device=ownership.p1.device,
                dtype=dtype,
            )
        )
    parts.append(penalty_p1.to(dtype=dtype).unsqueeze(-1).expand(-1, -1, p1_tokens))
    parts.append(penalty_p2.to(dtype=dtype).unsqueeze(-1).expand(-1, -1, p2_tokens))
    return torch.cat(parts, dim=-1).contiguous()


def build_self_attention_bias(
    ownership: OwnershipState,
    *,
    strength: float,
    dtype: torch.dtype,
    branch_count: int = 1,
) -> torch.Tensor:
    """Build the low-resolution self-attention cross-owner penalty.

    The same P1/P2/Unknown state used by cross-attention is consumed here.  Unknown
    positions stay open, same-owner communication stays open, and the diagonal is
    forced to zero exactly.
    """
    lam = max(0.0, float(strength))
    confidence = ownership.confidence
    p1 = ownership.p1
    p2 = ownership.p2

    cross_owner = p1.unsqueeze(2) * p2.unsqueeze(1)
    cross_owner = cross_owner + p2.unsqueeze(2) * p1.unsqueeze(1)
    pair_confidence = confidence.unsqueeze(2) * confidence.unsqueeze(1)
    bias = (-lam * pair_confidence * cross_owner).to(dtype=dtype)
    bias.diagonal(dim1=1, dim2=2).zero_()
    if branch_count > 1:
        bias = bias.repeat(branch_count, 1, 1)
    return bias
