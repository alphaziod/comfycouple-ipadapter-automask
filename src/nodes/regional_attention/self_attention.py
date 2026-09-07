"""Experimental V1 self-attention (attn1) regional binding patch.

V1 sits strictly on top of V0:

* V0 keeps owning cross-attention (attn2) with a joint Base/P1/P2 pre-softmax
  query-token gate. Nothing in this module changes that.
* V1 adds a *soft*, finite, pre-softmax bias on self-attention (attn1) between
  query/key positions that the spatial ownership prior assigns to clearly
  different owners. It never uses a hard -inf wall, always keeps the diagonal,
  the ambiguous/contact band and same-owner traffic fully open, and only runs on
  a bounded set of low/mid resolution blocks.

v21.7E derives one explicit ``P1/P2/Unknown`` state from the existing spatial
supports and uses that same definition for self- and cross-attention.  It deliberately
does not implement persistent or semantic instance tracking.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Self

import torch
from comfy.ldm.modules.attention import optimized_attention

from .context import attention_debug_enabled
from .masks import factor_token_grid
from .ownership import (
    build_self_attention_bias,
    build_tristate_ownership,
    project_ownership_mask,
)

# ``attn1_strength`` (0..1.5 on the UI) is mapped to a penalty in nats:
#   lambda = strength * _LAMBDA_SCALE
# A cross-owner query/key pair then has its softmax weight multiplied by
# exp(-lambda). strength 0.6 -> x0.05, strength 1.0 -> x0.0067. Never zero.
_LAMBDA_SCALE = 5.0

# SDXL 32x32 attention level (N=1024 at 1024px). Highest leverage for global
# morphology / identity homogenisation, and cheap: a [B, N, N] bias is ~4 MiB.
SDXL_LOWRES_SELF_BLOCKS: tuple[tuple[str, int, int], ...] = (
    ("middle", 0, 10),
    ("input", 7, 10),
    ("input", 8, 10),
    ("output", 0, 10),
    ("output", 1, 10),
    ("output", 2, 10),
)

# SDXL 64x64 attention level (N=4096). Only installed when lowres_only=False, and
# even then the runtime ``max_tokens`` fuse skips the bias unless raised.
SDXL_HIRES_SELF_BLOCKS: tuple[tuple[str, int, int], ...] = (
    ("input", 4, 2),
    ("input", 5, 2),
    ("output", 3, 2),
    ("output", 4, 2),
    ("output", 5, 2),
)

# V1.1 deliberately touches only the SDXL decoder-side high-resolution blocks.
# These are the local reconstruction blocks where extra spatial identity pressure
# is most likely to help small anatomy/morphology details, while avoiding the
# encoder-side high-res cost. Six transformer blocks total.
SDXL_HIRES_TARGET_SELF_BLOCKS: tuple[tuple[str, int, int], ...] = (
    ("output", 3, 2),
    ("output", 4, 2),
    ("output", 5, 2),
)

# SD1.5 deep attention levels (middle 8x8 + 16x16 blocks, N <= 256).
SD15_LOWRES_SELF_BLOCKS: tuple[tuple[str, int, int], ...] = (
    ("middle", 0, 1),
    ("input", 7, 1),
    ("input", 8, 1),
    ("output", 3, 1),
    ("output", 4, 1),
    ("output", 5, 1),
)


@dataclass(frozen=True, slots=True)
class SelfBindingConfig:
    """Immutable V1 self-attention parameters resolved from the node inputs."""

    strength: float = 0.6
    max_tokens: int = 1024
    ambiguous_band: float = 0.15
    lowres_only: bool = True
    min_tokens: int = 0
    profile: str = "V1"

    @classmethod
    def from_mapping(cls, raw: Any) -> "SelfBindingConfig":
        """Build a config from an arbitrary mapping, clamping every value.

        Non-mapping input, missing keys and unparsable values all fall back to
        the field defaults, so this never raises.
        """
        data = raw if isinstance(raw, dict) else {}

        def clamped_float(key: str, default: float, low: float, high: float) -> float:
            try:
                return max(low, min(high, float(data.get(key, default))))
            except (TypeError, ValueError):
                return default

        def clamped_int(key: str, default: int, low: int, high: int) -> int:
            try:
                return max(low, min(high, int(round(float(data.get(key, default))))))
            except (TypeError, ValueError):
                return default

        return cls(
            strength=clamped_float("attn1_strength", 0.6, 0.0, 1.5),
            max_tokens=clamped_int("attn1_max_tokens", 1024, 256, 16384),
            ambiguous_band=clamped_float("attn1_ambiguous_band", 0.15, 0.02, 0.45),
            lowres_only=bool(data.get("attn1_apply_lowres_only", True)),
        )


class SelfAttentionBindingPatch:
    """Callable attn1 replacement that applies a soft cross-owner bias."""

    def __init__(self, owner: Any, module: Any, config: SelfBindingConfig) -> None:
        """Bind the patch to the regional-attention node, the module and its config."""
        self.owner = owner
        self.module = module
        self.config = config
        # Diagnostic de-dup only. The object is recreated on every couple-node
        # run, so this never carries state across generations.
        self._logged: set[Any] = set()

    def __call__(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, extra_options: dict[str, Any]
    ) -> torch.Tensor:
        """Run self-attention, adding a cross-owner bias when one can be built.

        Any failure while building the bias falls back to native self-attention;
        the experimental path must never break the sampler.
        """
        heads = extra_options["n_heads"]
        attn_precision = getattr(self.module, "attn_precision", None)
        bias = None
        try:
            bias = self._build_bias(q, extra_options)
        except Exception as error:  # never break the sampler on the experimental path
            self._log_once(
                ("error", type(error).__name__),
                f"self-attention bias unavailable ({error}); using native self-attention",
            )
            bias = None
        if bias is None:
            return optimized_attention(
                q, k, v, heads, attn_precision=attn_precision, transformer_options=extra_options
            )
        return optimized_attention(
            q,
            k,
            v,
            heads,
            mask=bias,
            attn_precision=attn_precision,
            transformer_options=extra_options,
        )

    def _build_bias(self, q: torch.Tensor, extra_options: dict[str, Any]) -> torch.Tensor | None:
        """Build the additive ``[branch*B, N, N]`` self-attention bias.

        Returns ``None`` (stay native) when the patch is disabled, the grid is
        larger than ``attn1_max_tokens``, the token count is not a clean grid, or
        the ownership masks are missing. Otherwise it projects the fixed
        ownership masks to this grid, derives the P1/P2/Unknown state and turns it
        into the cross-owner penalty (result cached per resolution).
        """
        masks = getattr(self.owner, "binding_masks", None)
        if not masks or len(masks) != 2 or not isinstance(masks[0], torch.Tensor):
            return None

        cfg = self.config
        lam = max(0.0, min(1.5, float(cfg.strength))) * _LAMBDA_SCALE
        if lam <= 1e-6:
            return None

        total_batch, n_tokens, _channels = q.shape
        if n_tokens < int(cfg.min_tokens):
            return None
        if n_tokens > int(cfg.max_tokens):
            self._log_once(
                ("skip", n_tokens),
                f"skip self-attention grid N={n_tokens} > attn1_max_tokens={cfg.max_tokens}",
            )
            return None

        # The patch receives cond and uncond stacked on the batch axis; the same
        # spatial bias applies to every CFG branch.
        condition_flags = extra_options.get("cond_or_uncond") or [0]
        branch_count = len(condition_flags) if len(condition_flags) > 0 else 1
        if total_batch % branch_count != 0:
            branch_count = 1
        batch = total_batch // branch_count

        grid_h, grid_w = factor_token_grid(n_tokens, extra_options.get("original_shape"))
        if grid_h * grid_w != n_tokens:
            return None

        # Check the dense-bias cache BEFORE mask interpolation/pooling.
        band = float(cfg.ambiguous_band)
        cache_key = (
            int(getattr(self.owner, "_binding_revision", 0)),
            str(q.device), str(q.dtype), int(batch), int(branch_count), int(n_tokens),
            int(grid_h), int(grid_w), round(float(lam), 6), round(float(band), 6),
        )
        getter = getattr(self.owner, "_runtime_cache_get", None)
        bias = getter("_self_bias_cache", cache_key) if callable(getter) else None

        if not isinstance(bias, torch.Tensor) or bias.device != q.device:
            mask_p1, mask_p2 = masks[0], masks[1]
            grid_p1 = project_ownership_mask(
                mask_p1,
                batch=batch,
                grid_h=grid_h,
                grid_w=grid_w,
                device=q.device,
            )
            grid_p2 = project_ownership_mask(
                mask_p2,
                batch=batch,
                grid_h=grid_h,
                grid_w=grid_w,
                device=q.device,
            )
            ownership = build_tristate_ownership(
                grid_p1,
                grid_p2,
                unknown_band=band,
            )
            bias = build_self_attention_bias(
                ownership,
                strength=lam,
                dtype=q.dtype,
                branch_count=branch_count,
            )

            putter = getattr(self.owner, "_runtime_cache_put", None)
            if callable(putter):
                bias = putter("_self_bias_cache", cache_key, bias)
            else:
                self.owner._self_bias_cache[cache_key] = bias

        self._log_once(
            ("apply", grid_h, grid_w),
            f"applying self-attention bias at resolution {grid_h}x{grid_w} "
            f"(N={n_tokens}, lambda={lam:.2f}, unknown_band={band:.2f}, cached=yes)",
        )
        return bias

    def _log_once(self, key: Any, message: str) -> None:
        """Emit a diagnostic line at most once per (key) for this generation."""
        if not attention_debug_enabled(getattr(self.owner, "saya_debug", False)):
            return
        if key in self._logged:
            return
        self._logged.add(key)
        print(f"[ComfyCouple {self.config.profile}] {message}")


class SelfAttentionBindingMixin:
    """Factory mixin used by the regional-attention node to build attn1 patches."""

    def create_self_attention_patch(
        self: Self, module: Any, config: SelfBindingConfig
    ) -> SelfAttentionBindingPatch:
        """Create the callable attn1 replacement installed on one model block."""
        return SelfAttentionBindingPatch(self, module, config)

    @staticmethod
    def iter_self_attention_blocks(
        diffusion_model: Any, specs: Iterable[tuple[str, int, int]]
    ) -> Iterator[tuple[tuple[str, int, int], Any]]:
        """Yield ``((section, block_id, transformer_index), attn1_module)`` per spec.

        Each spec is ``(section, block_id, depth)`` where ``section`` is
        ``"input" | "middle" | "output"``. Missing blocks are skipped silently so
        an unexpected UNet topology cannot crash installation.
        """
        for section, block_id, depth in specs:
            try:
                if section == "middle":
                    container = diffusion_model.middle_block[1].transformer_blocks
                elif section == "input":
                    container = diffusion_model.input_blocks[block_id][1].transformer_blocks
                else:
                    container = diffusion_model.output_blocks[block_id][1].transformer_blocks
            except (AttributeError, IndexError, TypeError):
                continue
            for index in range(min(int(depth), len(container))):
                module = getattr(container[index], "attn1", None)
                if module is not None:
                    yield (section, block_id, index), module
