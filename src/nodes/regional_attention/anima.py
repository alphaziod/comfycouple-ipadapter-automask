"""Regional-attention path for ANIMA-style transformer models.

The SDXL path patches ``attn2`` through ComfyUI's ``patches_replace`` table.
ANIMA models (DiT / video transformers that expose ``dm.blocks[*].cross_attn``)
do not use that table, so :class:`AnimaAttentionMixin` monkey-patches each
``cross_attn.forward`` instead.

The forward runs one attention call per region, multiplies each region output by
its ownership mask, sums them, and optionally blends the result with the
unmodified attention (same "listening strength" / blend logic as the legacy SDXL
blend). It never runs for SDXL checkpoints.
"""

from __future__ import annotations

import types
from typing import Any, Self

import torch

from .context import (
    apply_region_strength,
    attention_debug_enabled,
    resolve_attention_mode,
    resolve_region_strength,
)
from .masks import (
    build_query_token_masks,
    extract_detailer_crop_context,
    fill_unassigned_mask_regions,
    masks_match_attention_pass,
    prepare_detailer_region_masks,
    prepare_masks_for_attention_output,
)
from .operations import calculate_regional_blend

_LOG_PREFIX = "[ComfyCouple ANIMA]"


class AnimaAttentionMixin:
    """ANIMA regional-attention behaviour mixed into ``RegionalAttentionNode``."""

    def prepare_anima_context(
        self: Self,
        ctx: Any,
        batch_size: int,
        x: Any,
        reference_context: Any = None,
        label: str = "",
    ) -> Any:
        """Broadcast one region's context to ``batch_size`` and match device/dtype.

        Raises ``RuntimeError`` when the region context's channel width does not
        match the live context (i.e. SDXL conditionings were fed to an ANIMA
        model).
        """
        if not isinstance(ctx, torch.Tensor):
            return ctx
        if ctx.ndim != 3:
            return ctx.to(device=x.device)

        if (
            isinstance(reference_context, torch.Tensor)
            and reference_context.ndim == 3
            and ctx.shape[-1] != reference_context.shape[-1]
        ):
            raise RuntimeError(
                f"{_LOG_PREFIX} {label} conditioning dim mismatch: "
                f"region_dim={ctx.shape[-1]} current_context_dim={reference_context.shape[-1]}. "
                "ANIMA must receive ANIMA-encoded conditionings, not SDXL conditionings."
            )

        if ctx.shape[0] == batch_size:
            broadcast = ctx
        elif ctx.shape[0] == 1:
            broadcast = ctx.repeat(batch_size, 1, 1)
        else:
            reps = (batch_size + ctx.shape[0] - 1) // ctx.shape[0]
            broadcast = ctx.repeat(reps, 1, 1)[:batch_size]

        target_dtype = (
            reference_context.dtype if isinstance(reference_context, torch.Tensor) else x.dtype
        )
        return broadcast.to(device=x.device, dtype=target_dtype)

    def install_anima_patch(self: Self, new_model: Any, dm: Any) -> None:
        """Monkey-patch ``cross_attn.forward`` on every ANIMA block of ``dm``.

        Idempotent per block (``_saya_anima_real_hooked`` guard). Raises when no
        hookable block is found.
        """
        patched_count = 0
        for block_id, block in enumerate(getattr(dm, "blocks", [])):
            if not hasattr(block, "cross_attn"):
                continue
            attn = block.cross_attn
            if getattr(attn, "_saya_anima_real_hooked", False):
                continue

            original_forward = attn.forward
            printed_flag = {"done": False}

            def saya_anima_forward(
                self_attn: Any,
                x: Any,
                context: Any = None,
                rope_emb: Any = None,
                transformer_options: Any = None,
                _orig: Any = original_forward,
                _bid: Any = block_id,
                _printed: Any = printed_flag,
                **kwargs: Any,
            ) -> Any:
                """Bound replacement for one ANIMA ``cross_attn.forward``."""
                return self.run_anima_regional_forward(
                    _orig,
                    _bid,
                    _printed,
                    x,
                    context=context,
                    rope_emb=rope_emb,
                    transformer_options=transformer_options,
                    **kwargs,
                )

            attn.forward = types.MethodType(saya_anima_forward, attn)
            attn._saya_anima_real_hooked = True
            attn._saya_anima_hooked = True
            patched_count += 1

        if patched_count <= 0:
            raise RuntimeError(
                f"{_LOG_PREFIX} No dm.blocks[*].cross_attn.forward hooks were installed"
            )
        print(f"{_LOG_PREFIX} real cross_attn hooks installed: {patched_count}")

    def run_anima_regional_forward(
        self: Self,
        original_forward: Any,
        block_id: int,
        printed_flag: Any,
        x: Any,
        context: Any = None,
        rope_emb: Any = None,
        transformer_options: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Run one ANIMA attention block with regional ownership + fallback blend."""
        if transformer_options is None:
            transformer_options = {}

        def bypass() -> Any:
            """Fall back to the model's own attention for this block."""
            return original_forward(
                x,
                context=context,
                rope_emb=rope_emb,
                transformer_options=transformer_options,
                **kwargs,
            )

        if context is None or not isinstance(transformer_options, dict):
            return bypass()

        cond_or_uncond = transformer_options.get("cond_or_uncond", None)
        original_shape = transformer_options.get("original_shape", None)
        if cond_or_uncond is None or original_shape is None or x.ndim != 3:
            return bypass()

        chunks = len(cond_or_uncond)
        if chunks <= 0 or x.shape[0] % chunks != 0:
            return bypass()

        if not printed_flag["done"]:
            try:
                print(
                    f"{_LOG_PREFIX} block={block_id} real couple active x={tuple(x.shape)} "
                    f"context={tuple(context.shape)} cond_or_uncond={cond_or_uncond} "
                    f"original_shape={original_shape}"
                )
            except Exception as error:
                print(f"{_LOG_PREFIX} debug print failed block={block_id}: {error}")
            printed_flag["done"] = True

        len_neg, len_pos = self.conditioning_length
        crop_context = extract_detailer_crop_context(transformer_options)
        detailer_bank_available = bool(
            crop_context is not None
            and getattr(self, "detailer_positive_masks", None) is not None
            and getattr(self, "detailer_positive_conds", None) is not None
            and int(getattr(self, "detailer_positive_count", 0)) > 0
        )

        x_list = x.chunk(chunks, dim=0)
        context_list = None
        if (
            isinstance(context, torch.Tensor)
            and context.ndim == 3
            and context.shape[0] % chunks == 0
        ):
            context_list = context.chunk(chunks, dim=0)

        out_chunks = []
        for chunk_index, cond_flag in enumerate(cond_or_uncond):
            x_chunk = x_list[chunk_index]
            chunk_batch = x_chunk.shape[0]
            owner_index = None

            if cond_flag == 0 and detailer_bank_available:
                region_conds = self.detailer_positive_conds
                source_masks, owner_index = prepare_detailer_region_masks(
                    self.detailer_positive_masks, crop_context, "cond-detailer-anima"
                )
                region_count = int(self.detailer_positive_count)
                label = "cond-detailer"
            elif cond_flag == 0:
                region_conds = self.negative_positive_conds[1]
                source_masks = self.negative_positive_masks[1]
                region_count = len_pos
                label = "cond"
            else:
                region_conds = self.negative_positive_conds[0]
                source_masks = self.negative_positive_masks[0]
                region_count = len_neg
                label = "uncond"

            local_options = dict(transformer_options)
            local_options["cond_or_uncond"] = [cond_flag]

            region_outputs = []
            for region_index in range(region_count):
                region_context = self.prepare_anima_context(
                    region_conds[region_index],
                    chunk_batch,
                    x_chunk,
                    reference_context=context,
                    label=f"{label}[{region_index}]",
                )
                region_outputs.append(
                    original_forward(
                        x_chunk,
                        context=region_context,
                        rope_emb=rope_emb,
                        transformer_options=local_options,
                        **kwargs,
                    )
                )
            region_stack = torch.cat(region_outputs, dim=0)

            try:
                masks = build_query_token_masks(source_masks, x_chunk, original_shape)
                masks = prepare_masks_for_attention_output(masks, region_stack, original_shape, label)
                if (
                    isinstance(masks, torch.Tensor)
                    and masks.ndim == 3
                    and masks.shape[0] == region_count * chunk_batch
                ):
                    _groups, mask_tokens, mask_channels = masks.shape
                    masks_v = masks.contiguous().view(
                        region_count, chunk_batch, mask_tokens, mask_channels
                    )
                    v3_mode = (
                        "DETAILER"
                        if crop_context is not None
                        else resolve_attention_mode(
                            getattr(self, "saya_mode", "GEN"), original_shape, region_stack
                        )
                    )
                    masks_aligned, align_reason = masks_match_attention_pass(
                        source_masks, original_shape
                    )
                    owner_locked = owner_index is not None
                    local_safe = (v3_mode == "DETAILER" or not masks_aligned) and not owner_locked
                    if not local_safe and not owner_locked:
                        masks_v = fill_unassigned_mask_regions(masks_v, original_shape)
                    masks_v = masks_v / masks_v.sum(dim=0, keepdim=True).clamp_min(1e-06)
                    v3_strength = resolve_region_strength(
                        v3_mode, getattr(self, "saya_strength", -1.0)
                    )
                    if local_safe:
                        v3_strength = min(max(v3_strength, 0.9), 1.0)
                    masks_v = apply_region_strength(masks_v, v3_strength)
                    couple_blend = calculate_regional_blend(
                        v3_mode,
                        masks_aligned=masks_aligned,
                        custom_strength=getattr(self, "saya_strength", -1.0),
                    )
                    if attention_debug_enabled(getattr(self, "saya_debug", False)):
                        print(
                            f"{_LOG_PREFIX} block={block_id} mode={v3_mode} "
                            f"mask_strength={v3_strength:.2f} blend={couple_blend:.2f} "
                            f"aligned={masks_aligned} local_safe={local_safe} "
                            f"owner_locked={owner_index is not None} reason={align_reason} "
                            f"masks={tuple(masks_v.shape)} qkv={tuple(region_stack.shape)}"
                        )
                    masks = masks_v.contiguous().view(
                        region_count * chunk_batch, mask_tokens, mask_channels
                    )
                else:
                    if attention_debug_enabled(getattr(self, "saya_debug", False)):
                        print(
                            f"{_LOG_PREFIX} block={block_id} bad mask group shape "
                            f"masks={getattr(masks, 'shape', None)} qkv={tuple(region_stack.shape)}, "
                            "neutral fallback"
                        )
                    mask_tokens = region_stack.shape[1]
                    mask_channels = region_stack.shape[2]
                    masks = torch.ones(
                        (region_count * chunk_batch, mask_tokens, mask_channels),
                        device=region_stack.device,
                        dtype=region_stack.dtype,
                    ) / max(1, region_count)
                    v3_mode = (
                        "DETAILER"
                        if crop_context is not None
                        else resolve_attention_mode(
                            getattr(self, "saya_mode", "GEN"), original_shape, region_stack
                        )
                    )
                    couple_blend = 1.0
                    local_safe = True
            except Exception as error:
                print(
                    f"{_LOG_PREFIX} block={block_id} mask ownership failed: {error}, "
                    "using neutral fallback"
                )
                masks = torch.ones_like(region_stack) / max(1, region_count)
                v3_mode = (
                    "DETAILER"
                    if crop_context is not None
                    else resolve_attention_mode(
                        getattr(self, "saya_mode", "GEN"), original_shape, region_stack
                    )
                )
                couple_blend = 1.0
                local_safe = True

            qkv_regional = region_stack * masks
            qkv_regional = qkv_regional.view(
                region_count, chunk_batch, region_stack.shape[1], region_stack.shape[2]
            ).sum(dim=0)

            normal_out = None
            if (local_safe or couple_blend < 0.999) and context_list is not None:
                try:
                    normal_context = self.prepare_anima_context(
                        context_list[chunk_index],
                        chunk_batch,
                        x_chunk,
                        reference_context=context,
                        label=f"{label}[normal]",
                    )
                    normal_out = original_forward(
                        x_chunk,
                        context=normal_context,
                        rope_emb=rope_emb,
                        transformer_options=local_options,
                        **kwargs,
                    )
                except Exception as error:
                    if attention_debug_enabled(getattr(self, "saya_debug", False)):
                        print(
                            f"{_LOG_PREFIX} block={block_id} normal fallback unavailable: {error}"
                        )
                    normal_out = None

            if normal_out is not None:
                out_chunk = normal_out * (1.0 - couple_blend) + qkv_regional * couple_blend
            else:
                out_chunk = qkv_regional
            out_chunks.append(out_chunk)

        return torch.cat(out_chunks, dim=0)
