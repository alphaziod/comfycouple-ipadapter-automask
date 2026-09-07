"""Cross-attention (``attn2``) replacement for the Saya Couple regional binding.

One :class:`CrossAttentionPatch` instance is installed per SDXL cross-attention
block. On every forward it dispatches to one of two paths:

* **Joint pre-softmax path** (``_call_v0_pre_softmax``) - used for the
  ``V0_PRE_SOFTMAX`` / ``V1_ATTN1`` / ``V1_1_HIRES`` / ``V2_QUERY_OWNERSHIP``
  binding modes, outside detailer crops. The conditional branch attends to a
  single joint ``[Base ; P1 ; P2]`` context with an additive, finite,
  pre-softmax bias derived from the tri-state ``P1/P2/Unknown`` ownership. The
  unconditional branch runs ComfyUI's native attention unchanged.

* **Legacy blend path** (``_call_legacy``) - used for ``LEGACY`` mode and for all
  detailer crops. Runs one attention per region, multiplies each by a spatial
  mask, sums, and blends with the native output.

The ``_diagnose_*`` / ``_debug_v2_*`` methods at the bottom are the semantic
"query-derived ownership" experiment. **They are not on the live v21.7E path**
(``_process_v0_conditional`` sets ``v2_dynamic = False``); they are kept only for
research and offline diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Self

import torch
import torch.nn.functional as F
from comfy.ldm.modules.attention import optimized_attention

from ..couple_config import PRE_SOFTMAX_BINDING_MODES
from .context import (
    apply_region_strength,
    attention_debug_enabled,
    concatenate_context_tensors,
    resolve_attention_mode,
    resolve_region_strength,
)
from .masks import (
    build_query_token_masks,
    crop_masks_to_detailer_region,
    extract_detailer_crop_context,
    factor_token_grid,
    fill_unassigned_mask_regions,
    masks_match_attention_pass,
    prepare_detailer_region_masks,
    prepare_masks_for_attention_output,
)
from .operations import calculate_regional_blend, try_standard_attention
from .ownership import (
    build_cross_attention_bias,
    build_tristate_ownership,
    project_ownership_mask,
)

_LOG_PREFIX = "[ComfyCouple SayaPatch v3.3]"


@dataclass(frozen=True, slots=True)
class BranchProjection:
    """Per-branch projected conditioning + masks for the legacy blend path."""

    masks: Any
    source_masks: Any
    keys: Any
    values: Any
    region_count: int
    owner_index: int | None = None


class CrossAttentionPatch:
    """Callable ``attn2`` replacement bound to one node (``owner``) and one block."""

    def __init__(self, owner: Any, module: Any, key: Any = None) -> None:
        """Store the node state, the target attention module and its block key."""
        self.owner = owner
        self.module = module
        self.key = key

    def __call__(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, extra_options: dict[str, Any]
    ) -> torch.Tensor:
        """Dispatch to the joint pre-softmax path or the legacy blend path."""
        if self._v0_pre_softmax_active(extra_options):
            return self._call_v0_pre_softmax(q, k, v, extra_options)
        return self._call_legacy(q, k, v, extra_options)

    def _v0_pre_softmax_active(self, extra_options: dict[str, Any]) -> bool:
        """True when the joint pre-softmax path applies to this pass.

        Requires a pre-softmax binding mode, a captured joint context, ownership
        masks, and *no* detailer crop (crops keep the proven legacy path because
        their prompts are already character-local).
        """
        return bool(
            str(getattr(self.owner, "binding_mode", "LEGACY")).upper()
            in PRE_SOFTMAX_BINDING_MODES
            and getattr(self.owner, "binding_contexts", None) is not None
            and getattr(self.owner, "binding_masks", None) is not None
            and extract_detailer_crop_context(extra_options) is None
        )

    # ------------------------------------------------------------------ #
    # Joint pre-softmax path (V0 / V1 / V2 modes)                        #
    # ------------------------------------------------------------------ #

    def _call_v0_pre_softmax(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, extra_options: dict[str, Any]
    ) -> torch.Tensor:
        """Process every CFG branch: gated joint attention for cond, native for uncond."""
        condition_flags = extra_options["cond_or_uncond"]
        branch_count = len(condition_flags)
        query_chunks = q.chunk(branch_count, dim=0)

        outputs: list[torch.Tensor] = []
        for branch_index, condition_flag in enumerate(condition_flags):
            query = query_chunks[branch_index]
            if condition_flag != 0:
                # The negative conditioning has no private P1/P2 token groups.
                # Keep ComfyUI's native unconditional attention exactly.
                output = try_standard_attention(
                    query, k, v, branch_count, branch_index, extra_options["n_heads"]
                )
                if output is None:
                    raise RuntimeError(
                        "V0_PRE_SOFTMAX could not evaluate the unconditional branch"
                    )
                outputs.append(output)
                continue
            outputs.append(self._process_v0_conditional(query, extra_options))
        return torch.cat(outputs, dim=0)

    def _process_v0_conditional(
        self, query: torch.Tensor, extra_options: dict[str, Any]
    ) -> torch.Tensor:
        """One conditional-branch attention over the joint ``[Base;P1;P2]`` context.

        The additive pre-softmax bias comes from the *fixed geometric* tri-state
        ownership. v21.7E deliberately keeps semantic (query-derived) ownership
        out of the decision path; the ``V2_QUERY_OWNERSHIP`` mode name is kept
        only for workflow compatibility.
        """
        contexts = getattr(self.owner, "binding_contexts", None) or {}
        base = contexts.get("base")
        person_1 = contexts.get("person_1")
        person_2 = contexts.get("person_2")
        if person_1 is None or person_2 is None:
            raise RuntimeError("V0_PRE_SOFTMAX is missing Person 1/Person 2 contexts")

        context_parts = [part for part in (base, person_1, person_2) if part is not None]
        joint_context = torch.cat(context_parts, dim=1)
        keys = self.module.to_k(joint_context)
        values = self.module.to_v(joint_context)
        keys = self._match_batch(keys, int(query.shape[0]))
        values = self._match_batch(values, int(query.shape[0]))
        if keys.dtype != query.dtype:
            keys = keys.to(query.dtype)
        if values.dtype != query.dtype:
            values = values.to(query.dtype)

        base_tokens = int(base.shape[1]) if base is not None else 0
        p1_tokens = int(person_1.shape[1])
        p2_tokens = int(person_2.shape[1])

        gate_p1 = self._project_ownership_gate(
            self.owner.binding_masks[0], query, extra_options.get("original_shape")
        )
        gate_p2 = self._project_ownership_gate(
            self.owner.binding_masks[1], query, extra_options.get("original_shape")
        )

        binding_mode = str(getattr(self.owner, "binding_mode", "LEGACY")).upper()
        v2_dynamic = False  # semantic ownership is diagnostic-only in v21.7E
        attention_bias = self._get_fixed_attention_bias(
            gate_p1,
            gate_p2,
            base_tokens=base_tokens,
            p1_tokens=p1_tokens,
            p2_tokens=p2_tokens,
            dtype=query.dtype,
            device=query.device,
        )

        self._debug_binding_once(
            binding_mode=binding_mode,
            query=query,
            joint_context=joint_context,
            base_tokens=base_tokens,
            p1_tokens=p1_tokens,
            p2_tokens=p2_tokens,
            dynamic=v2_dynamic,
        )

        return optimized_attention(
            query,
            keys,
            values,
            extra_options["n_heads"],
            mask=attention_bias,
            attn_precision=getattr(self.module, "attn_precision", None),
            transformer_options=extra_options,
        )

    def _get_fixed_attention_bias(
        self,
        gate_p1: torch.Tensor,
        gate_p2: torch.Tensor,
        *,
        base_tokens: int,
        p1_tokens: int,
        p2_tokens: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Tri-state ``[B, N, T]`` cross-attention bias, cached per resolution.

        The gates are geometric and constant across diffusion steps, so the
        resulting bias is memoised on the owner keyed by binding revision,
        device/dtype, shape, gate identity and the unknown band.
        """
        config = getattr(self.owner, "self_binding_config", None)
        unknown_band = float(getattr(config, "ambiguous_band", 0.15))
        cache_key = (
            int(getattr(self.owner, "_binding_revision", 0)),
            str(device),
            str(dtype),
            int(gate_p1.shape[0]),
            int(gate_p1.shape[1]),
            id(gate_p1),
            id(gate_p2),
            round(unknown_band, 6),
            int(base_tokens),
            int(p1_tokens),
            int(p2_tokens),
        )
        getter = getattr(self.owner, "_runtime_cache_get", None)
        cached = getter("_cross_bias_cache", cache_key) if callable(getter) else None
        if isinstance(cached, torch.Tensor) and cached.device == device:
            return cached

        ownership = build_tristate_ownership(gate_p1, gate_p2, unknown_band=unknown_band)
        bias = build_cross_attention_bias(
            ownership,
            base_tokens=base_tokens,
            p1_tokens=p1_tokens,
            p2_tokens=p2_tokens,
            dtype=dtype,
        )

        putter = getattr(self.owner, "_runtime_cache_put", None)
        if callable(putter):
            return putter("_cross_bias_cache", cache_key, bias)
        self.owner._cross_bias_cache[cache_key] = bias
        return bias

    @staticmethod
    def _match_batch(tensor: torch.Tensor, target_batch: int) -> torch.Tensor:
        """Broadcast/repeat a context projection along dim 0 to ``target_batch``."""
        if int(tensor.shape[0]) == target_batch:
            return tensor
        if int(tensor.shape[0]) == 1:
            return tensor.expand(target_batch, -1, -1)
        repeats = (target_batch + int(tensor.shape[0]) - 1) // int(tensor.shape[0])
        return tensor.repeat(repeats, 1, 1)[:target_batch]

    def _project_ownership_gate(
        self, mask: torch.Tensor, query: torch.Tensor, original_shape: Any
    ) -> torch.Tensor:
        """Bilinearly resize one fixed ownership mask to the query grid; cached.

        Returns a ``[B, N]`` gate in ``[0, 1]``. Raises when the mask is not a
        tensor or the resulting token count does not match the query.
        """
        if not isinstance(mask, torch.Tensor):
            raise TypeError("V0_PRE_SOFTMAX ownership mask must be a tensor")
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.ndim != 3:
            raise ValueError(f"unexpected V0 ownership mask shape: {tuple(mask.shape)}")

        target_batch = int(query.shape[0])
        target_h, target_w = factor_token_grid(int(query.shape[1]), original_shape)
        cache_key = (
            int(getattr(self.owner, "_binding_revision", 0)),
            id(mask),
            str(query.device),
            int(target_batch),
            int(query.shape[1]),
            int(target_h),
            int(target_w),
        )
        getter = getattr(self.owner, "_runtime_cache_get", None)
        cached = getter("_gate_cache", cache_key) if callable(getter) else None
        if isinstance(cached, torch.Tensor) and cached.device == query.device:
            return cached

        gate = project_ownership_mask(
            mask, batch=target_batch, grid_h=target_h, grid_w=target_w, device=query.device
        )
        if int(gate.shape[1]) != int(query.shape[1]):
            raise ValueError(
                f"V0 ownership grid mismatch: {tuple(gate.shape)} vs query tokens={query.shape[1]}"
            )
        gate = gate.to(device=query.device, dtype=torch.float32).clamp(0.0, 1.0)

        putter = getattr(self.owner, "_runtime_cache_put", None)
        if callable(putter):
            return putter("_gate_cache", cache_key, gate)
        return gate

    @staticmethod
    def _build_v0_attention_bias(
        gate_p1: torch.Tensor,
        gate_p2: torch.Tensor,
        *,
        base_tokens: int,
        p1_tokens: int,
        p2_tokens: int,
        dtype: torch.dtype,
        unknown_band: float = 0.15,
    ) -> torch.Tensor:
        """Uncached tri-state cross-attention bias from two ownership gates."""
        ownership = build_tristate_ownership(gate_p1, gate_p2, unknown_band=unknown_band)
        return build_cross_attention_bias(
            ownership,
            base_tokens=base_tokens,
            p1_tokens=p1_tokens,
            p2_tokens=p2_tokens,
            dtype=dtype,
        )

    def _debug_binding_once(
        self,
        *,
        binding_mode: str,
        query: torch.Tensor,
        joint_context: torch.Tensor,
        base_tokens: int,
        p1_tokens: int,
        p2_tokens: int,
        dynamic: bool,
    ) -> None:
        """Print the joint-context shape once per (token count, dynamic) pair."""
        if not attention_debug_enabled(getattr(self.owner, "saya_debug", False)):
            return
        seen = getattr(self.owner, "_v2_debug_seen", None)
        if seen is None:
            seen = set()
            self.owner._v2_debug_seen = seen
        key = ("binding", int(query.shape[1]), bool(dynamic))
        if key in seen:
            return
        seen.add(key)
        tag = "V2-tristate" if binding_mode == "V2_QUERY_OWNERSHIP" else "V0-tristate"
        print(
            f"[ComfyCouple Binding {tag}] q={tuple(query.shape)} context={tuple(joint_context.shape)} "
            f"tokens(base,p1,p2)=({base_tokens},{p1_tokens},{p2_tokens}) semantic_dynamic={dynamic}"
        )

    # ------------------------------------------------------------------ #
    # Legacy blend path (LEGACY mode + all detailer crops)              #
    # ------------------------------------------------------------------ #

    def _call_legacy(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, extra_options: dict[str, Any]
    ) -> torch.Tensor:
        """Per-region attention, spatially masked and summed, blended with native."""
        condition_flags = extra_options["cond_or_uncond"]
        query_chunks = q.chunk(len(condition_flags), dim=0)
        batch_size = int(query_chunks[0].shape[0])
        conditional, unconditional = self._build_branch_projections(query_chunks[0], extra_options)

        outputs: list[torch.Tensor] = []
        for branch_index, condition_flag in enumerate(condition_flags):
            projection = conditional if condition_flag == 0 else unconditional
            outputs.append(
                self._process_branch(
                    query=query_chunks[branch_index],
                    original_keys=k,
                    original_values=v,
                    branch_index=branch_index,
                    branch_count=len(condition_flags),
                    batch_size=batch_size,
                    projection=projection,
                    extra_options=extra_options,
                )
            )
        return torch.cat(outputs, dim=0)

    def _build_branch_projections(
        self, reference_query: Any, extra_options: dict[str, Any]
    ) -> tuple[BranchProjection, BranchProjection]:
        """Build (conditional, unconditional) projections for the legacy path.

        Honours detailer crops: masks are cropped to the region and, when a
        detected segment clearly belongs to one character, that crop is
        hard-locked to it (``owner_index``).
        """
        crop_context = extract_detailer_crop_context(extra_options)
        unconditional_masks = crop_masks_to_detailer_region(
            self.owner.negative_positive_masks[0], crop_context, "uncond"
        )
        detailer_bank_available = bool(
            crop_context is not None
            and getattr(self.owner, "detailer_positive_masks", None) is not None
            and getattr(self.owner, "detailer_positive_conds", None) is not None
            and int(getattr(self.owner, "detailer_positive_count", 0)) > 0
        )
        conditional_source_masks = (
            self.owner.detailer_positive_masks
            if detailer_bank_available
            else self.owner.negative_positive_masks[1]
        )
        conditional_source_conds = (
            self.owner.detailer_positive_conds
            if detailer_bank_available
            else self.owner.negative_positive_conds[1]
        )
        if detailer_bank_available:
            conditional_masks, owner_index = prepare_detailer_region_masks(
                conditional_source_masks, crop_context, "cond-detailer"
            )
        else:
            conditional_masks = crop_masks_to_detailer_region(
                conditional_source_masks, crop_context, "cond"
            )
            owner_index = None

        original_shape = extra_options["original_shape"]
        unconditional_query_masks = build_query_token_masks(
            unconditional_masks, reference_query, original_shape
        )
        conditional_query_masks = build_query_token_masks(
            conditional_masks, reference_query, original_shape
        )

        unconditional_context = concatenate_context_tensors(
            self.owner.negative_positive_conds[0], dim=0
        )
        conditional_context = concatenate_context_tensors(conditional_source_conds, dim=0)
        negative_count, main_positive_count = self.owner.conditioning_length
        positive_count = (
            int(self.owner.detailer_positive_count)
            if detailer_bank_available
            else main_positive_count
        )
        return (
            BranchProjection(
                masks=conditional_query_masks,
                source_masks=conditional_masks,
                keys=self.module.to_k(conditional_context),
                values=self.module.to_v(conditional_context),
                region_count=positive_count,
                owner_index=owner_index,
            ),
            BranchProjection(
                masks=unconditional_query_masks,
                source_masks=unconditional_masks,
                keys=self.module.to_k(unconditional_context),
                values=self.module.to_v(unconditional_context),
                region_count=negative_count,
            ),
        )

    def _process_branch(
        self,
        *,
        query: Any,
        original_keys: Any,
        original_values: Any,
        branch_index: int,
        branch_count: int,
        batch_size: int,
        projection: BranchProjection,
        extra_options: dict[str, Any],
    ) -> Any:
        """Run one legacy branch: masked per-region attention summed then blended."""
        repeated_query = query.repeat(projection.region_count, 1, 1)
        keys = self._repeat_projection(projection.keys, projection.region_count, batch_size)
        values = self._repeat_projection(projection.values, projection.region_count, batch_size)
        if keys.dtype != repeated_query.dtype or values.dtype != repeated_query.dtype:
            keys = keys.to(repeated_query.dtype)
            values = values.to(repeated_query.dtype)

        standard_output = try_standard_attention(
            query,
            original_keys,
            original_values,
            branch_count,
            branch_index,
            extra_options["n_heads"],
        )
        regional_output = optimized_attention(
            repeated_query, keys, values, extra_options["n_heads"]
        )
        masks = prepare_masks_for_attention_output(
            projection.masks,
            regional_output,
            extra_options.get("original_shape"),
            "cond" if branch_index == 0 else "uncond",
        )
        normalized_masks, blend, local_fallback = self._normalize_masks(
            masks=masks,
            source_masks=projection.source_masks,
            regional_output=regional_output,
            region_count=projection.region_count,
            batch_size=batch_size,
            original_shape=extra_options.get("original_shape"),
            detailer_crop_active=extract_detailer_crop_context(extra_options) is not None,
            detailer_owner_locked=projection.owner_index is not None,
        )
        regional_output = regional_output * normalized_masks
        regional_output = regional_output.view(
            projection.region_count,
            batch_size,
            -1,
            self.module.heads * self.module.dim_head,
        ).sum(dim=0)

        if standard_output is not None and (local_fallback or blend < 0.999):
            return standard_output * (1.0 - blend) + regional_output * blend
        return regional_output

    def _normalize_masks(
        self,
        *,
        masks: Any,
        source_masks: Any,
        regional_output: Any,
        region_count: int,
        batch_size: int,
        original_shape: Any,
        detailer_crop_active: bool = False,
        detailer_owner_locked: bool = False,
    ) -> tuple[Any, float, bool]:
        """Normalize per-region masks and return ``(masks, blend, local_fallback)``.

        On any error returns a uniform mask, ``blend=1.0`` and
        ``local_fallback=True`` so the caller degrades gracefully.
        """
        mode = (
            "DETAILER"
            if detailer_crop_active
            else resolve_attention_mode(
                getattr(self.owner, "saya_mode", "GEN"), original_shape, regional_output
            )
        )
        try:
            expected_batch = region_count * batch_size
            if not isinstance(masks, torch.Tensor) or masks.ndim != 3:
                raise ValueError(f"unexpected mask rank: {getattr(masks, 'shape', None)}")
            if masks.shape[0] != expected_batch:
                raise ValueError(f"unexpected mask batch: {masks.shape[0]} != {expected_batch}")

            _, token_count, channels = masks.shape
            grouped_masks = masks.contiguous().view(
                region_count, batch_size, token_count, channels
            )
            aligned, reason = masks_match_attention_pass(source_masks, original_shape)
            local_fallback = (mode == "DETAILER" or not aligned) and not detailer_owner_locked
            if not local_fallback and not detailer_owner_locked:
                grouped_masks = fill_unassigned_mask_regions(grouped_masks, original_shape)
            grouped_masks = grouped_masks / grouped_masks.sum(dim=0, keepdim=True).clamp_min(1e-06)

            strength = resolve_region_strength(
                mode, getattr(self.owner, "saya_strength", -1.0)
            )
            if local_fallback:
                strength = min(max(strength, 0.9), 1.0)
            grouped_masks = apply_region_strength(grouped_masks, strength)
            blend = calculate_regional_blend(
                mode,
                masks_aligned=aligned,
                custom_strength=getattr(self.owner, "saya_strength", -1.0),
            )
            self._log_mask_state(
                mode=mode,
                strength=strength,
                blend=blend,
                aligned=aligned,
                local_fallback=local_fallback,
                reason=reason,
                owner_locked=detailer_owner_locked,
                masks=grouped_masks,
                output=regional_output,
            )
            return (
                grouped_masks.contiguous().view(expected_batch, token_count, channels),
                blend,
                local_fallback,
            )
        except Exception as error:
            print(f"{_LOG_PREFIX} mask ownership failed: {error}; using neutral fallback")
            neutral_masks = torch.ones_like(regional_output) / max(1, region_count)
            return neutral_masks, 1.0, True

    def _log_mask_state(self, **state: Any) -> None:
        """Print legacy-path mask state when regional-attention diagnostics are on."""
        if not attention_debug_enabled(getattr(self.owner, "saya_debug", False)):
            return
        print(
            f"{_LOG_PREFIX} "
            f"mode={state['mode']} mask_strength={state['strength']:.2f} "
            f"blend={state['blend']:.2f} aligned={state['aligned']} "
            f"local_safe={state['local_fallback']} owner_locked={state['owner_locked']} "
            f"reason={state['reason']} "
            f"masks={tuple(state['masks'].shape)} qkv={tuple(state['output'].shape)}"
        )

    @staticmethod
    def _repeat_projection(projection: Any, region_count: int, batch_size: int) -> Any:
        """Stack ``region_count`` copies of each region projection, batch-repeated."""
        return torch.cat(
            [
                projection[index].unsqueeze(0).repeat(batch_size, 1, 1)
                for index in range(region_count)
            ],
            dim=0,
        )

    # ------------------------------------------------------------------ #
    # DIAGNOSTIC / RESEARCH ONLY - NOT ON THE LIVE v21.7E PATH           #
    #                                                                    #
    # The methods below implement the abandoned "query-derived ownership" #
    # experiment. `_process_v0_conditional` never calls them             #
    # (`v2_dynamic = False`). Kept for offline analysis only.            #
    # ------------------------------------------------------------------ #

    def _diagnostic_v2_dynamic_enabled(self, query: torch.Tensor) -> bool:
        """(Diagnostic) Whether the semantic estimate would run for this block.

        Only the six SDXL decoder high-res blocks (``output`` 3/4/5, ``N > 1024``)
        qualified: low-res block-local flips visibly mixed faces.
        """
        if int(query.shape[1]) <= 1024:
            return False
        key = self.key
        return bool(
            isinstance(key, tuple)
            and len(key) >= 2
            and key[0] == "output"
            and int(key[1]) in (3, 4, 5)
        )

    def _diagnose_v2_query_ownership(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        prior_p1: torch.Tensor,
        prior_p2: torch.Tensor,
        *,
        base_tokens: int,
        p1_tokens: int,
        p2_tokens: int,
        extra_options: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(Diagnostic) Query/text ownership estimate; never authoritative.

        Builds each character anchor from its most discriminative private-token
        keys (furthest from the opposite bank's centroid), projects the query
        onto ``anchor_p1 - anchor_p2``, z-normalizes per block, polarity-locks to
        the geometric prior and only lets a boundary crossing happen at high
        confidence.
        """
        batch, n_tokens, channels = query.shape
        if n_tokens < 64 or p1_tokens <= 0 or p2_tokens <= 0:
            return prior_p1, prior_p2
        heads = int(extra_options.get("n_heads", 1) or 1)
        if heads <= 0 or channels % heads != 0:
            return prior_p1, prior_p2
        head_dim = channels // heads
        p1_start = int(base_tokens)
        p1_end = p1_start + int(p1_tokens)
        p2_end = p1_end + int(p2_tokens)
        if p2_end > int(keys.shape[1]):
            return prior_p1, prior_p2

        query_heads = query.reshape(batch, n_tokens, heads, head_dim)
        p1_keys = keys[:, p1_start:p1_end].reshape(batch, p1_tokens, heads, head_dim)
        p2_keys = keys[:, p1_end:p2_end].reshape(batch, p2_tokens, heads, head_dim)

        p1_keys_f = p1_keys.float()
        p2_keys_f = p2_keys.float()
        p1_centroid = p1_keys_f.mean(dim=1, keepdim=True)
        p2_centroid = p2_keys_f.mean(dim=1, keepdim=True)
        p1_importance = (p1_keys_f - p2_centroid).square().mean(dim=(-1, -2))
        p2_importance = (p2_keys_f - p1_centroid).square().mean(dim=(-1, -2))
        top1 = min(24, p1_tokens)
        top2 = min(24, p2_tokens)
        p1_top_idx = p1_importance.topk(top1, dim=1, largest=True, sorted=False).indices
        p2_top_idx = p2_importance.topk(top2, dim=1, largest=True, sorted=False).indices
        p1_gather = p1_top_idx[:, :, None, None].expand(-1, -1, heads, head_dim)
        p2_gather = p2_top_idx[:, :, None, None].expand(-1, -1, heads, head_dim)
        anchor_p1 = torch.gather(p1_keys_f, 1, p1_gather).mean(dim=1)
        anchor_p2 = torch.gather(p2_keys_f, 1, p2_gather).mean(dim=1)
        delta = F.normalize(anchor_p1 - anchor_p2, dim=-1, eps=1e-6).to(dtype=query.dtype)

        query_norm = F.normalize(query_heads, dim=-1, eps=1e-6)
        score = (query_norm * delta.unsqueeze(1)).sum(dim=-1).float().mean(dim=-1)
        mean = score.mean(dim=1, keepdim=True)
        std = score.std(dim=1, keepdim=True, unbiased=False)
        valid = std > 1e-4
        # No bool(valid.any()): that forces a GPU->CPU sync. Invalid batches are
        # neutralised below by alpha=0 and therefore return the fixed prior.
        z = ((score - mean) / std.clamp_min(1e-4)).clamp(-4.0, 4.0)

        grid_h, grid_w = factor_token_grid(n_tokens, extra_options.get("original_shape"))
        if grid_h * grid_w == n_tokens and min(grid_h, grid_w) >= 5:
            z = F.avg_pool2d(
                z.reshape(batch, 1, grid_h, grid_w),
                kernel_size=5,
                stride=1,
                padding=2,
            ).reshape(batch, n_tokens)

        dynamic_p1 = torch.sigmoid(z * 1.10)

        # Lock the block-local discriminant's global polarity to the geometric prior
        # so P1/P2 can never be globally swapped by one block.
        prior_sign = prior_p1.float() * 2.0 - 1.0
        dynamic_sign = dynamic_p1 * 2.0 - 1.0
        alignment = (prior_sign * dynamic_sign).mean(dim=1, keepdim=True)
        dynamic_p1 = torch.where(alignment < 0.0, 1.0 - dynamic_p1, dynamic_p1)

        raw = getattr(self.owner, "_self_binding_raw", {})
        try:
            requested_strength = max(0.0, min(1.0, float(raw.get("v2_ownership_strength", 0.80))))
        except (TypeError, ValueError, AttributeError):
            requested_strength = 0.80
        try:
            requested_floor = max(0.0, min(0.45, float(raw.get("v2_confidence_floor", 0.10))))
        except (TypeError, ValueError, AttributeError):
            requested_floor = 0.10

        # Safety caps: a moving map may only cross a fixed territory on strong evidence.
        ownership_strength = min(requested_strength, 0.60)
        confidence_floor = max(requested_floor, 0.25)
        confidence = (dynamic_p1 - 0.5).abs() * 2.0
        confidence = (
            (confidence - confidence_floor) / max(1e-6, 1.0 - confidence_floor)
        ).clamp(0.0, 1.0)
        alpha = confidence * ownership_strength
        alpha = torch.where(valid.expand_as(alpha), alpha, torch.zeros_like(alpha))

        gate_p1 = prior_p1.float() * (1.0 - alpha) + dynamic_p1 * alpha
        gate_p1 = gate_p1.clamp(0.0, 1.0)
        gate_p2 = (1.0 - gate_p1).clamp(0.0, 1.0)

        self._debug_v2_once(
            grid_h=grid_h,
            grid_w=grid_w,
            n_tokens=n_tokens,
            dynamic_p1=dynamic_p1,
            confidence=confidence,
            prior_p1=prior_p1,
            gate_p1=gate_p1,
        )
        return gate_p1, gate_p2

    def _debug_v2_once(
        self,
        *,
        grid_h: int,
        grid_w: int,
        n_tokens: int,
        dynamic_p1: torch.Tensor,
        confidence: torch.Tensor,
        prior_p1: torch.Tensor,
        gate_p1: torch.Tensor,
    ) -> None:
        """(Diagnostic) Print the semantic-ownership stats once per grid."""
        if not attention_debug_enabled(getattr(self.owner, "saya_debug", False)):
            return
        seen = getattr(self.owner, "_v2_debug_seen", None)
        if seen is None:
            seen = set()
            self.owner._v2_debug_seen = seen
        key = ("ownership", grid_h, grid_w)
        if key in seen:
            return
        seen.add(key)
        override = ((prior_p1 >= 0.5) != (gate_p1 >= 0.5)).float().mean().item()
        confident = (confidence > 0.5).float().mean().item()
        print(
            "[ComfyCouple V2-stable] query ownership "
            f"grid={grid_h}x{grid_w} N={n_tokens} dynamic_mean={dynamic_p1.mean().item():.3f} "
            f"confident={confident:.3f} boundary_override={override:.3f} block={self.key}"
        )


class CrossAttentionPatchMixin:
    """Factory mixin used by the regional-attention node to build attn2 patches."""

    def create_cross_attention_patch(
        self: Self, module: Any, key: Any = None
    ) -> CrossAttentionPatch:
        """Create the callable cross-attention replacement for one model block."""
        return CrossAttentionPatch(self, module, key)
