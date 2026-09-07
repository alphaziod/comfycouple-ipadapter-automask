"""Spatial-mask plumbing for the regional-attention patches.

These helpers turn full-image ownership masks into the per-layer tensors the
attention patches need:

* :func:`factor_token_grid` - guess the ``(h, w)`` grid behind a flattened token
  count (used by every path, including the v21.7E ownership path).
* detailer-crop helpers - crop full-image masks to a detected region and, when a
  detected segment is clearly one character, hard-lock ownership for that crop.
* legacy-blend helpers - :func:`resize_masks_to_token_grid`,
  :func:`fill_unassigned_mask_regions`, :func:`prepare_masks_for_attention_output`
  and :func:`build_query_token_masks` feed the *legacy* post-softmax regional
  blend (``CrossAttentionPatch._call_legacy`` and the ANIMA path). The V0/V1/V2
  pre-softmax paths do not use them.

Every function is defensive: on unexpected input it logs (when debugging) and
returns a safe neutral value instead of raising, because most run inside the
sampler hot loop.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F

from .context import attention_debug_enabled

_DEBUG_ENV_VAR = "SAYA_COUPLE_DEBUG"
_LOG_V2 = "[ComfyCouple SayaPatch v2]"
_LOG_V33 = "[ComfyCouple SayaPatch v3.3]"
_LOG_V34 = "[ComfyCouple SayaPatch v3.4]"
_LOG_V35 = "[ComfyCouple SayaPatch v3.5]"


def first_valid_mask_shape(source_masks: Any) -> tuple[int, int] | None:
    """Return ``(height, width)`` of the first rank>=2 tensor in ``source_masks``."""
    try:
        for mask in source_masks:
            if isinstance(mask, torch.Tensor) and mask.ndim >= 2:
                return (int(mask.shape[-2]), int(mask.shape[-1]))
    except Exception:
        pass
    return None


def extract_detailer_crop_context(options: Any) -> dict[str, Any] | None:
    """Read normalized detailer-crop metadata from ComfyUI execution options.

    Looks for ``saya_couple_crop`` directly, then inside the common nested option
    dicts. Returns ``None`` when no usable crop region is present.
    """
    try:
        if not isinstance(options, dict):
            return None

        ctx = options.get("saya_couple_crop")
        if not isinstance(ctx, dict):
            for key in ("transformer_options", "model_options", "extra_options"):
                nested = options.get(key)
                if isinstance(nested, dict):
                    candidate = nested.get("saya_couple_crop")
                    if isinstance(candidate, dict):
                        ctx = candidate
                        break
        if not isinstance(ctx, dict):
            return None

        region = ctx.get("crop_region", ctx.get("bbox", ctx.get("crop")))
        if isinstance(region, dict):
            region = [region.get("x1"), region.get("y1"), region.get("x2"), region.get("y2")]
        if not isinstance(region, (list, tuple)) or len(region) != 4:
            return None
        x1, y1, x2, y2 = (float(v) for v in region)
        if x2 <= x1 or y2 <= y1:
            return None

        full_w = ctx.get("full_width")
        full_h = ctx.get("full_height")
        full_size = ctx.get("full_size", ctx.get("image_size"))
        if (
            (full_w is None or full_h is None)
            and isinstance(full_size, (list, tuple))
            and len(full_size) >= 2
        ):
            full_w, full_h = full_size[0], full_size[1]

        return {
            "crop_region": (x1, y1, x2, y2),
            "full_width": int(full_w) if full_w is not None else None,
            "full_height": int(full_h) if full_h is not None else None,
            "label": str(ctx.get("label", "detailer")),
            "segment_mask": ctx.get("segment_mask"),
        }
    except Exception as error:
        if attention_debug_enabled():
            print(f"{_LOG_V34} invalid crop context: {error}")
        return None


def crop_masks_to_detailer_region(
    source_masks: Any, crop_ctx: Any, label: str = ""
) -> Any:
    """Crop each full-image ownership mask to the active detailer region.

    Coordinates in ``crop_ctx["crop_region"]`` are in ``full_width/full_height``
    space and rescaled to every mask's own resolution. Masks that would become
    empty, or that are not tensors, are passed through untouched. Returns
    ``source_masks`` unchanged when there is no crop context.
    """
    if crop_ctx is None:
        return source_masks
    try:
        x1, y1, x2, y2 = crop_ctx["crop_region"]
        result: list[Any] = []
        debug_rows: list[str] = []
        for mask in source_masks:
            if not isinstance(mask, torch.Tensor) or mask.ndim < 2:
                result.append(mask)
                continue

            mask_h = int(mask.shape[-2])
            mask_w = int(mask.shape[-1])
            full_w = max(1, int(crop_ctx.get("full_width") or mask_w))
            full_h = max(1, int(crop_ctx.get("full_height") or mask_h))
            scale_x = mask_w / full_w
            scale_y = mask_h / full_h
            crop_x1 = max(0, min(mask_w - 1, int(x1 * scale_x)))
            crop_y1 = max(0, min(mask_h - 1, int(y1 * scale_y)))
            crop_x2 = max(crop_x1 + 1, min(mask_w, int(x2 * scale_x + 0.999999)))
            crop_y2 = max(crop_y1 + 1, min(mask_h, int(y2 * scale_y + 0.999999)))
            cropped = mask[..., crop_y1:crop_y2, crop_x1:crop_x2]

            if cropped.shape[-2] <= 0 or cropped.shape[-1] <= 0:
                result.append(mask)
                debug_rows.append(f"{mask_h}x{mask_w}->fallback")
            else:
                result.append(cropped)
                debug_rows.append(f"{mask_h}x{mask_w}->{cropped.shape[-2]}x{cropped.shape[-1]}")

        if attention_debug_enabled():
            print(
                f"{_LOG_V34} exact detailer crop "
                f"label={label or crop_ctx.get('label', 'detailer')} "
                f"region={crop_ctx['crop_region']} "
                f"full={crop_ctx.get('full_width')}x{crop_ctx.get('full_height')} masks={debug_rows}"
            )
        return result
    except Exception as error:
        print(f"{_LOG_V34} crop mask failed: {error}; using full masks")
        return source_masks


def normalize_detailer_segment_mask(segment_mask: Any) -> torch.Tensor | None:
    """Coerce a detected segment mask to a contiguous 2-D CPU float tensor in ``[0, 1]``."""
    if segment_mask is None:
        return None
    try:
        if isinstance(segment_mask, torch.Tensor):
            mask = segment_mask.detach().to(device="cpu", dtype=torch.float32)
        else:
            mask = torch.as_tensor(segment_mask, dtype=torch.float32, device="cpu")
        while mask.ndim > 2 and mask.shape[0] == 1:
            mask = mask.squeeze(0)
        while mask.ndim > 2 and mask.shape[-1] == 1:
            mask = mask.squeeze(-1)
        if mask.ndim != 2 or mask.numel() <= 0:
            return None
        return mask.contiguous().clamp(0.0, 1.0)
    except Exception as error:
        if attention_debug_enabled():
            print(f"{_LOG_V35} invalid detailer segment mask: {error}")
        return None


def clone_model_with_detailer_crop(
    model: Any,
    crop_region: Any,
    full_width: int,
    full_height: int,
    label: str = "detailer",
    segment_mask: Any = None,
) -> Any:
    """Clone ``model`` and attach exact detailer crop + segment metadata to it."""
    cloned = model.clone()
    transformer_options = cloned.model_options.setdefault("transformer_options", {})
    transformer_options["saya_couple_crop"] = {
        "crop_region": [float(v) for v in crop_region],
        "full_width": int(full_width),
        "full_height": int(full_height),
        "label": str(label),
        "segment_mask": normalize_detailer_segment_mask(segment_mask),
    }
    return cloned


def _resize_segment_mask(segment_mask: Any, target_mask: Any) -> torch.Tensor | None:
    """Bilinearly resize a crop-local segment mask to ``target_mask``'s resolution."""
    normalized = normalize_detailer_segment_mask(segment_mask)
    if normalized is None or not isinstance(target_mask, torch.Tensor):
        return None
    target_height = int(target_mask.shape[-2])
    target_width = int(target_mask.shape[-1])
    if target_height <= 0 or target_width <= 0:
        return None
    resized = F.interpolate(
        normalized.view(1, 1, *normalized.shape),
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    return resized.to(device=target_mask.device, dtype=torch.float32).clamp(0.0, 1.0)


def dominant_detailer_owner(cropped_masks: Any, crop_ctx: Any) -> int | None:
    """Return the index of the character that clearly owns a detected crop.

    Each ownership mask is scored by its mean value inside the detected segment
    (or the whole crop when no segment is given). The top scorer must reach 70 %
    of the total score *and* beat the runner-up by 30 % to be accepted;
    otherwise ``None`` (leave the crop under soft regional control).
    """
    if crop_ctx is None or not isinstance(cropped_masks, (list, tuple)):
        return None
    tensor_masks = [mask for mask in cropped_masks if isinstance(mask, torch.Tensor)]
    if len(tensor_masks) < 2 or len(tensor_masks) != len(cropped_masks):
        return None

    segment_weight = _resize_segment_mask(crop_ctx.get("segment_mask"), tensor_masks[0])
    if segment_weight is None or float(segment_weight.sum().item()) <= 1e-06:
        segment_weight = torch.ones(
            tensor_masks[0].shape[-2:], device=tensor_masks[0].device, dtype=torch.float32
        )

    scores: list[float] = []
    for mask in tensor_masks:
        owner_mask = mask.to(dtype=torch.float32)
        weight = segment_weight
        while weight.ndim < owner_mask.ndim:
            weight = weight.unsqueeze(0)
        weight = weight.expand_as(owner_mask)
        denominator = weight.sum().clamp_min(1e-06)
        scores.append(float((owner_mask * weight).sum().item() / denominator.item()))

    total = sum(max(0.0, score) for score in scores)
    if total <= 1e-06:
        return None

    ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
    owner_index, owner_score = ranked[0]
    second_score = ranked[1][1]
    confidence = max(0.0, owner_score) / total
    margin = max(0.0, owner_score - second_score) / total
    if confidence < 0.70 or margin < 0.30:
        return None
    return int(owner_index)


def prepare_detailer_region_masks(
    source_masks: Any, crop_ctx: Any, label: str = ""
) -> tuple[Any, int | None]:
    """Crop the regional masks and, if one owner is clear, hard-lock the crop to it.

    Returns ``(masks, owner_index)``. ``owner_index`` is ``None`` when no clear
    owner is found (masks stay soft); otherwise the returned masks are 1/0
    indicators for the locked owner.
    """
    cropped_masks = crop_masks_to_detailer_region(source_masks, crop_ctx, label)
    owner_index = dominant_detailer_owner(cropped_masks, crop_ctx)
    if owner_index is None:
        return (cropped_masks, None)

    locked_masks: list[torch.Tensor] = []
    for index, mask in enumerate(cropped_masks):
        if not isinstance(mask, torch.Tensor):
            return (cropped_masks, None)
        locked_masks.append(
            torch.ones_like(mask) if index == owner_index else torch.zeros_like(mask)
        )
    if attention_debug_enabled():
        print(
            f"{_LOG_V35} detailer owner lock "
            f"label={label or crop_ctx.get('label', 'detailer')} owner={owner_index + 1}"
        )
    return (locked_masks, owner_index)


def masks_match_attention_pass(
    source_masks: Any, original_shape: Any = None
) -> tuple[bool, str]:
    """Heuristically decide whether ``source_masks`` describe the current pass.

    Returns ``(ok, reason)``. ``ok`` is ``False`` when the mask is much larger
    than the (latent x8) pass size or its aspect ratio differs by more than 35 %,
    which usually means we are inside a small detailer crop and the full-image
    masks should not be trusted.
    """
    try:
        shape = first_valid_mask_shape(source_masks)
        if shape is None or original_shape is None:
            return (True, "unknown")

        mask_h, mask_w = shape
        pass_h = int(original_shape[-2])
        pass_w = int(original_shape[-1])
        if mask_h <= 0 or mask_w <= 0 or pass_h <= 0 or pass_w <= 0:
            return (True, "bad_shape_unknown")

        mask_aspect = mask_w / max(1, mask_h)
        pass_aspect = pass_w / max(1, pass_h)
        aspect_ratio = max(mask_aspect, pass_aspect) / max(1e-06, min(mask_aspect, pass_aspect))
        approx_pass_h = pass_h * 8
        approx_pass_w = pass_w * 8
        much_smaller_crop = approx_pass_h < mask_h * 0.7 and approx_pass_w < mask_w * 0.7
        aspect_mismatch = aspect_ratio > 1.35

        ok = not (much_smaller_crop or aspect_mismatch)
        reason = (
            f"aspect_ratio={aspect_ratio:.2f} crop_smaller={much_smaller_crop} "
            f"mask={mask_h}x{mask_w} pass={pass_h}x{pass_w}"
        )
        return (ok, reason)
    except Exception as error:
        return (True, f"detect_failed:{error}")


def factor_token_grid(token_count: int, original_shape: Any = None) -> tuple[int, int]:
    """Infer the ``(h, w)`` grid that produced a flattened ``token_count``.

    Prefers an exact ``h * w == token_count`` at one of the standard UNet
    downsample rates of ``original_shape``; then a perfect square; then the
    integer factor pair whose aspect ratio best matches ``original_shape`` (or
    that is closest to square when no shape is given).
    """
    token_count = int(token_count)
    if token_count <= 0:
        return (1, 1)

    aspect: float | None = None
    if original_shape is not None:
        try:
            shape_h = int(original_shape[-2])
            shape_w = int(original_shape[-1])
        except Exception:
            shape_h = 0
            shape_w = 0
        if shape_h > 0 and shape_w > 0:
            for rate in (1, 2, 4, 8, 16, 32, 64):
                grid_h = max(1, shape_h // rate)
                grid_w = max(1, shape_w // rate)
                if grid_h * grid_w == token_count:
                    return (grid_h, grid_w)
            aspect = shape_w / max(1, shape_h)

    side = int(round(token_count**0.5))
    if side * side == token_count:
        return (side, side)

    best: tuple[int, int] | None = None
    best_score: float | None = None
    for grid_h in range(1, int(token_count**0.5) + 1):
        if token_count % grid_h == 0:
            grid_w = token_count // grid_h
            if aspect is None:
                score = abs(grid_w - grid_h)
            else:
                score = abs(grid_w / max(1, grid_h) - aspect)
            if best_score is None or score < best_score:
                best_score = score
                best = (grid_h, grid_w)
    return best if best is not None else (1, token_count)


def resize_masks_to_token_grid(
    masks: torch.Tensor, target_tokens: int, original_shape: Any = None
) -> torch.Tensor:
    """Nearest-resample ``[batch, tokens, channels]`` masks to ``target_tokens``.

    Both the source and target token counts must factor into a clean 2-D grid;
    otherwise the input is returned unchanged with a warning.
    """
    if masks.shape[1] == target_tokens:
        return masks

    old_tokens = int(masks.shape[1])
    new_tokens = int(target_tokens)
    old_h, old_w = factor_token_grid(old_tokens, None)
    new_h, new_w = factor_token_grid(new_tokens, original_shape)
    if old_h * old_w != old_tokens or new_h * new_w != new_tokens:
        print(
            f"{_LOG_V33} impossible resize: masks={tuple(masks.shape)} target_tokens={new_tokens}"
        )
        return masks

    batch, _tokens, channels = masks.shape
    grid = masks.permute(0, 2, 1).contiguous().reshape(batch * channels, 1, old_h, old_w)
    grid = F.interpolate(grid.float(), size=(new_h, new_w), mode="nearest")
    resized = grid.reshape(batch, channels, new_tokens).permute(0, 2, 1).contiguous()
    resized = resized.to(device=masks.device, dtype=masks.dtype).clamp(0, 1)
    if attention_debug_enabled():
        print(
            f"{_LOG_V33} resized mask {old_tokens} ({old_h}x{old_w}) -> "
            f"{new_tokens} ({new_h}x{new_w})"
        )
    return resized


def fill_unassigned_mask_regions(
    masks_v: torch.Tensor, original_shape: Any = None
) -> torch.Tensor:
    """Assign token positions with zero total weight to their nearest region centre.

    ``masks_v`` is ``[regions, batch, tokens, channels]``. For every batch item
    with uncovered tokens, each region's spatial centre of mass is computed and
    every empty token is one-hot assigned to the closest centre.
    """
    try:
        if not isinstance(masks_v, torch.Tensor) or masks_v.ndim != 4:
            return masks_v
        region_count, batch, tokens, _channels = masks_v.shape
        if region_count <= 1 or tokens <= 0:
            return masks_v

        grid_h, grid_w = factor_token_grid(tokens, original_shape)
        if grid_h * grid_w != tokens:
            return masks_v

        out = masks_v.clone()
        weights = out.mean(dim=-1).clamp(0, 1)
        device = out.device
        dtype = out.dtype
        row_coords = (
            torch.arange(grid_h, device=device, dtype=dtype)
            .view(grid_h, 1)
            .expand(grid_h, grid_w)
            .reshape(tokens)
        )
        col_coords = (
            torch.arange(grid_w, device=device, dtype=dtype)
            .view(1, grid_w)
            .expand(grid_h, grid_w)
            .reshape(tokens)
        )
        total_filled = 0
        fully_empty_batches = 0

        for batch_index in range(batch):
            batch_weights = weights[:, batch_index, :]
            empty = batch_weights.sum(dim=0) <= 1e-06
            if not bool(empty.any()):
                continue

            empty_idx = empty.nonzero(as_tuple=False).flatten()
            total_filled += int(empty_idx.numel())
            if not bool((batch_weights.sum(dim=1) > 1e-06).any()):
                fully_empty_batches += 1

            centers_x: list[torch.Tensor] = []
            centers_y: list[torch.Tensor] = []
            for region in range(region_count):
                region_map = batch_weights[region]
                region_total = region_map.sum()
                if region_total > 1e-06:
                    center_x = (region_map * col_coords).sum() / region_total
                    center_y = (region_map * row_coords).sum() / region_total
                else:
                    center_x = torch.tensor(
                        (region + 0.5) * grid_w / max(1, region_count),
                        device=device,
                        dtype=dtype,
                    )
                    center_y = torch.tensor((grid_h - 1) * 0.5, device=device, dtype=dtype)
                centers_x.append(center_x)
                centers_y.append(center_y)
            centers_x = torch.stack(centers_x)
            centers_y = torch.stack(centers_y)

            empty_x = col_coords[empty_idx]
            empty_y = row_coords[empty_idx]
            dist = (centers_x[:, None] - empty_x[None, :]) ** 2 + (
                centers_y[:, None] - empty_y[None, :]
            ) ** 2
            assignment = dist.argmin(dim=0)
            out[:, batch_index, empty_idx, :] = 0
            for region in range(region_count):
                region_idx = empty_idx[assignment == region]
                if region_idx.numel() > 0:
                    out[region, batch_index, region_idx, :] = 1

        debug = os.environ.get(_DEBUG_ENV_VAR, "0") == "1"
        if debug and total_filled > 0:
            print(
                f"{_LOG_V33} filled empty mask tokens={total_filled}, "
                f"fully_empty_batches={fully_empty_batches}, grid={grid_h}x{grid_w}"
            )
        return out
    except Exception as error:
        print(f"{_LOG_V33} nearest fill failed: {error}")
        return masks_v


def prepare_masks_for_attention_output(
    masks: Any, qkv: torch.Tensor, original_shape: Any = None, label: str = ""
) -> torch.Tensor:
    """Reshape/broadcast ``masks`` so they multiply an attention output ``qkv``.

    Aligns dtype/device, token count (resampling if needed), channel count and
    batch. On any unrecoverable mismatch returns ``ones_like(qkv)`` (neutral).
    """
    try:
        if not isinstance(masks, torch.Tensor):
            return masks
        masks = masks.to(device=qkv.device, dtype=qkv.dtype)
        if masks.ndim == 2:
            masks = masks.unsqueeze(-1)
        if masks.ndim != 3:
            print(f"{_LOG_V33} bad mask ndim={getattr(masks, 'ndim', None)}, fallback neutral")
            return torch.ones_like(qkv)

        if masks.shape[1] != qkv.shape[1]:
            masks = resize_masks_to_token_grid(masks, qkv.shape[1], original_shape)
        if masks.shape[1] != qkv.shape[1]:
            print(
                f"{_LOG_V33} token mismatch still present "
                f"masks={tuple(masks.shape)} qkv={tuple(qkv.shape)}, fallback neutral"
            )
            return torch.ones_like(qkv)

        if masks.shape[2] == 1 and qkv.shape[2] != 1:
            masks = masks.expand(-1, -1, qkv.shape[2])
        elif masks.shape[2] != qkv.shape[2]:
            if masks.shape[2] > qkv.shape[2]:
                masks = masks[:, :, : qkv.shape[2]]
            else:
                masks = masks[:, :, :1].expand(-1, -1, qkv.shape[2])

        if masks.shape[0] != qkv.shape[0]:
            old_batch = masks.shape[0]
            target_batch = qkv.shape[0]
            if old_batch == 1:
                masks = masks.expand(target_batch, -1, -1)
            else:
                reps = (target_batch + old_batch - 1) // old_batch
                masks = masks.repeat(reps, 1, 1)[:target_batch]
            print(f"{_LOG_V2} aligned batch {old_batch} -> {target_batch}")
        return masks
    except Exception as error:
        print(f"{_LOG_V33} prepare failed: {error}, fallback neutral")
        return torch.ones_like(qkv)


def build_query_token_masks(masks: Sequence[Any], q: torch.Tensor, original_shape: Any) -> torch.Tensor:
    """Project spatial ownership masks onto the flattened query-token layout.

    The downsample rate is inferred by matching ``q``'s token count against
    ``original_shape`` at rates 1/2/4/8. Non-tensor entries (e.g. ``False``)
    become an all-ones mask. Result: ``[len(masks) * B, N, C]``.
    """
    if original_shape[2] * original_shape[3] == q.shape[1]:
        down_sample_rate = 1
    elif original_shape[2] // 2 * (original_shape[3] // 2) == q.shape[1]:
        down_sample_rate = 2
    elif original_shape[2] // 4 * (original_shape[3] // 4) == q.shape[1]:
        down_sample_rate = 4
    else:
        down_sample_rate = 8

    projected: list[torch.Tensor] = []
    for mask in masks:
        if isinstance(mask, torch.Tensor):
            size = (
                original_shape[2] // down_sample_rate,
                original_shape[3] // down_sample_rate,
            )
            downsampled = F.interpolate(mask.unsqueeze(0), size=size, mode="nearest")
            downsampled = downsampled.view(1, -1, 1).repeat(q.shape[0], 1, q.shape[2])
            projected.append(downsampled)
        else:
            projected.append(torch.ones_like(q))
    return torch.cat(projected, dim=0)
