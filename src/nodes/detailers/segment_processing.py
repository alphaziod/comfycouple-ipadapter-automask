"""Per-segment model cloning, enhancement, retry, and paste operations."""

from __future__ import annotations

from typing import Any

from ..regional_attention.masks import clone_model_with_detailer_crop
from . import runtime
from .configuration import (
    DetailerOptions,
    DetailerResults,
    SegmentAction,
    WildcardSelection,
    crop_conditioning,
    is_dummy_model,
    select_segment_wildcard,
)


def clone_models_for_segment(
    image: Any, segment: Any, model: Any, refiner_model: Any
) -> tuple[Any, Any]:
    """Clone main and refiner models with exact crop metadata attached."""
    detail_model = model
    detail_refiner_model = refiner_model
    full_width = int(image.shape[2])
    full_height = int(image.shape[1])
    label = str(getattr(segment, "label", "detailer"))
    if (
        refiner_model is not None
        and (not isinstance(refiner_model, str))
        and hasattr(refiner_model, "clone")
    ):
        detail_refiner_model = clone_model_with_detailer_crop(
            model=refiner_model,
            crop_region=segment.crop_region,
            full_width=full_width,
            full_height=full_height,
            label=f"{label} · refiner",
            segment_mask=segment.cropped_mask,
        )
    if not is_dummy_model(model) and hasattr(model, "clone"):
        detail_model = clone_model_with_detailer_crop(
            model=model,
            crop_region=segment.crop_region,
            full_width=full_width,
            full_height=full_height,
            label=label,
            segment_mask=segment.cropped_mask,
        )
        if runtime.detailer_debug_enabled():
            print(
                f"[Saya Detailer] label={label} crop={tuple(segment.crop_region)} full={full_width}x{full_height}"
            )
    return (detail_model, detail_refiner_model)


def enhance_segment(
    cropped_image: Any,
    segment: Any,
    segment_seed: int,
    cropped_positive: Any,
    cropped_negative: Any,
    cropped_mask: Any,
    wildcard_item: Any,
    wildcard_selection: WildcardSelection,
    options: DetailerOptions,
    detail_model: Any,
    detail_refiner_model: Any,
) -> tuple[Any, Any]:
    """Run Impact Pack enhancement and retry rejected patches when configured."""
    if is_dummy_model(options.model):
        return (cropped_image, None)
    enhanced_image = cropped_image
    control_images = None
    for retry_index in range(max(1, int(options.max_retries))):
        enhanced_image, control_images = runtime.core.enhance_detail(
            cropped_image,
            detail_model,
            options.clip,
            options.vae,
            options.guide_size,
            options.guide_size_for_bbox,
            options.max_size,
            segment.bbox,
            segment_seed + retry_index,
            options.steps,
            options.cfg,
            options.sampler_name,
            options.scheduler,
            cropped_positive,
            cropped_negative,
            options.denoise,
            cropped_mask,
            options.force_inpaint,
            wildcard_opt=wildcard_item,
            wildcard_opt_concat_mode=wildcard_selection.concat_mode,
            detailer_hook=options.detailer_hook,
            refiner_ratio=options.refiner_ratio,
            refiner_model=detail_refiner_model,
            refiner_clip=options.refiner_clip,
            refiner_positive=options.refiner_positive,
            refiner_negative=options.refiner_negative,
            control_net_wrapper=segment.control_net_wrapper,
            cycle=options.cycle,
            inpaint_model=options.inpaint_model,
            noise_mask_feather=options.noise_mask_feather,
            scheduler_func=options.scheduler_func,
            vae_tiled_encode=options.tiled_encode,
            vae_tiled_decode=options.tiled_decode,
        )
        should_retry = (
            options.detailer_hook is not None
            and options.detailer_hook.should_retry_patch(enhanced_image)
        )
        if not should_retry:
            return (enhanced_image, control_images)
        if retry_index + 1 >= max(1, int(options.max_retries)):
            raise RuntimeError("Max retries reached")
        print("Detect bad patch, retrying...")
    return (enhanced_image, control_images)


def process_single_segment(
    image: Any,
    segment: Any,
    index: int,
    wildcard_selection: WildcardSelection,
    options: DetailerOptions,
    results: DetailerResults,
) -> tuple[Any, SegmentAction]:
    """Enhance one segment, paste it into the image, and collect metadata."""
    cropped_image = runtime.utils.to_tensor(
        runtime.utils.crop_ndarray4(image.cpu().numpy(), segment.crop_region)
    )
    mask = runtime.utils.tensor_gaussian_blur_mask(
        runtime.utils.to_tensor(segment.cropped_mask), options.feather
    )
    if (segment.cropped_mask == 0).all().item():
        runtime.logging.info("Detailer: segment skip [empty mask]")
        return (image, "continue")
    # Derive the default seed from the segment's own position rather than its
    # rank in the ordered list, so the same physical location keeps the same
    # seed regardless of detector ordering jitter between runs.
    position_offset = (
        int(segment.bbox[0]) * 73856093 ^ int(segment.bbox[1]) * 19349663
    ) % 1_000_000
    segment_seed, wildcard_item = select_segment_wildcard(
        wildcard_selection, segment, options.seed + position_offset
    )
    if wildcard_item and wildcard_item.strip() == "[SKIP]":
        return (image, "continue")
    if wildcard_item and wildcard_item.strip() == "[STOP]":
        return (image, "stop")
    cropped_positive = crop_conditioning(options.positive, image, segment.crop_region)
    cropped_negative = crop_conditioning(options.negative, image, segment.crop_region)
    cropped_mask = segment.cropped_mask if options.noise_mask else None
    original_crop = cropped_image.clone()
    detail_model, detail_refiner_model = clone_models_for_segment(
        image, segment, options.model, options.refiner_model
    )
    enhanced_image, control_images = enhance_segment(
        cropped_image,
        segment,
        segment_seed,
        cropped_positive,
        cropped_negative,
        cropped_mask,
        wildcard_item,
        wildcard_selection,
        options,
        detail_model,
        detail_refiner_model,
    )
    if control_images is not None:
        results.control_images.extend(control_images)
    new_segment_image = None
    if enhanced_image is not None:
        image = image.cpu()
        enhanced_image = enhanced_image.cpu()
        runtime.utils.tensor_paste(
            image, enhanced_image, (segment.crop_region[0], segment.crop_region[1]), mask
        )
        results.enhanced_images.append(enhanced_image)
        if options.detailer_hook is not None:
            image = options.detailer_hook.post_paste(image)
        alpha_image = runtime.utils.tensor_convert_rgba(enhanced_image)
        resized_mask = runtime.utils.tensor_resize(
            mask, *runtime.utils.tensor_get_size(enhanced_image)
        )
        runtime.utils.tensor_putalpha(alpha_image, resized_mask)
        results.enhanced_alpha_images.append(alpha_image)
        new_segment_image = enhanced_image.numpy()
    results.cropped_images.append(original_crop)
    results.updated_segments.append(
        runtime.SEG(
            new_segment_image,
            segment.cropped_mask,
            segment.confidence,
            segment.crop_region,
            segment.bbox,
            segment.label,
            segment.control_net_wrapper,
        )
    )
    return (image, "continue")
