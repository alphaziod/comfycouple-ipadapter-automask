"""Top-level crop-aware detailer pipeline."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from . import runtime
from .configuration import (
    DetailerOptions,
    DetailerResults,
    order_segments,
    parse_wildcard_selection,
    prepare_detailer_model,
)
from .segment_processing import process_single_segment


def process_detected_regions(
    image: Any, segments: Any, options: DetailerOptions
) -> tuple[Any, list[Any], list[Any], list[Any], list[Any], Any]:
    """Process every detected region and return the full Impact Pack result tuple."""
    runtime.initialize_impact_pack_runtime()
    if len(image) > 1:
        raise ValueError(
            "SayaDetailerForEach does not allow image batches. Use an Impact Pack batching detailer workflow instead."
        )
    working_image = image.clone()
    scaled_segments = runtime.core.segs_scale_match(segments, working_image.shape)
    wildcard_selection = parse_wildcard_selection(options.wildcard)
    ordered_segments = order_segments(scaled_segments[1], wildcard_selection.mode)
    model = prepare_detailer_model(options.model, options.noise_mask_feather)
    effective_options = replace(options, model=model)
    results = DetailerResults()
    for index, segment in enumerate(ordered_segments):
        working_image, action = process_single_segment(
            working_image, segment, index, wildcard_selection, effective_options, results
        )
        if action == "stop":
            break
    results.cropped_images.sort(key=lambda item: item.shape, reverse=True)
    results.enhanced_images.sort(key=lambda item: item.shape, reverse=True)
    results.enhanced_alpha_images.sort(key=lambda item: item.shape, reverse=True)
    return (
        runtime.utils.tensor_convert_rgb(working_image),
        results.cropped_images,
        results.enhanced_images,
        results.enhanced_alpha_images,
        results.control_images,
        (scaled_segments[0], results.updated_segments),
    )
