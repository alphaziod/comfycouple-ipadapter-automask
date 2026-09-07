"""Configuration, wildcard parsing, and shared state for detailer execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from . import runtime

SegmentAction = Literal["continue", "stop"]


@dataclass(frozen=True, slots=True)
class DetailerOptions:
    """Immutable options required to enhance detected segments."""

    model: Any
    clip: Any
    vae: Any
    guide_size: float
    guide_size_for_bbox: bool
    max_size: float
    seed: int
    steps: int
    cfg: float
    sampler_name: str
    scheduler: str
    positive: Any
    negative: Any
    denoise: float
    feather: float
    noise_mask: bool
    force_inpaint: bool
    wildcard: str | None = None
    detailer_hook: Any = None
    refiner_ratio: float | None = None
    refiner_model: Any = None
    refiner_clip: Any = None
    refiner_positive: Any = None
    refiner_negative: Any = None
    cycle: int = 1
    inpaint_model: bool = False
    noise_mask_feather: float = 0
    scheduler_func: Any = None
    tiled_encode: bool = False
    tiled_decode: bool = False
    max_retries: int = 1


@dataclass(slots=True)
class DetailerResults:
    """Mutable collections built while segments are processed."""

    cropped_images: list[Any] = field(default_factory=list)
    enhanced_images: list[Any] = field(default_factory=list)
    enhanced_alpha_images: list[Any] = field(default_factory=list)
    control_images: list[Any] = field(default_factory=list)
    updated_segments: list[Any] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class WildcardSelection:
    """Parsed wildcard routing state for one detailer execution."""

    mode: str | None
    chooser: Any
    concat_mode: str | None


def parse_wildcard_selection(wildcard: str | None) -> WildcardSelection:
    """Parse wildcard ordering and CONCAT mode once per detailer execution."""
    if wildcard is None:
        return WildcardSelection(mode=None, chooser=None, concat_mode=None)
    concat_mode: str | None = None
    normalized = wildcard
    if normalized.startswith("[CONCAT]"):
        concat_mode = "concat"
        normalized = normalized[8:]
    mode, chooser = runtime.wildcards.process_wildcard_for_segs(normalized)
    return WildcardSelection(mode=mode, chooser=chooser, concat_mode=concat_mode)


def order_segments(segments: list[Any], mode: str | None) -> list[Any]:
    """Return segments in the ordering requested by the wildcard mode.

    When no explicit mode is requested, the order previously fell back to
    whatever the detector pipeline produced (list(segments)). That order is
    not guaranteed stable across runs on near-identical images, and each
    segment's default seed is derived from its index in this list (see
    segment_processing.py), so a permutation silently swaps which seed lands
    on which segment. The default is therefore a deterministic top-to-bottom,
    left-to-right spatial sort.
    """
    if mode == "ASC":
        return sorted(segments, key=lambda segment: (segment.bbox[0], segment.bbox[1]))
    if mode == "DSC":
        return sorted(
            segments, key=lambda segment: (segment.bbox[0], segment.bbox[1]), reverse=True
        )
    if mode == "ASC-SIZE":
        return sorted(segments, key=segment_area)
    if mode == "DSC-SIZE":
        return sorted(segments, key=segment_area, reverse=True)
    return sorted(segments, key=lambda segment: (segment.bbox[1], segment.bbox[0]))


def crop_conditioning(conditioning: Any, image: Any, crop_region: Any) -> Any:
    """Crop conditioning masks to one segment while preserving all other metadata."""
    if isinstance(conditioning, str):
        return conditioning
    return [
        [
            condition,
            {
                key: (
                    runtime.core.crop_condition_mask(value, image, crop_region)
                    if key == "mask"
                    else value
                )
                for key, value in details.items()
            },
        ]
        for condition, details in conditioning
    ]


def select_segment_wildcard(
    selection: WildcardSelection, segment: Any, default_seed: int
) -> tuple[int, Any]:
    """Resolve the seed and wildcard item assigned to one segment."""
    if selection.chooser is None:
        return (default_seed, None)
    if selection.mode == "LAB":
        wildcard_item = selection.chooser.get(segment)
        return (default_seed, wildcard_item)
    selected_seed, wildcard_item = selection.chooser.get(segment)
    return (default_seed if selected_seed is None else selected_seed, wildcard_item)


def prepare_detailer_model(model: Any, noise_mask_feather: float) -> Any:
    """Apply differential diffusion once when feathered denoise masks require it."""
    if (
        not is_dummy_model(model)
        and noise_mask_feather > 0
        and ("denoise_mask_function" not in model.model_options)
    ):
        return runtime.utils.apply_differential_diffusion(model)
    return model


def is_dummy_model(model: Any) -> bool:
    """Return whether the Impact Pack dummy model disables inference."""
    return isinstance(model, str) and model == "DUMMY"


def segment_area(segment: Any) -> float:
    """Return the bounding-box area used by size-based wildcard ordering."""
    return float(segment.bbox[2] - segment.bbox[0]) * float(segment.bbox[3] - segment.bbox[1])
