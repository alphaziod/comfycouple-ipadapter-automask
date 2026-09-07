"""Standard crop-aware Impact Pack detailer node."""

from __future__ import annotations

from typing import Any, Self

from . import runtime
from .configuration import DetailerOptions
from .pipeline import process_detected_regions
from .schemas import build_detailer_input_schema


class SayaDetailerForEach:
    """Impact Pack detailer wrapper that propagates exact regional crop metadata."""

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "doit"
    CATEGORY = "ImpactPack/Detailer"
    DESCRIPTION = (
        "Enhance each detected region while preserving exact crop metadata "
        "for Saya regional attention."
    )

    @classmethod
    def INPUT_TYPES(cls: type[Self]) -> dict[str, Any]:
        """Return the ComfyUI input schema exposed by this node."""
        return build_detailer_input_schema(include_retry_limit=False)

    @staticmethod
    def get_core_module() -> Any:
        """Return the initialized Impact Pack core module."""
        runtime.initialize_impact_pack_runtime()
        return runtime.core

    def doit(
        self: Self,
        image: Any,
        segs: Any,
        model: Any,
        clip: Any,
        vae: Any,
        guide_size: float,
        guide_size_for: bool,
        max_size: float,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        positive: Any,
        negative: Any,
        denoise: float,
        feather: float,
        noise_mask: bool,
        force_inpaint: bool,
        wildcard: str,
        cycle: int = 1,
        detailer_hook: Any = None,
        inpaint_model: bool = False,
        noise_mask_feather: float = 0,
        scheduler_func_opt: Any = None,
        tiled_encode: bool = False,
        tiled_decode: bool = False,
    ) -> tuple[Any]:
        """Execute one detailer pass and return the enhanced image."""
        options = DetailerOptions(
            model=model,
            clip=clip,
            vae=vae,
            guide_size=guide_size,
            guide_size_for_bbox=guide_size_for,
            max_size=max_size,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            denoise=denoise,
            feather=feather,
            noise_mask=noise_mask,
            force_inpaint=force_inpaint,
            wildcard=wildcard,
            detailer_hook=detailer_hook,
            cycle=cycle,
            inpaint_model=inpaint_model,
            noise_mask_feather=noise_mask_feather,
            scheduler_func=scheduler_func_opt,
            tiled_encode=tiled_encode,
            tiled_decode=tiled_decode,
            max_retries=1,
        )
        enhanced_image, *_ = process_detected_regions(image, segs, options)
        return (enhanced_image,)
