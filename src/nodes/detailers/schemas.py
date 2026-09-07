"""ComfyUI input schemas shared by the crop-aware detailer nodes."""

from __future__ import annotations

from typing import Any

from . import runtime


def build_detailer_input_schema(*, include_retry_limit: bool) -> dict[str, Any]:
    """Return the shared Impact Pack detailer schema.

    Args:
        include_retry_limit: Add the ``max_retries`` widget used by the retry node.

    Returns:
        A ComfyUI ``INPUT_TYPES`` dictionary.

    """
    runtime.initialize_impact_pack_runtime()

    required: dict[str, Any] = {
        "image": ("IMAGE",),
        "segs": ("SEGS",),
        "model": (
            "MODEL",
            {
                "tooltip": (
                    "If ImpactDummyInput is connected to the model, "
                    "the inference stage is skipped."
                )
            },
        ),
        "clip": ("CLIP",),
        "vae": ("VAE",),
        "guide_size": (
            "FLOAT",
            {
                "default": 512,
                "min": 64,
                "max": runtime.nodes.MAX_RESOLUTION,
                "step": 8,
            },
        ),
        "guide_size_for": (
            "BOOLEAN",
            {"default": True, "label_on": "bbox", "label_off": "crop_region"},
        ),
        "max_size": (
            "FLOAT",
            {
                "default": 1024,
                "min": 64,
                "max": runtime.nodes.MAX_RESOLUTION,
                "step": 8,
            },
        ),
        "seed": ("INT", {"default": 0, "min": 0, "max": 18446744073709551615}),
        "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
        "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
        "sampler_name": (runtime.comfy.samplers.KSampler.SAMPLERS,),
        "scheduler": (runtime.core.get_schedulers(),),
        "positive": ("CONDITIONING",),
        "negative": ("CONDITIONING",),
        "denoise": (
            "FLOAT",
            {"default": 0.5, "min": 0.0001, "max": 1.0, "step": 0.01},
        ),
        "feather": ("INT", {"default": 5, "min": 0, "max": 100, "step": 1}),
        "noise_mask": (
            "BOOLEAN",
            {"default": True, "label_on": "enabled", "label_off": "disabled"},
        ),
        "force_inpaint": (
            "BOOLEAN",
            {"default": True, "label_on": "enabled", "label_off": "disabled"},
        ),
        "wildcard": ("STRING", {"multiline": True, "dynamicPrompts": False}),
        "cycle": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
    }

    if include_retry_limit:
        required["max_retries"] = (
            "INT",
            {"default": 1, "min": 1, "max": 10, "step": 1},
        )

    optional: dict[str, Any] = {
        "detailer_hook": ("DETAILER_HOOK",),
        "inpaint_model": (
            "BOOLEAN",
            {"default": False, "label_on": "enabled", "label_off": "disabled"},
        ),
        "noise_mask_feather": (
            "INT",
            {"default": 20, "min": 0, "max": 100, "step": 1},
        ),
        "scheduler_func_opt": ("SCHEDULER_FUNC",),
        "tiled_encode": (
            "BOOLEAN",
            {"default": False, "label_on": "enabled", "label_off": "disabled"},
        ),
        "tiled_decode": (
            "BOOLEAN",
            {"default": False, "label_on": "enabled", "label_off": "disabled"},
        ),
    }

    return {"required": required, "optional": optional}
