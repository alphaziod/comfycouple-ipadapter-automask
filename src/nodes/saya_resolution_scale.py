"""Resolution / upscale-target calculator nodes.

* :class:`SayaResolutionScaleCalculator` - pick an exact generation resolution
  from a preset ladder, or derive one from a source image / custom aspect.
* :class:`SayaUpscalePresetModelLoader` - choose a final-upscale pixel budget and
  load the upscale model.
* :class:`SayaUpscaleTargetCalculator` - turn that budget into concrete
  aspect-preserving, 16-divisible dimensions for a source image.
"""

from __future__ import annotations

import math
from typing import Any


class SayaResolutionScaleCalculator:
    """Resolution calculator copied from the local DaSiWa node and simplified for Saya."""

    # Existing exact presets are preserved. Missing useful aspect ratios are appended
    # with one AI-friendly ~1 MP size divisible by 32.
    FIXED_RESOLUTION_PRESETS = {
        # Existing 16:9 ladder
        "Landscape 16:9 · 768x432": (768, 432),
        "Landscape 16:9 · 896x512": (896, 512),
        "Landscape 16:9 · 1024x576": (1024, 576),
        "Landscape 16:9 · 1152x648": (1152, 648),
        "Landscape 16:9 · 1280x720": (1280, 720),
        "Landscape 16:9 · 1344x768": (1344, 768),
        "Landscape 16:9 · 1536x864": (1536, 864),

        # Existing wide ladder
        "Landscape Wide · 1024x576": (1024, 576),
        "Landscape Wide · 1152x640": (1152, 640),
        "Landscape Wide · 1280x704": (1280, 704),
        "Landscape Wide · 1408x768": (1408, 768),
        "Landscape Wide · 1536x832": (1536, 832),

        # Existing 3:2 ladder
        "Landscape 3:2 · 960x640": (960, 640),
        "Landscape 3:2 · 1152x768": (1152, 768),
        "Landscape 3:2 · 1216x832": (1216, 832),

        # Existing portrait ladders
        "Portrait 2:3 · 768x1152": (768, 1152),
        "Portrait 2:3 · 832x1216": (832, 1216),
        "Portrait 7:9 · 768x992": (768, 992),
        "Portrait 7:9 · 896x1152": (896, 1152),

        # Existing square ladder
        "Square · 896x896": (896, 896),
        "Square · 1024x1024": (1024, 1024),

        # Added missing portrait ratios, ~1 MP and Div32
        "Portrait 4:5 · 896x1120": (896, 1120),
        "Portrait 3:4 · 864x1152": (864, 1152),
        "Portrait 9:16 · 768x1376": (768, 1376),
        "Portrait 5:12 · 640x1536": (640, 1536),

        # Added missing landscape ratios, ~1 MP and Div32
        "Landscape 5:4 · 1120x896": (1120, 896),
        "Landscape 4:3 · 1152x864": (1152, 864),
        "Landscape 9:7 · 1152x896": (1152, 896),
        "Landscape 12:5 · 1536x640": (1536, 640),
    }

    PRESETS = FIXED_RESOLUTION_PRESETS

    ASPECT_PRESETS = {
        "1:1 - Square": (1, 1),
        "4:5 - Portrait": (4, 5),
        "3:4 - Portrait": (3, 4),
        "7:9 - Portrait": (7, 9),
        "2:3 - Portrait": (2, 3),
        "9:16 - Portrait": (9, 16),
        "5:12 - Portrait": (5, 12),
        "5:4 - Landscape": (5, 4),
        "4:3 - Landscape": (4, 3),
        "9:7 - Landscape": (9, 7),
        "3:2 - Landscape": (3, 2),
        "16:9 - Landscape": (16, 9),
        "12:5 - Landscape": (12, 5),
        "CUSTOM": (0, 0),
    }

    DESCRIPTION = """
    Saya Resolution Scale Calculator

    Exact Saya presets only:
    - clean labels
    - AI-friendly divisible sizes
    - exact dimensions for generation
    - no weird CivitAI / Scene naming

    Fixed presets return their dimensions verbatim.
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "resolution_preset": (
                    list(cls.PRESETS.keys()),
                    {
                        "default": "Landscape 16:9 · 1344x768",
                        "description": "Exact Saya generation resolutions only.",
                    },
                ),
                "no_scale": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "ON (Source Dims)",
                        "label_off": "OFF (Calculated)",
                        "description": (
                            "Bypass calculations and output the source dimensions."
                        ),
                    },
                ),
                "scale_from_image": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "IMAGE ASPECT",
                        "label_off": "USE ASPECT BELOW",
                        "description": (
                            "Used by megapixel targets. Fixed presets ignore the "
                            "image aspect."
                        ),
                    },
                ),
                "aspect_preset_when_not_image": (
                    list(cls.ASPECT_PRESETS.keys()),
                    {
                        "default": "16:9 - Landscape",
                        "description": (
                            "Used only by megapixel targets when IMAGE ASPECT is off."
                        ),
                    },
                ),
                "swap_aspect_when_not_image": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "yes",
                        "label_off": "no",
                        "description": (
                            "Flip width and height. Also rotates a fixed preset."
                        ),
                    },
                ),
                "custom_aspect_width": (
                    "INT",
                    {
                        "default": 16,
                        "min": 1,
                        "max": 9999,
                        "step": 1,
                        "description": "Custom aspect width when CUSTOM is selected.",
                    },
                ),
                "custom_aspect_height": (
                    "INT",
                    {
                        "default": 9,
                        "min": 1,
                        "max": 9999,
                        "step": 1,
                        "description": "Custom aspect height when CUSTOM is selected.",
                    },
                ),
                "mode": (
                    ["WAN/LTX (Div32)", "FLUX/SDXL (Div8)", "Custom Divisor"],
                    {
                        "default": "WAN/LTX (Div32)",
                        "description": "Divisor snapping mode.",
                    },
                ),
                "custom_divisor": (
                    "INT",
                    {
                        "default": 8,
                        "min": 1,
                        "max": 512,
                        "step": 1,
                        "description": "Divisor used when Custom Divisor is selected.",
                    },
                ),
            },
            "optional": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("INT", "INT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("width_int", "height_int", "width_float", "height_float")
    FUNCTION = "calculate"
    CATEGORY = "Saya/Scaling"

    def calculate(
        self,
        resolution_preset: str,
        no_scale: bool,
        scale_from_image: bool,
        aspect_preset_when_not_image: str,
        swap_aspect_when_not_image: bool,
        custom_aspect_width: int,
        custom_aspect_height: int,
        mode: str,
        custom_divisor: int,
        image=None,
    ):
        if image is not None:
            _, image_h, image_w, _ = image.shape
            source_w, source_h = int(image_w), int(image_h)
        else:
            source_w, source_h = 1024, 1024

        if no_scale:
            width, height = source_w, source_h
            if swap_aspect_when_not_image:
                width, height = height, width
            return width, height, float(width), float(height)

        if resolution_preset in self.FIXED_RESOLUTION_PRESETS:
            width, height = self.FIXED_RESOLUTION_PRESETS[resolution_preset]
            if swap_aspect_when_not_image:
                width, height = height, width
            return width, height, float(width), float(height)

        target_mp = float(self.PRESETS[resolution_preset])

        if scale_from_image and image is not None:
            aspect_w, aspect_h = source_w, source_h
        else:
            if aspect_preset_when_not_image == "CUSTOM":
                aspect_w, aspect_h = custom_aspect_width, custom_aspect_height
            else:
                aspect_w, aspect_h = self.ASPECT_PRESETS[aspect_preset_when_not_image]

            if swap_aspect_when_not_image:
                aspect_w, aspect_h = aspect_h, aspect_w

        if aspect_w <= 0 or aspect_h <= 0:
            aspect_w, aspect_h = 16, 9

        ratio = aspect_w / aspect_h
        total_pixels = target_mp * 1_000_000
        width = math.sqrt(total_pixels * ratio)
        height = width / ratio

        if mode == "WAN/LTX (Div32)":
            divisor = 32
        elif mode == "FLUX/SDXL (Div8)":
            divisor = 8
        else:
            divisor = max(1, int(custom_divisor))

        width = max(divisor, round(width / divisor) * divisor)
        height = max(divisor, round(height / divisor) * divisor)

        return int(width), int(height), float(width), float(height)


class SayaUpscalePresetModelLoader:
    """Visible final-upscale settings node: target preset plus upscale model."""

    PRESETS = {
        "1080p · FAST · 1920x1080 equivalent": 1920 * 1080,
        "2K · BALANCED · 2560x1440 equivalent": 2560 * 1440,
        "3K · HIGH · 3200x1800 equivalent": 3200 * 1800,
        "4K · MAX · 3840x2160 equivalent": 3840 * 2160,
    }

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        import folder_paths

        models = list(folder_paths.get_filename_list("upscale_models"))
        preferred = "RealESRGAN_x4plus_anime_6B.safetensors"
        default_model = preferred if preferred in models else (models[0] if models else "")

        model_options: tuple[Any, ...]
        if models:
            model_options = (
                models,
                {
                    "default": default_model,
                    "description": "Upscale model used by the final automatic pass.",
                },
            )
        else:
            model_options = (
                [""],
                {
                    "default": "",
                    "description": "No upscale model was found in models/upscale_models.",
                },
            )

        return {
            "required": {
                "target_preset": (
                    list(cls.PRESETS.keys()),
                    {
                        "default": "4K · MAX · 3840x2160 equivalent",
                        "description": (
                            "Pixel budget equivalent. The source aspect ratio is "
                            "preserved and dimensions are rounded to multiples of 16."
                        ),
                    },
                ),
                "upscale_model": model_options,
            },
        }

    RETURN_TYPES = ("UPSCALE_MODEL", "INT", "STRING")
    RETURN_NAMES = ("upscale_model", "target_pixels", "preset_name")
    FUNCTION = "load"
    CATEGORY = "Saya/Scaling"

    def load(self, target_preset: str, upscale_model: str):
        if not upscale_model:
            raise RuntimeError(
                "No upscale model is available. Put one in "
                "ComfyUI/models/upscale_models."
            )

        from comfy_extras.nodes_upscale_model import UpscaleModelLoader

        loaded_model = UpscaleModelLoader().load_model(upscale_model)[0]
        target_pixels = int(self.PRESETS[target_preset])
        return loaded_model, target_pixels, target_preset


class SayaUpscaleTargetCalculator:
    """Calculate an aspect-preserving target close to the selected pixel budget.

    The selected 1080p / 2K / 3K / 4K preset represents a standard 16:9
    pixel budget. Landscape, portrait, square, and wide source ratios are
    preserved. Final dimensions are snapped to multiples of 16.
    """

    DIVISOR = 16

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),
                "target_pixels": (
                    "INT",
                    {
                        "default": 3840 * 2160,
                        "min": 256 * 256,
                        "max": 16384 * 16384,
                        "step": 1,
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "FLOAT", "FLOAT")
    RETURN_NAMES = (
        "image",
        "target_width",
        "target_height",
        "model_factor",
        "target_megapixels",
    )
    FUNCTION = "calculate"
    CATEGORY = "Saya/Scaling"

    @classmethod
    def _best_target(
        cls,
        source_w: int,
        source_h: int,
        target_pixels: int,
    ) -> tuple[int, int]:
        source_w = max(1, int(source_w))
        source_h = max(1, int(source_h))
        target_pixels = max(1, int(target_pixels))

        ratio = source_w / source_h
        divisor = cls.DIVISOR

        ideal_w = math.sqrt(target_pixels * ratio)
        ideal_h = ideal_w / ratio

        center_w = max(divisor, round(ideal_w / divisor) * divisor)
        center_h = max(divisor, round(ideal_h / divisor) * divisor)

        candidates: set[tuple[int, int]] = set()

        for offset in range(-24, 25):
            width = max(divisor, center_w + offset * divisor)
            height = max(divisor, round((width / ratio) / divisor) * divisor)
            candidates.add((width, height))

            height = max(divisor, center_h + offset * divisor)
            width = max(divisor, round((height * ratio) / divisor) * divisor)
            candidates.add((width, height))

        def score(candidate: tuple[int, int]) -> tuple[float, float, int]:
            width, height = candidate
            candidate_ratio = width / height
            ratio_error = abs(candidate_ratio - ratio) / ratio
            pixel_error = abs((width * height) - target_pixels) / target_pixels
            dimensional_drift = abs(width - ideal_w) + abs(height - ideal_h)

            # Preserve composition first, then match the selected pixel budget.
            combined = ratio_error * 8.0 + pixel_error
            return combined, dimensional_drift, width * height

        return min(candidates, key=score)

    def calculate(self, image, target_pixels: int):
        _, source_h, source_w, _ = image.shape
        source_w = int(source_w)
        source_h = int(source_h)

        target_w, target_h = self._best_target(
            source_w,
            source_h,
            int(target_pixels),
        )

        required_factor = max(
            target_w / max(1, source_w),
            target_h / max(1, source_h),
        )

        # WLSH supports a floating upscale factor. Keep the useful model range;
        # the exact Lanczos fit directly after it lands on the selected target.
        model_factor = min(4.0, max(1.0, required_factor))
        target_megapixels = (target_w * target_h) / 1_000_000.0

        return (
            image,
            int(target_w),
            int(target_h),
            float(round(model_factor, 4)),
            float(target_megapixels),
        )


# Backward-compatible alias for workflows made before preset support.
SayaNear4KTargetCalculator = SayaUpscaleTargetCalculator
