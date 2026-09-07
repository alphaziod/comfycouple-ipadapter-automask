"""Prompt-bundle nodes that let isolated hires phases rebuild a couple.

* :class:`SayaCouplePromptBundlePack` - serialize the four prompts + couple
  config into one JSON string that survives ComfyUI's metadata round-trip.
* :class:`SayaCouplePromptBundleUnpack` - restore them (plus a combined positive
  prompt) from that string.
* :class:`SayaLatentShapeFromImage` - a zero latent carrying only an image's
  shape, so couple masks can be rebuilt for a later phase.
"""

from __future__ import annotations

import json
from typing import Any, Self

import torch

from .couple_config import COUPLE_CONFIG_TYPE, normalize_couple_config

#: Schema version stored in a packed prompt bundle.
_BUNDLE_VERSION = 2


class SayaCouplePromptBundlePack:
    """Pack the four external couple prompts into one manifest-safe JSON string."""

    @classmethod
    def INPUT_TYPES(cls: type[Self]) -> dict[str, Any]:
        """Return the ComfyUI input schema exposed by this node."""
        return {
            "required": {
                "base_prompt": ("STRING", {"default": "", "multiline": True}),
                "person_1_prompt": ("STRING", {"default": "", "multiline": True}),
                "person_2_prompt": ("STRING", {"default": "", "multiline": True}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "couple_config": (COUPLE_CONFIG_TYPE,),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt_bundle_json",)
    FUNCTION = "pack"
    CATEGORY = "saya/image phases"

    def pack(
        self: Self,
        base_prompt: str,
        person_1_prompt: str,
        person_2_prompt: str,
        negative_prompt: str,
        couple_config: Any = None,
    ) -> tuple[str]:
        """Serialize the four prompts and the normalized couple config to JSON."""
        payload = {
            "version": _BUNDLE_VERSION,
            "base_prompt": str(base_prompt or ""),
            "person_1_prompt": str(person_1_prompt or ""),
            "person_2_prompt": str(person_2_prompt or ""),
            "negative_prompt": str(negative_prompt or ""),
            "couple_config": normalize_couple_config(couple_config),
        }
        return (json.dumps(payload, ensure_ascii=False, separators=(",", ":")),)


class SayaCouplePromptBundleUnpack:
    """Restore the four prompts, a combined positive prompt and the couple config."""

    @classmethod
    def INPUT_TYPES(cls: type[Self]) -> dict[str, Any]:
        """Return the ComfyUI input schema exposed by this node."""
        return {
            "required": {
                "prompt_bundle_json": (
                    "STRING",
                    {"default": "", "multiline": True, "forceInput": True},
                )
            }
        }

    RETURN_TYPES = (
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        COUPLE_CONFIG_TYPE,
    )
    RETURN_NAMES = (
        "base_prompt",
        "person_1_prompt",
        "person_2_prompt",
        "negative_prompt",
        "combined_positive_prompt",
        "couple_config",
    )
    FUNCTION = "unpack"
    CATEGORY = "saya/image phases"

    def unpack(
        self: Self, prompt_bundle_json: str
    ) -> tuple[str, str, str, str, str, dict[str, Any]]:
        """Parse the bundle; on any JSON error treat the whole input as the base prompt.

        Person 2 is blanked when the config says it is disabled, and the combined
        positive prompt joins the non-empty base / P1 / P2 prompts with ``", "``.
        """
        raw = str(prompt_bundle_json or "")
        base = raw
        person_1 = ""
        person_2 = ""
        negative = ""
        config = normalize_couple_config()
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                base = str(parsed.get("base_prompt", ""))
                person_1 = str(parsed.get("person_1_prompt", ""))
                person_2 = str(parsed.get("person_2_prompt", ""))
                negative = str(parsed.get("negative_prompt", ""))
                config = normalize_couple_config(parsed.get("couple_config"))
        except json.JSONDecodeError:
            pass

        if not config["person_2_enabled"]:
            person_2 = ""
        combined = ", ".join(
            text.strip() for text in (base, person_1, person_2) if text and text.strip()
        )
        return base, person_1, person_2, negative, combined, config


class SayaLatentShapeFromImage:
    """Create a zero latent that only carries an image's dimensions (for mask rebuild)."""

    @classmethod
    def INPUT_TYPES(cls: type[Self]) -> dict[str, Any]:
        """Return the ComfyUI input schema exposed by this node."""
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent_shape",)
    FUNCTION = "build"
    CATEGORY = "saya/image phases"

    def build(self: Self, image: Any) -> tuple[dict[str, Any]]:
        """Return ``{"samples": zeros[batch, 4, H//8, W//8]}`` for the given image."""
        if not isinstance(image, torch.Tensor) or image.ndim != 4:
            raise ValueError(
                "Saya Latent Shape From Image attend une IMAGE [batch, height, width, channels]."
            )
        batch, height, width, _channels = image.shape
        latent_height = max(1, int(height) // 8)
        latent_width = max(1, int(width) // 8)
        samples = torch.zeros(
            (int(batch), 4, latent_height, latent_width),
            dtype=torch.float32,
            device=image.device,
        )
        return ({"samples": samples},)
