"""Two small "rescue" nodes:

* :class:`DualClipTextEncoderNode` - a CLIP text encoder with an explicit
  "disabled" state (returns an empty conditioning instead of encoding).
* :class:`HiresModelRouterNode` - routes one shared positive/negative plus a pool
  of models/VAEs to the four hires stages (base / mid / final / last).
"""

from __future__ import annotations

from typing import Any, Self

from ..services.models import build_model_choice_list, load_vae_or_fallback

#: Model slots the hires router can select from.
_MODEL_SOURCES = ["main", "dual_sampling", "support1", "usdu1", "usdu2"]
#: Hires stages driven by the router, in order.
_HIRES_STAGES = ["base", "mid", "final", "last"]


class DualClipTextEncoderNode:
    """Encode prompt text, or emit an empty conditioning when ``send_data`` is off."""

    @classmethod
    def INPUT_TYPES(cls: type[Self]) -> dict[str, Any]:
        """Return the ComfyUI input schema exposed by this node."""
        return {
            "required": {
                "clip": ("CLIP",),
                "text": ("STRING", {"multiline": True, "default": ""}),
                "send_data": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = "saya/rescue"

    def encode(self: Self, clip: Any, text: str, send_data: bool = True) -> Any:
        """Return the encoded conditioning, or ``([],)`` when ``send_data`` is False."""
        if not send_data:
            return ([],)
        tokens = clip.tokenize(text)
        return (clip.encode_from_tokens_scheduled(tokens),)


class HiresModelRouterNode:
    """Route one shared conditioning + a model/VAE pool to the four hires stages."""

    @classmethod
    def INPUT_TYPES(cls: type[Self]) -> dict[str, Any]:
        """Return the ComfyUI input schema exposed by this node."""
        vae_names = build_model_choice_list(
            "vae",
            [
                "AAA%20Anime%20VAE%20SDXL%20v2.safetensors",
                "Dark%20VAE%20SDXL%20Weak.safetensors",
                "crystalVAESDXL_vaeV3.safetensors",
            ],
        )
        vae_choice = [
            "none",
            "main",
            "dual_sampling",
            "support1",
            "usdu1",
            "usdu2",
            "custom_vae_1",
            "custom_vae_2",
            "custom_vae_3",
        ]
        required: dict[str, Any] = {
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
            "main_model": ("MODEL",),
            "main_vae": ("VAE",),
            "dual_sampling_model": ("MODEL",),
            "dual_sampling_vae": ("VAE",),
            "support1_model": ("MODEL",),
            "support1_vae": ("VAE",),
            "usdu1_model": ("MODEL",),
            "usdu1_vae": ("VAE",),
            "usdu2_model": ("MODEL",),
            "usdu2_vae": ("VAE",),
            "━━ CUSTOM VAE ━━": ("STRING", {"default": "━━ CUSTOM VAE ━━"}),
            "custom_vae_1": (vae_names,),
            "custom_vae_2": (vae_names,),
            "custom_vae_3": (vae_names,),
        }
        for stage in _HIRES_STAGES:
            required[f"━━ {stage.upper()} HIRES ━━"] = (
                "STRING",
                {"default": f"━━ {stage.upper()} HIRES ━━"},
            )
            required[f"{stage}_source"] = (_MODEL_SOURCES, {"default": "main"})
            required[f"{stage}_vae"] = (vae_choice, {"default": "none"})
        return {"required": required}

    RETURN_TYPES = (
        "CONDITIONING",
        "CONDITIONING",
        "MODEL",
        "VAE",
        "MODEL",
        "VAE",
        "MODEL",
        "VAE",
        "MODEL",
        "VAE",
    )
    RETURN_NAMES = (
        "positive",
        "negative",
        "base_model",
        "base_vae",
        "mid_model",
        "mid_vae",
        "final_model",
        "final_vae",
        "last_model",
        "last_vae",
    )
    FUNCTION = "route"
    CATEGORY = "saya/rescue"

    def route(self: Self, **kwargs: Any) -> Any:
        """Return ``(positive, negative, then (model, vae) per hires stage)``."""
        models = {
            "main": kwargs["main_model"],
            "dual_sampling": kwargs["dual_sampling_model"],
            "support1": kwargs["support1_model"],
            "usdu1": kwargs["usdu1_model"],
            "usdu2": kwargs["usdu2_model"],
        }
        vaes = {
            "main": kwargs["main_vae"],
            "dual_sampling": kwargs["dual_sampling_vae"],
            "support1": kwargs["support1_vae"],
            "usdu1": kwargs["usdu1_vae"],
            "usdu2": kwargs["usdu2_vae"],
        }
        vaes["custom_vae_1"] = load_vae_or_fallback(kwargs.get("custom_vae_1"), kwargs["main_vae"])
        vaes["custom_vae_2"] = load_vae_or_fallback(kwargs.get("custom_vae_2"), kwargs["main_vae"])
        vaes["custom_vae_3"] = load_vae_or_fallback(kwargs.get("custom_vae_3"), kwargs["main_vae"])

        out: list[Any] = [kwargs["positive"], kwargs["negative"]]
        for stage in _HIRES_STAGES:
            source_name = kwargs.get(f"{stage}_source", "main")
            vae_choice_value = kwargs.get(f"{stage}_vae", "none")
            out.append(models.get(source_name, kwargs["main_model"]))
            out.append(
                vaes.get(
                    source_name if vae_choice_value == "none" else vae_choice_value,
                    kwargs["main_vae"],
                )
            )
        return tuple(out)
