"""The ``Saya Comfy Couple`` node and its config-driven ``COPY`` variant.

:class:`CoupleConditioningNode` is the user-facing entry point. Given Base / P1 /
P2 / Negative conditionings and a latent it:

1. builds the two complementary geometric ownership masks from ``center`` /
   ``transition`` / ``mask_floor`` / ``orientation``;
2. decides a *mode* - ``SOLO`` (Person 2 off), ``FALLBACK`` (unusable
   conditioning shapes) or ``COUPLE``;
3. in ``COUPLE`` mode, builds the joint ``[Base;P1;P2]`` binding context and
   patches every connected model (main, dual sampler, up to three supports) via
   :class:`~..regional_attention.node.RegionalAttentionNode`, de-duplicating
   models that are the same object.

:class:`CoupleConditioningCopyNode` runs the same pipeline from a stored
``SAYA_COUPLE_CONFIG`` only, so hires phases reproduce the exact same couple.
"""

from __future__ import annotations

import os
from typing import Any, Self

try:
    import torch
except Exception:  # pragma: no cover - torch is always present under ComfyUI
    torch = None

from .couple_config import BINDING_MODES, COUPLE_CONFIG_TYPE, normalize_couple_config
from .regional_attention.node import RegionalAttentionNode

_DEBUG_ENV_VAR = "SAYA_COUPLE_DEBUG"
_LOG_PREFIX = "[Saya Comfy Couple]"
_ERR_PREFIX = "Saya Comfy Couple"

#: Keys copied from a normalized config into the ``self_binding`` mapping passed
#: to the regional-attention node.
_SELF_BINDING_KEYS: tuple[tuple[str, Any], ...] = (
    ("attn1_strength", 0.6),
    ("attn1_max_tokens", 1024),
    ("attn1_ambiguous_band", 0.15),
    ("attn1_apply_lowres_only", True),
    ("attn1_hires_strength", 0.30),
    ("v2_ownership_strength", 0.80),
    ("v2_confidence_floor", 0.10),
)


class CoupleConditioningNode:
    """Build two character regions and patch every connected model consistently."""

    @classmethod
    def INPUT_TYPES(cls: type[Self]) -> dict[str, Any]:
        """Return the ComfyUI input schema exposed by this node."""
        return {
            "required": {
                "model_main": ("MODEL",),
                "main_positive": ("CONDITIONING",),
                "person_1_positive": ("CONDITIONING",),
                "person_2_positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent": ("LATENT",),
                "use_couple_attention": ("BOOLEAN", {"default": True}),
                "orientation": (["horizontal", "vertical"], {"default": "horizontal"}),
                "center": ("FLOAT", {"default": 0.5, "min": 0.15, "max": 0.85, "step": 0.01}),
                "transition": ("FLOAT", {"default": 0.03, "min": 0.01, "max": 0.20, "step": 0.01}),
                "mask_floor": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.20, "step": 0.01}),
                "swap_person_positions": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "binding_mode": (list(BINDING_MODES), {"default": "LEGACY"}),
                "attn1_strength": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.5, "step": 0.05}),
                "attn1_max_tokens": ("INT", {"default": 1024, "min": 256, "max": 16384, "step": 256}),
                "attn1_ambiguous_band": (
                    "FLOAT",
                    {"default": 0.15, "min": 0.02, "max": 0.45, "step": 0.01},
                ),
                "attn1_apply_lowres_only": ("BOOLEAN", {"default": True}),
                "attn1_hires_strength": (
                    "FLOAT",
                    {"default": 0.30, "min": 0.0, "max": 0.80, "step": 0.05},
                ),
                "v2_ownership_strength": (
                    "FLOAT",
                    {"default": 0.80, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "v2_confidence_floor": (
                    "FLOAT",
                    {"default": 0.10, "min": 0.0, "max": 0.45, "step": 0.01},
                ),
                "dual_sampling_model": ("MODEL",),
                "support_model_1": ("MODEL",),
                "support_model_2": ("MODEL",),
                "support_model_3": ("MODEL",),
            },
        }

    RETURN_TYPES = (
        "MODEL",
        "MODEL",
        "MODEL",
        "MODEL",
        "MODEL",
        "CONDITIONING",
        "CONDITIONING",
        "CONDITIONING",
        "MASK",
        "MASK",
        COUPLE_CONFIG_TYPE,
    )
    RETURN_NAMES = (
        "patched_model_main",
        "patched_dual_sampling_model",
        "patched_support_model_1",
        "patched_support_model_2",
        "patched_support_model_3",
        "positive_final",
        "detailer_positive",
        "negative",
        "mask_person_1",
        "mask_person_2",
        "couple_config",
    )
    FUNCTION = "run"
    CATEGORY = "saya/rescue"

    # ------------------------------------------------------------------ #
    # Conditioning helpers                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def copy_conditioning(conditioning: Any) -> Any:
        """Shallow-copy a CONDITIONING list: share the tensors, copy the metadata dicts."""
        if not conditioning:
            return []
        return [[entry[0], dict(entry[1])] for entry in conditioning]

    @staticmethod
    def describe_conditioning_error(name: str, conditioning: Any) -> Any:
        """Return a human-readable validation error for one conditioning, or ``None``."""
        if conditioning is None or conditioning == []:
            return None
        if not isinstance(conditioning, list):
            return f"{name} is not a CONDITIONING list"
        for index, entry in enumerate(conditioning):
            if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                return f"{name}[{index}] is not a [tensor, metadata] entry"
            if not isinstance(entry[0], torch.Tensor) or entry[0].ndim != 3:
                return f"{name}[{index}] context is not a rank-3 tensor"
            if not isinstance(entry[1], dict):
                return f"{name}[{index}] metadata is not a dictionary"
        return None

    @classmethod
    def build_regional_conditioning(cls: type[Self], conditioning: Any, mask: Any) -> Any:
        """Copy ``conditioning`` and attach one regional ``mask`` (strength 1, no bounds)."""
        result = cls.copy_conditioning(conditioning)
        for entry in result:
            entry[1]["mask"] = mask
            entry[1]["mask_strength"] = 1.0
            entry[1]["set_area_to_bounds"] = False
        return result

    @classmethod
    def build_couple_region(
        cls: type[Self], main_positive: Any, person_positive: Any, mask: Any
    ) -> Any:
        """Concatenate global Base and one character conditioning, then attach ``mask``."""
        if len(person_positive) != 1 or len(main_positive) > 1:
            raise ValueError(
                "a couple region requires one person entry and at most one Base entry"
            )
        if main_positive:
            from nodes import ConditioningConcat

            region = ConditioningConcat().concat(main_positive, person_positive)[0]
        else:
            region = cls.copy_conditioning(person_positive)
        return cls.build_regional_conditioning(region, mask)

    @staticmethod
    def log_debug_message(message: str) -> None:
        """Print ``message`` only when ``SAYA_COUPLE_DEBUG=1``."""
        if os.environ.get(_DEBUG_ENV_VAR, "0") == "1":
            print(f"{_LOG_PREFIX} {message}")

    # ------------------------------------------------------------------ #
    # Mask geometry                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def read_latent_dimensions(latent: Any) -> Any:
        """Validate ``latent`` and return ``(samples, batch, image_height, image_width)``."""
        if not isinstance(latent, dict) or "samples" not in latent:
            raise ValueError(f"{_ERR_PREFIX}: latent must contain latent['samples']")
        samples = latent["samples"]
        if torch is None or not isinstance(samples, torch.Tensor):
            raise ValueError(f"{_ERR_PREFIX}: latent['samples'] must be a torch.Tensor")
        if samples.ndim != 4:
            raise ValueError(
                f"{_ERR_PREFIX}: latent['samples'] must have shape [batch, channels, height, width]"
            )
        batch, _channels, latent_height, latent_width = samples.shape
        if batch < 1 or latent_height < 1 or latent_width < 1:
            raise ValueError(
                f"{_ERR_PREFIX}: latent['samples'] has an empty batch or spatial dimension"
            )
        return (samples, batch, latent_height * 8, latent_width * 8)

    @classmethod
    def build_character_masks(
        cls: type[Self],
        latent: Any,
        use_couple_attention: bool,
        orientation: str,
        center: float,
        transition: float,
        mask_floor: float,
        swap_person_positions: bool,
    ) -> Any:
        """Build the two complementary ownership masks ``(person_1, person_2)``.

        With ``use_couple_attention`` off the masks are all-ones / all-zeros
        (SOLO). Otherwise Person 1 owns a linear ramp along the chosen axis:
        ``1`` before ``center - transition/2``, ``0`` after ``center +
        transition/2``, blended in between; ``mask_floor`` lifts the plateaus off
        pure 0/1 (A/B testing only); ``swap_person_positions`` flips the ramp.
        """
        samples, batch, height, width = cls.read_latent_dimensions(latent)
        dtype = samples.dtype if samples.is_floating_point() else torch.float32
        device = samples.device
        if not use_couple_attention:
            return (
                torch.ones((batch, height, width), device=device, dtype=dtype),
                torch.zeros((batch, height, width), device=device, dtype=dtype),
            )

        axis_size = width if orientation == "horizontal" else height
        coordinate = (torch.arange(axis_size, device=device, dtype=torch.float32) + 0.5) / axis_size
        transition = min(0.20, max(0.01, float(transition)))
        mask_floor = min(0.20, max(0.0, float(mask_floor)))
        start = float(center) - transition / 2.0
        ownership = ((start + transition - coordinate) / transition).clamp(0.0, 1.0)
        person_1_axis = mask_floor + (1.0 - 2.0 * mask_floor) * ownership
        if swap_person_positions:
            person_1_axis = 1.0 - person_1_axis

        if orientation == "horizontal":
            person_1 = person_1_axis.view(1, 1, width).expand(batch, height, width)
        else:
            person_1 = person_1_axis.view(1, height, 1).expand(batch, height, width)
        person_1 = person_1.to(dtype=dtype).clamp(0.0, 1.0)
        person_2 = (1.0 - person_1).clamp(0.0, 1.0)
        return (person_1, person_2)

    # ------------------------------------------------------------------ #
    # Model patching                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def regional_attention_node_class() -> Any:
        """Resolve the regional-attention node lazily to avoid import cycles."""
        return RegionalAttentionNode

    @staticmethod
    def _self_binding_from_config(config: dict[str, Any]) -> dict[str, Any]:
        """Extract the attn1 / V2 parameters the regional-attention node consumes."""
        return {key: config.get(key, default) for key, default in _SELF_BINDING_KEYS}

    @classmethod
    def patch_model_with_regional_attention(
        cls: type[Self],
        model: Any,
        positive: Any,
        negative: Any,
        branch: str,
        detailer_positive: Any = None,
        binding_mode: str = "LEGACY",
        binding_contexts: Any = None,
        binding_masks: Any = None,
        binding_carrier: Any = None,
        self_binding: Any = None,
    ) -> Any:
        """Clone + patch one connected model; ``(None, None, None)`` when ``model`` is None."""
        if model is None:
            return (None, None, None)
        try:
            attention_couple = cls.regional_attention_node_class()()
            cls.log_debug_message(
                f"branch={branch} patch=install regions={len(positive)} "
                f"native_token_lengths={[entry[0].shape[1] for entry in positive]} padding=none"
            )
            return attention_couple.attention_couple(
                model,
                positive,
                negative,
                "Attention",
                detailer_positive=detailer_positive,
                binding_mode=binding_mode,
                binding_contexts=binding_contexts,
                binding_masks=binding_masks,
                binding_carrier=binding_carrier,
                self_binding=self_binding,
            )
        except Exception as exc:
            raise RuntimeError(
                f"{_ERR_PREFIX}: failed to patch connected {branch} model: {exc}"
            ) from exc

    def _patch_branches(
        self,
        branches: tuple[tuple[Any, str], ...],
        *,
        couple_positive: Any,
        negative_public: Any,
        detailer_positive: Any,
        binding_mode: str,
        binding_contexts: Any,
        binding_masks: Any,
        binding_carrier: Any,
        self_binding: dict[str, Any],
    ) -> list[Any]:
        """Patch every non-None branch model, reusing the result for identical objects."""
        patched: list[Any] = []
        patched_by_identity: dict[int, Any] = {}
        for model, branch in branches:
            if model is None:
                patched.append((None, None, None))
                continue
            identity = id(model)
            if identity in patched_by_identity:
                self.log_debug_message(f"branch={branch} patch=reuse identical_model")
                patched.append(patched_by_identity[identity])
                continue
            result = self.patch_model_with_regional_attention(
                model,
                couple_positive,
                negative_public,
                branch,
                detailer_positive=detailer_positive,
                binding_mode=binding_mode,
                binding_contexts=binding_contexts,
                binding_masks=binding_masks,
                binding_carrier=binding_carrier,
                self_binding=self_binding,
            )
            patched_by_identity[identity] = result
            patched.append(result)
        return patched

    # ------------------------------------------------------------------ #
    # Orchestration                                                     #
    # ------------------------------------------------------------------ #

    def _run_with_config(
        self: Self,
        model_main: Any,
        main_positive: Any,
        person_1_positive: Any,
        person_2_positive: Any,
        negative: Any,
        latent: Any,
        config: Any,
        dual_sampling_model: Any = None,
        support_model_1: Any = None,
        support_model_2: Any = None,
        support_model_3: Any = None,
    ) -> Any:
        """Execute one master/copy pass from an already resolved configuration."""
        config = normalize_couple_config(config)
        person_2_enabled = bool(config["person_2_enabled"])
        use_couple_attention = bool(config["use_couple_attention"])
        orientation = str(config["orientation"])
        center = float(config["center"])
        transition = float(config["transition"])
        mask_floor = float(config["mask_floor"])
        swap_person_positions = bool(config["swap_person_positions"])
        binding_mode = str(config.get("binding_mode", "LEGACY"))
        self_binding = self._self_binding_from_config(config)

        mask_p1, mask_p2 = self.build_character_masks(
            latent,
            use_couple_attention,
            orientation,
            center,
            transition,
            mask_floor,
            swap_person_positions,
        )

        structure_error = next(
            (
                error
                for error in (
                    self.describe_conditioning_error("Base", main_positive),
                    self.describe_conditioning_error("Person 1", person_1_positive),
                    self.describe_conditioning_error("Person 2", person_2_positive),
                    self.describe_conditioning_error("Negative", negative),
                )
                if error
            ),
            None,
        )
        if structure_error:
            raise ValueError(f"{_ERR_PREFIX}: invalid conditioning structure: {structure_error}")

        main_public = self.copy_conditioning(main_positive)
        p1_public = self.copy_conditioning(person_1_positive)
        p2_public = self.copy_conditioning(person_2_positive) if person_2_enabled else []
        negative_public = self.copy_conditioning(negative)
        branches = (
            (model_main, "main"),
            (dual_sampling_model, "dual-sampling"),
            (support_model_1, "support 1"),
            (support_model_2, "support 2"),
            (support_model_3, "support 3"),
        )
        counts = (len(main_public), len(p1_public), len(p2_public), len(negative_public))

        # --- pick a mode -------------------------------------------------
        fallback_reason = None
        if not use_couple_attention:
            mode = "SOLO"
            if main_public and p1_public:
                from nodes import ConditioningConcat

                positive_final = ConditioningConcat().concat(main_public, p1_public)[0]
            else:
                positive_final = self.copy_conditioning(main_public or p1_public)
            detailer_positive = self.copy_conditioning(p1_public)
        elif not p1_public:
            mode, fallback_reason = ("FALLBACK", "Person 1 conditioning is empty")
        elif not p2_public:
            mode, fallback_reason = ("FALLBACK", "Person 2 conditioning is empty")
        elif not negative_public:
            mode, fallback_reason = ("FALLBACK", "Negative conditioning is empty")
        elif (
            len(main_public) > 1
            or len(p1_public) != 1
            or len(p2_public) != 1
            or len(negative_public) != 1
        ):
            mode, fallback_reason = (
                "FALLBACK",
                "multi-entry conditioning cannot form exactly two stable logical regions",
            )
        else:
            mode = "COUPLE"

        if mode == "FALLBACK":
            positive_final = (
                main_public + self.copy_conditioning(p1_public) + self.copy_conditioning(p2_public)
            )
            detailer_positive = self.copy_conditioning(p1_public) + self.copy_conditioning(p2_public)

        self.log_debug_message(
            f"mode={mode} entries(base,p1,p2,neg)={counts} "
            f"person2_enabled={person_2_enabled} fallback={fallback_reason or 'none'}"
        )
        if os.environ.get(_DEBUG_ENV_VAR, "0") == "1":
            mask_sum = mask_p1 + mask_p2
            self.log_debug_message(
                f"masks transition={transition:.3f} floor={mask_floor:.3f} "
                f"p1(min={mask_p1.min().item():.3f},max={mask_p1.max().item():.3f},"
                f"mean={mask_p1.float().mean().item():.3f}) "
                f"p2(min={mask_p2.min().item():.3f},max={mask_p2.max().item():.3f},"
                f"mean={mask_p2.float().mean().item():.3f}) "
                f"sum(min={mask_sum.min().item():.3f},max={mask_sum.max().item():.3f},"
                f"mean={mask_sum.float().mean().item():.3f})"
            )

        # --- COUPLE: build the joint context and patch every model ------
        if mode == "COUPLE":
            region_p1 = self.build_couple_region(main_public, p1_public, mask_p1)
            region_p2 = self.build_couple_region(main_public, p2_public, mask_p2)
            couple_positive = region_p1 + region_p2
            binding_contexts = {
                "base": main_public[0][0] if main_public else None,
                "person_1": p1_public[0][0],
                "person_2": p2_public[0][0],
            }
            binding_masks = (mask_p1, mask_p2)
            binding_carrier = self.copy_conditioning(main_public or p1_public)
            # Detailer prompts stay character-local: global Base/composition tokens
            # are intentionally excluded from tiny crops to preserve detail quality.
            detailer_positive = self.build_regional_conditioning(
                p1_public, mask_p1
            ) + self.build_regional_conditioning(p2_public, mask_p2)

            patched = self._patch_branches(
                branches,
                couple_positive=couple_positive,
                negative_public=negative_public,
                detailer_positive=detailer_positive,
                binding_mode=binding_mode,
                binding_contexts=binding_contexts,
                binding_masks=binding_masks,
                binding_carrier=binding_carrier,
                self_binding=self_binding,
            )
            models = tuple(item[0] for item in patched)
            positive_final = self.copy_conditioning(patched[0][1])
            negative_public = self.copy_conditioning(patched[0][2])

            pooled_source = "Base" if main_public else "Person"
            region_summary = [
                (
                    tuple(entry[0].shape),
                    (
                        tuple(entry[1]["pooled_output"].shape)
                        if isinstance(entry[1].get("pooled_output"), torch.Tensor)
                        else None
                    ),
                )
                for entry in couple_positive
            ]
            self.log_debug_message(
                f"logical_regions=2 region(context,pooled)={region_summary} "
                f"pooled_source={pooled_source} patch=installed binding={binding_mode}"
            )
            if not main_public:
                self.log_debug_message("carrier=P1 reason=Base_absent pooled_global_is_asymmetric")
        else:
            models = tuple(model for model, _branch in branches)
            self.log_debug_message("logical_regions=0 patch=bypassed models=unchanged")

        return models + (
            positive_final,
            detailer_positive,
            negative_public,
            mask_p1,
            mask_p2,
            dict(config),
        )

    def run(
        self: Self,
        model_main: Any,
        main_positive: Any,
        person_1_positive: Any,
        person_2_positive: Any,
        negative: Any,
        latent: Any,
        use_couple_attention: bool = True,
        orientation: str = "horizontal",
        center: float = 0.5,
        transition: float = 0.03,
        mask_floor: float = 0.0,
        swap_person_positions: bool = False,
        binding_mode: str = "LEGACY",
        attn1_strength: float = 0.6,
        attn1_max_tokens: int = 1024,
        attn1_ambiguous_band: float = 0.15,
        attn1_apply_lowres_only: bool = True,
        attn1_hires_strength: float = 0.30,
        v2_ownership_strength: float = 0.80,
        v2_confidence_floor: float = 0.10,
        dual_sampling_model: Any = None,
        support_model_1: Any = None,
        support_model_2: Any = None,
        support_model_3: Any = None,
    ) -> Any:
        """Run the visible MASTER node and export its exact live parameters."""
        config = normalize_couple_config(
            {
                # The actual Person 2 encoder state is authoritative. When its
                # conditioning is empty, later phases must not re-enable its raw prompt.
                "person_2_enabled": bool(person_2_positive),
                "use_couple_attention": use_couple_attention,
                "orientation": orientation,
                "center": center,
                "transition": transition,
                "mask_floor": mask_floor,
                "swap_person_positions": swap_person_positions,
                "binding_mode": binding_mode,
                "attn1_strength": attn1_strength,
                "attn1_max_tokens": attn1_max_tokens,
                "attn1_ambiguous_band": attn1_ambiguous_band,
                "attn1_apply_lowres_only": attn1_apply_lowres_only,
                "attn1_hires_strength": attn1_hires_strength,
                "v2_ownership_strength": v2_ownership_strength,
                "v2_confidence_floor": v2_confidence_floor,
            }
        )
        return self._run_with_config(
            model_main,
            main_positive,
            person_1_positive,
            person_2_positive,
            negative,
            latent,
            config,
            dual_sampling_model,
            support_model_1,
            support_model_2,
            support_model_3,
        )


class CoupleConditioningCopyNode(CoupleConditioningNode):
    """Run the couple pipeline from a stored ``SAYA_COUPLE_CONFIG`` only (no widgets)."""

    @classmethod
    def INPUT_TYPES(cls: type[Self]) -> dict[str, Any]:
        """Return the ComfyUI input schema exposed by this node."""
        return {
            "required": {
                "model_main": ("MODEL",),
                "main_positive": ("CONDITIONING",),
                "person_1_positive": ("CONDITIONING",),
                "person_2_positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent": ("LATENT",),
                "couple_config": (COUPLE_CONFIG_TYPE,),
            },
            "optional": {
                "dual_sampling_model": ("MODEL",),
                "support_model_1": ("MODEL",),
                "support_model_2": ("MODEL",),
                "support_model_3": ("MODEL",),
            },
        }

    FUNCTION = "run_copy"

    def run_copy(
        self: Self,
        model_main: Any,
        main_positive: Any,
        person_1_positive: Any,
        person_2_positive: Any,
        negative: Any,
        latent: Any,
        couple_config: Any,
        dual_sampling_model: Any = None,
        support_model_1: Any = None,
        support_model_2: Any = None,
        support_model_3: Any = None,
    ) -> Any:
        """Execute exclusively from the MASTER configuration input."""
        return self._run_with_config(
            model_main,
            main_positive,
            person_1_positive,
            person_2_positive,
            negative,
            latent,
            couple_config,
            dual_sampling_model,
            support_model_1,
            support_model_2,
            support_model_3,
        )
