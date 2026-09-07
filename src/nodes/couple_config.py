"""Couple configuration: the shared config type, its normalizer and the two
optional "master / copy" nodes.

A ``SAYA_COUPLE_CONFIG`` is a plain ``dict`` produced by
:func:`normalize_couple_config`. It carries every knob the couple pipeline needs
(region geometry, binding mode, and the experimental attn1 / V2 parameters) so
later hires phases can rebuild the exact same couple without re-entering values.
"""

from __future__ import annotations

from typing import Any, Self

#: ComfyUI socket type used to pass a normalized couple config between nodes.
COUPLE_CONFIG_TYPE = "SAYA_COUPLE_CONFIG"

#: All binding modes selectable on the couple node, in UI order.
BINDING_MODES = ("LEGACY", "V0_PRE_SOFTMAX", "V1_ATTN1", "V1_1_HIRES", "V2_QUERY_OWNERSHIP")

#: Modes that run the joint Base/P1/P2 pre-softmax cross-attention path.
#: ``V1_ATTN1`` == ``V0_PRE_SOFTMAX`` for cross-attention, plus a soft attn1 bias.
PRE_SOFTMAX_BINDING_MODES = frozenset(
    {"V0_PRE_SOFTMAX", "V1_ATTN1", "V1_1_HIRES", "V2_QUERY_OWNERSHIP"}
)

#: Schema version stored in every normalized config.
CONFIG_VERSION = 5


def _clamped_float(raw: dict[str, Any], key: str, default: float, low: float, high: float) -> float:
    """Parse ``raw[key]`` as a float and clamp it to ``[low, high]`` (default on error)."""
    try:
        value = float(raw.get(key, default))
    except (TypeError, ValueError):
        value = default
    return min(high, max(low, value))


def _clamped_int(raw: dict[str, Any], key: str, default: int, low: int, high: int) -> int:
    """Parse ``raw[key]`` as ``int(round(float(...)))`` and clamp it (default on error)."""
    try:
        value = int(round(float(raw.get(key, default))))
    except (TypeError, ValueError):
        value = default
    return min(high, max(low, value))


def normalize_couple_config(config: Any = None) -> dict[str, Any]:
    """Return a complete, validated couple-configuration dictionary.

    Accepts ``None``, a partial dict or a non-dict; every field falls back to its
    default and is range-clamped. ``use_couple_attention`` is forced off whenever
    Person 2 is disabled (a disabled second person always means SOLO).
    """
    raw = config if isinstance(config, dict) else {}

    person_2_enabled = bool(raw.get("person_2_enabled", True))
    requested_attention = bool(
        raw.get("requested_use_couple_attention", raw.get("use_couple_attention", True))
    )

    orientation = str(raw.get("orientation", "horizontal"))
    if orientation not in {"horizontal", "vertical"}:
        orientation = "horizontal"

    binding_mode = str(raw.get("binding_mode", "LEGACY")).upper()
    if binding_mode not in BINDING_MODES:
        binding_mode = "LEGACY"

    return {
        "version": CONFIG_VERSION,
        "person_2_enabled": person_2_enabled,
        "use_couple_attention": requested_attention and person_2_enabled,
        "requested_use_couple_attention": requested_attention,
        "orientation": orientation,
        "center": _clamped_float(raw, "center", 0.5, 0.15, 0.85),
        "transition": _clamped_float(raw, "transition", 0.03, 0.01, 0.20),
        "mask_floor": _clamped_float(raw, "mask_floor", 0.0, 0.0, 0.20),
        "swap_person_positions": bool(raw.get("swap_person_positions", False)),
        "binding_mode": binding_mode,
        # --- V1 (attn1 / self-attention) experimental controls ---
        "attn1_strength": _clamped_float(raw, "attn1_strength", 0.6, 0.0, 1.5),
        "attn1_max_tokens": _clamped_int(raw, "attn1_max_tokens", 1024, 256, 16384),
        "attn1_ambiguous_band": _clamped_float(raw, "attn1_ambiguous_band", 0.15, 0.02, 0.45),
        "attn1_apply_lowres_only": bool(raw.get("attn1_apply_lowres_only", True)),
        "attn1_hires_strength": _clamped_float(raw, "attn1_hires_strength", 0.30, 0.0, 0.80),
        # --- V2 query-derived dynamic ownership (diagnostic-only in v21.7E) ---
        "v2_ownership_strength": _clamped_float(raw, "v2_ownership_strength", 0.80, 0.0, 1.0),
        "v2_confidence_floor": _clamped_float(raw, "v2_confidence_floor", 0.10, 0.0, 0.45),
    }


class SayaCoupleConfigMaster:
    """Build one ``SAYA_COUPLE_CONFIG`` from visible widgets (single source of truth)."""

    @classmethod
    def INPUT_TYPES(cls: type[Self]) -> dict[str, Any]:
        """Return the ComfyUI input schema exposed by this node."""
        return {
            "required": {
                "person_2_enabled": ("BOOLEAN", {"default": True}),
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
            },
        }

    RETURN_TYPES = (COUPLE_CONFIG_TYPE, "BOOLEAN")
    RETURN_NAMES = ("couple_config", "person_2_enabled")
    FUNCTION = "build"
    CATEGORY = "saya/rescue"

    def build(
        self: Self,
        person_2_enabled: bool,
        use_couple_attention: bool,
        orientation: str,
        center: float,
        transition: float,
        mask_floor: float,
        swap_person_positions: bool,
        binding_mode: str = "LEGACY",
        attn1_strength: float = 0.6,
        attn1_max_tokens: int = 1024,
        attn1_ambiguous_band: float = 0.15,
        attn1_apply_lowres_only: bool = True,
        attn1_hires_strength: float = 0.30,
        v2_ownership_strength: float = 0.80,
        v2_confidence_floor: float = 0.10,
    ) -> tuple[dict[str, Any], bool]:
        """Normalize the widget values into a couple config."""
        config = normalize_couple_config(
            {
                "person_2_enabled": person_2_enabled,
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
        return config, bool(config["person_2_enabled"])


class SayaCoupleConfigCopy:
    """Re-normalize an upstream master config (used to relay it into hires phases)."""

    @classmethod
    def INPUT_TYPES(cls: type[Self]) -> dict[str, Any]:
        """Return the ComfyUI input schema exposed by this node."""
        return {"required": {"master_config": (COUPLE_CONFIG_TYPE,)}}

    RETURN_TYPES = (COUPLE_CONFIG_TYPE, "BOOLEAN")
    RETURN_NAMES = ("couple_config", "person_2_enabled")
    FUNCTION = "copy"
    CATEGORY = "saya/rescue"

    def copy(self: Self, master_config: Any) -> tuple[dict[str, Any], bool]:
        """Return a fresh normalized copy of ``master_config`` and its Person 2 flag."""
        config = normalize_couple_config(master_config)
        return dict(config), bool(config["person_2_enabled"])
