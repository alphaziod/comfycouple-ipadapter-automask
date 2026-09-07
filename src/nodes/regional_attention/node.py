"""The ``Attention couple`` ComfyUI node.

:class:`RegionalAttentionNode` clones the incoming model and installs the Saya
regional-attention patches on it:

* ``attn2`` (cross-attention) replacements on every SDXL/SD1.5 attention block -
  or ``cross_attn`` monkey-patches for ANIMA models;
* optionally ``attn1`` (self-attention) replacements on a bounded set of low-res
  blocks for the ``V1_ATTN1`` / ``V1_1_HIRES`` / ``V2_QUERY_OWNERSHIP`` modes.

It also captures, on the node instance, all state the patches read at sample time
(joint contexts, ownership masks, per-model runtime caches). The couple node
(:mod:`..couple_conditioning`) drives it; users normally do not wire it directly.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any, Self

import comfy
import torch

from ..couple_config import PRE_SOFTMAX_BINDING_MODES
from .anima import AnimaAttentionMixin
from .cross_attention import CrossAttentionPatchMixin
from .replacement import install_attention_replacement
from .self_attention import (
    SD15_LOWRES_SELF_BLOCKS,
    SDXL_HIRES_SELF_BLOCKS,
    SDXL_HIRES_TARGET_SELF_BLOCKS,
    SDXL_LOWRES_SELF_BLOCKS,
    SelfAttentionBindingMixin,
    SelfBindingConfig,
)

#: Binding modes that add the low-res ``attn1`` self-attention patches.
_SELF_ATTENTION_MODES = {"V1_ATTN1", "V1_1_HIRES", "V2_QUERY_OWNERSHIP"}

#: Short log tag per binding mode.
_BINDING_TAGS = {
    "V2_QUERY_OWNERSHIP": "V2",
    "V1_1_HIRES": "V1.1",
    "V1_ATTN1": "V1",
    "V0_PRE_SOFTMAX": "V0",
}

#: Names of the per-model runtime structures moved with the model on offload.
_RUNTIME_STATE_NAMES = (
    "binding_contexts",
    "binding_masks",
    "negative_positive_masks",
    "negative_positive_conds",
    "detailer_positive_masks",
    "detailer_positive_conds",
)
_DENSE_CACHE_NAMES = ("_cross_bias_cache", "_self_bias_cache", "_gate_cache")


class RegionalAttentionNode(
    AnimaAttentionMixin, CrossAttentionPatchMixin, SelfAttentionBindingMixin
):
    """ComfyUI node that installs character-aware regional attention on a model."""

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("model", "positive", "negative")
    FUNCTION = "attention_couple"
    CATEGORY = "Saya/Conditioning"

    @classmethod
    def INPUT_TYPES(cls: type[Self]) -> dict[str, Any]:
        """Return the ComfyUI input schema exposed by this node."""
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "mode": (["Attention", "Latent"],),
            },
            "optional": {
                "saya_mode": (["GEN", "REFINER", "DETAILER", "AUTO"], {"default": "GEN"}),
                "saya_strength": (
                    "FLOAT",
                    {"default": -1.0, "min": -1.0, "max": 1.5, "step": 0.05},
                ),
                "saya_debug": ("BOOLEAN", {"default": False}),
            },
        }

    def attention_couple(
        self: Self,
        model: Any,
        positive: Any,
        negative: Any,
        mode: str,
        saya_mode: str = "GEN",
        saya_strength: float = -1.0,
        saya_debug: bool = False,
        detailer_positive: Any = None,
        binding_mode: str = "LEGACY",
        binding_contexts: Any = None,
        binding_masks: Any = None,
        binding_carrier: Any = None,
        self_binding: Any = None,
    ) -> Any:
        """Clone ``model`` and install the regional-attention patches.

        ``mode == "Latent"`` is a pass-through (no patching). Otherwise the node
        stores all sample-time state on ``self``, clones the model, installs the
        cross-attention patches (and self-attention patches for the relevant
        binding modes) and returns ``(patched_model, positive_carrier, negative)``.
        """
        if mode == "Latent":
            return (model, positive, negative)

        self.saya_mode = saya_mode
        self.saya_strength = saya_strength
        self.saya_debug = bool(saya_debug)
        self.binding_mode = str(binding_mode or "LEGACY").upper()
        self.binding_contexts = None
        self.binding_masks = None
        self.binding_carrier = None
        self._self_binding_raw = dict(self_binding) if isinstance(self_binding, dict) else {}
        self.self_binding_config = SelfBindingConfig.from_mapping(self_binding)

        # Per-patched-model runtime state. Caches are bounded and tied to one
        # binding revision so tensors from an old configuration cannot be reused.
        self._binding_revision = int(getattr(self, "_binding_revision", 0)) + 1
        self._cross_bias_cache: dict[Any, Any] = {}
        self._self_bias_cache: dict[Any, Any] = {}
        self._gate_cache: dict[Any, Any] = {}
        self._runtime_cache_limit = 16
        self._runtime_device: torch.device | None = None
        self._v2_debug_seen: set[Any] = set()
        self.negative_positive_masks: list[Any] = []
        self.negative_positive_conds: list[Any] = []
        self.detailer_positive_masks = None
        self.detailer_positive_conds = None
        self.detailer_positive_count = 0

        new_positive = copy.deepcopy(positive)
        new_negative = copy.deepcopy(negative)
        new_detailer_positive = (
            copy.deepcopy(detailer_positive) if detailer_positive else None
        )

        dtype = model.model.diffusion_model.dtype
        device = comfy.model_management.get_torch_device()
        try:
            self._runtime_device = (
                device if isinstance(device, torch.device) else torch.device(device)
            )
        except (TypeError, ValueError, RuntimeError):
            self._runtime_device = None

        if self.binding_mode in PRE_SOFTMAX_BINDING_MODES:
            self._capture_binding_state(binding_contexts, binding_masks, binding_carrier, device, dtype)

        def extract_bank(
            conditions: Any, *, strip_carrier_mask: bool
        ) -> tuple[list[Any], list[Any]]:
            """Return ``(masks, contexts)`` for one conditioning bank.

            A multi-entry bank yields per-entry normalized masks; a single-entry
            "carrier" bank yields ``([False], [context])`` and (when
            ``strip_carrier_mask``) has its mask metadata removed in place.
            """
            if len(conditions) != 1:
                mask_dtype = torch.bfloat16 if "float8" in str(dtype) else dtype
                mask_norm = torch.stack(
                    [
                        cond[1]["mask"].to(device, dtype=mask_dtype) * cond[1]["mask_strength"]
                        for cond in conditions
                    ]
                )
                mask_norm = mask_norm / mask_norm.sum(dim=0).clamp_min(1e-06)
                masks = [mask_norm[index] for index in range(mask_norm.shape[0])]
                contexts = [cond[0].to(device, dtype=dtype) for cond in conditions]
                if strip_carrier_mask:
                    conditions[0][1].pop("mask", None)
                    conditions[0][1].pop("mask_strength", None)
                return (masks, contexts)
            return ([False], [conditions[0][0].to(device, dtype=dtype)])

        for conditions in (new_negative, new_positive):
            bank_masks, bank_conds = extract_bank(conditions, strip_carrier_mask=True)
            self.negative_positive_masks.append(bank_masks)
            self.negative_positive_conds.append(bank_conds)
        self.conditioning_length = (len(new_negative), len(new_positive))

        if new_detailer_positive:
            detailer_masks, detailer_conds = extract_bank(
                new_detailer_positive, strip_carrier_mask=False
            )
            self.detailer_positive_masks = detailer_masks
            self.detailer_positive_conds = detailer_conds
            self.detailer_positive_count = len(new_detailer_positive)

        new_model = model.clone()
        dm = new_model.model.diffusion_model
        self.sdxl = hasattr(dm, "label_emb")
        has_unet_blocks = (
            hasattr(dm, "input_blocks")
            and hasattr(dm, "middle_block")
            and hasattr(dm, "output_blocks")
        )

        if not has_unet_blocks:
            self._install_non_unet(new_model, dm)
            return (new_model, self._resolve_positive_carrier(new_positive), [new_negative[0]])

        self._install_cross_attention_patches(new_model, dm)
        if self.binding_mode in _SELF_ATTENTION_MODES:
            self._install_self_attention_binding(new_model, dm)
        return (new_model, self._resolve_positive_carrier(new_positive), [new_negative[0]])

    # ------------------------------------------------------------------ #
    # Binding-state capture                                             #
    # ------------------------------------------------------------------ #

    def _capture_binding_state(
        self,
        binding_contexts: Any,
        binding_masks: Any,
        binding_carrier: Any,
        device: Any,
        dtype: Any,
    ) -> None:
        """Validate and store the joint Base/P1/P2 contexts + ownership masks."""
        if (
            not isinstance(binding_contexts, dict)
            or not isinstance(binding_masks, (list, tuple))
            or len(binding_masks) != 2
        ):
            raise ValueError(
                f"{self.binding_mode} requires Base/P1/P2 contexts and two ownership masks"
            )

        converted_contexts: dict[str, Any] = {}
        for name in ("base", "person_1", "person_2"):
            context = binding_contexts.get(name)
            if context is None:
                converted_contexts[name] = None
            elif not isinstance(context, torch.Tensor) or context.ndim != 3:
                raise ValueError(f"{self.binding_mode} context {name} must be a rank-3 tensor")
            else:
                converted_contexts[name] = context.to(device=device, dtype=dtype)
        if converted_contexts["person_1"] is None or converted_contexts["person_2"] is None:
            raise ValueError(
                f"{self.binding_mode} requires both Person 1 and Person 2 contexts"
            )

        self.binding_contexts = converted_contexts
        self.binding_masks = tuple(binding_masks)
        self.binding_carrier = binding_carrier if binding_carrier else None
        tag = _BINDING_TAGS.get(self.binding_mode, "V0")
        print(
            f"[ComfyCouple Binding {tag}] cross-attention: joint Base/P1/P2 "
            "pre-softmax tri-state P1/P2/Unknown binding enabled; v21.7E finite penalties"
        )

    # ------------------------------------------------------------------ #
    # Patch installation                                                #
    # ------------------------------------------------------------------ #

    def _install_non_unet(self, new_model: Any, dm: Any) -> None:
        """Handle ANIMA / unsupported architectures (no ComfyUI UNet blocks)."""
        has_anima_blocks = hasattr(dm, "blocks") and any(
            hasattr(block, "cross_attn") for block in getattr(dm, "blocks", [])
        )
        if has_anima_blocks:
            print(
                "[ComfyCouple ANIMA] transformer model detected: "
                "installing real cross_attn regional hooks"
            )
            self.install_anima_patch(new_model, dm)
        else:
            print(
                f"[ComfyCouple ANIMA] Unsupported architecture type={type(dm).__name__}; "
                "no UNet blocks and no dm.blocks[*].cross_attn found."
            )
        if self.binding_mode in _SELF_ATTENTION_MODES:
            tag = _BINDING_TAGS.get(self.binding_mode, "V1")
            print(
                f"[ComfyCouple {tag}] attn1 binding skipped: non-UNet architecture "
                "(V0 cross-attention still active)"
            )

    def _install_cross_attention_patches(self, new_model: Any, dm: Any) -> None:
        """Install one ``attn2`` replacement on every attention transformer block."""
        if not self.sdxl:
            block_plan = (
                [("input", block_id, 0) for block_id in (1, 2, 4, 5, 7, 8)]
                + [("middle", 0, 0)]
                + [("output", block_id, 0) for block_id in (3, 4, 5, 6, 7, 8, 9, 10, 11)]
            )
        else:
            block_plan = []
            for block_id in (4, 5, 7, 8):
                depth = 2 if block_id in (4, 5) else 10
                block_plan += [("input", block_id, index) for index in range(depth)]
            block_plan += [("middle", 0, index) for index in range(10)]
            for block_id in range(6):
                depth = 2 if block_id in (3, 4, 5) else 10
                block_plan += [("output", block_id, index) for index in range(depth)]

        for section, block_id, index in block_plan:
            attn2 = self._section_container(dm, section, block_id)[index].attn2
            key = (section, block_id) if not self.sdxl else (section, block_id, index)
            install_attention_replacement(
                new_model,
                self.create_cross_attention_patch(attn2, key),
                key,
            )

    @staticmethod
    def _section_container(dm: Any, section: str, block_id: int) -> Any:
        """Return the ``transformer_blocks`` list for one UNet section/block."""
        if section == "middle":
            return dm.middle_block[1].transformer_blocks
        if section == "input":
            return dm.input_blocks[block_id][1].transformer_blocks
        return dm.output_blocks[block_id][1].transformer_blocks

    # ------------------------------------------------------------------ #
    # Per-model runtime caches + device moves                           #
    # ------------------------------------------------------------------ #

    def _runtime_cache_get(self, cache_name: str, key: Any) -> Any:
        """Return one runtime-cache entry, or ``None``."""
        cache = getattr(self, cache_name, None)
        if not isinstance(cache, dict):
            return None
        return cache.get(key)

    def _runtime_cache_put(self, cache_name: str, key: Any, value: Any) -> Any:
        """Insert one entry and evict oldest entries past ``_runtime_cache_limit``."""
        cache = getattr(self, cache_name, None)
        if not isinstance(cache, dict):
            cache = {}
            setattr(self, cache_name, cache)
        cache[key] = value
        limit = max(1, int(getattr(self, "_runtime_cache_limit", 16)))
        while len(cache) > limit:
            cache.pop(next(iter(cache)), None)
        return value

    def _clear_runtime_caches(self) -> None:
        """Drop the dense device caches owned by this patched model."""
        for name in _DENSE_CACHE_NAMES:
            cache = getattr(self, name, None)
            if isinstance(cache, dict):
                cache.clear()

    @staticmethod
    def _move_runtime_value(value: Any, device: torch.device, memo: dict[int, Any]) -> Any:
        """Recursively move tensors inside dict/list/tuple structures, preserving aliases."""
        if isinstance(value, torch.Tensor):
            ident = id(value)
            if ident in memo:
                return memo[ident]
            moved = value.to(device=device)
            memo[ident] = moved
            return moved
        if isinstance(value, dict):
            return {
                key: RegionalAttentionNode._move_runtime_value(item, device, memo)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [
                RegionalAttentionNode._move_runtime_value(item, device, memo) for item in value
            ]
        if isinstance(value, tuple):
            return tuple(
                RegionalAttentionNode._move_runtime_value(item, device, memo) for item in value
            )
        return value

    def _regional_attention_to(self, device: Any) -> None:
        """Follow ComfyUI model offload/reload for Couple-owned tensors."""
        try:
            target = device if isinstance(device, torch.device) else torch.device(device)
        except (TypeError, ValueError, RuntimeError):
            return

        current = getattr(self, "_runtime_device", None)
        if isinstance(current, torch.device) and current == target:
            return

        # Dense caches are device/geometry specific: discard, never migrate.
        self._clear_runtime_caches()

        memo: dict[int, Any] = {}
        for name in _RUNTIME_STATE_NAMES:
            value = getattr(self, name, None)
            if value is not None:
                setattr(self, name, self._move_runtime_value(value, target, memo))
        self._runtime_device = target

    def _resolve_positive_carrier(self, new_positive: Any) -> Any:
        """Base-pure carrier for pre-softmax modes; else the first positive entry."""
        if self.binding_mode in PRE_SOFTMAX_BINDING_MODES and self.binding_carrier:
            return self.binding_carrier
        return [new_positive[0]]

    # ------------------------------------------------------------------ #
    # Self-attention (attn1) installation                               #
    # ------------------------------------------------------------------ #

    def _install_self_attention_binding(self, new_model: Any, dm: Any) -> None:
        """Install V1 low-res ``attn1`` binding and, for V1.1, the targeted high-res one."""
        base_config = getattr(self, "self_binding_config", None) or SelfBindingConfig()
        if self.binding_mode == "V2_QUERY_OWNERSHIP":
            print(
                "[ComfyCouple v21.7E] semantic V2 ownership disabled as authority; "
                "P1/P2/Unknown geometry shared by cross + low-res self; "
                "high-res attn1 remains disabled"
            )

        if not self.sdxl:
            config = replace(base_config, profile="V1", min_tokens=0)
            installed = self._install_self_blocks(new_model, dm, SD15_LOWRES_SELF_BLOCKS, config)
            print(
                f"[ComfyCouple V1] attn1 binding enabled: {installed} self-attention blocks patched "
                f"(strength={config.strength:.2f}, max_tokens={config.max_tokens}, "
                f"ambiguous_band={config.ambiguous_band:.2f}, arch=SD1.5)"
            )
            return

        # V1 low-res path: preserve v21.3 behaviour exactly.
        low_config = replace(base_config, profile="V1-lowres", min_tokens=0)
        low_specs = list(SDXL_LOWRES_SELF_BLOCKS)
        if self.binding_mode == "V1_ATTN1" and not base_config.lowres_only:
            # Backward-compatible manual full-hires experiment from v21.3.
            low_specs += list(SDXL_HIRES_SELF_BLOCKS)
        low_installed = self._install_self_blocks(new_model, dm, low_specs, low_config)

        if self.binding_mode != "V1_1_HIRES":
            print(
                f"[ComfyCouple V1] attn1 binding enabled: {low_installed} self-attention blocks patched "
                f"(strength={low_config.strength:.2f}, max_tokens={low_config.max_tokens}, "
                f"ambiguous_band={low_config.ambiguous_band:.2f}, lowres_only={low_config.lowres_only}, "
                "arch=SDXL)"
            )
            return

        # V1.1: decoder-side high-res only, on top of the low-res V1 pressure.
        raw = getattr(self, "_self_binding_raw", {})
        try:
            hires_strength = max(0.0, min(0.80, float(raw.get("attn1_hires_strength", 0.30))))
        except (TypeError, ValueError, AttributeError):
            hires_strength = 0.30
        hires_config = replace(
            base_config,
            strength=hires_strength,
            min_tokens=1025,
            max_tokens=4096,
            profile="V1.1-highres",
            lowres_only=False,
        )
        hires_installed = self._install_self_blocks(
            new_model, dm, SDXL_HIRES_TARGET_SELF_BLOCKS, hires_config, log_tag="V1.1"
        )
        print(
            f"[ComfyCouple V1.1] attn1 binding enabled: lowres={low_installed} + "
            f"targeted_hires={hires_installed} blocks "
            f"(low_strength={low_config.strength:.2f}, hires_strength={hires_strength:.2f}, "
            "hires_tokens=1025..4096, hires_blocks=decoder-only, arch=SDXL)"
        )

    def _install_self_blocks(
        self,
        new_model: Any,
        dm: Any,
        specs: Any,
        config: SelfBindingConfig,
        *,
        log_tag: str = "V1",
    ) -> int:
        """Install the self-attention patch for every block in ``specs``; return the count."""
        installed = 0
        for key, module in self.iter_self_attention_blocks(dm, specs):
            try:
                if install_attention_replacement(
                    new_model,
                    self.create_self_attention_patch(module, config),
                    key,
                    attn_name="attn1",
                ):
                    installed += 1
            except Exception as error:
                print(f"[ComfyCouple {log_tag}] could not patch attn1 {key}: {error}")
        return installed
