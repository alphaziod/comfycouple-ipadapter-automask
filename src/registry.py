"""ComfyUI node registry for the Saya Couple / image-phase extension.

``NODE_CLASS_MAPPINGS`` maps the internal node ids (which the saved workflows
reference) to their classes; ``NODE_DISPLAY_NAME_MAPPINGS`` gives each a menu
label. Neither the ids nor the labels should change without also updating every
saved workflow. The grouping is roughly: image-phase nodes, the couple /
regional-attention nodes, prompt-bundle helpers, encoding + routing helpers,
resolution / upscale calculators, and the crop-aware detailers.
"""

from __future__ import annotations

from typing import Any

from .nodes.couple_conditioning import CoupleConditioningCopyNode, CoupleConditioningNode
from .nodes.couple_phase_prompts import (
    SayaCouplePromptBundlePack,
    SayaCouplePromptBundleUnpack,
    SayaLatentShapeFromImage,
)
from .nodes.detailers.retry import SayaDetailerForEachAutoRetry
from .nodes.detailers.standard import SayaDetailerForEach
from .nodes.image_phases import (
    SayaImageGenerationReview,
    SayaImageModelHubSettings,
    SayaImagePhase1Stop,
    SayaImagePhase2Load,
    SayaImagePhase2Stop,
    SayaImagePhase3Load,
    SayaImagePhase3Stop,
    SayaImagePhase4Load,
    SayaImagePhase4Stop,
    SayaImagePhase5Load,
    SayaImagePhase5Stop,
    SayaImagePhase6Load,
    SayaImagePhase6Stop,
    SayaImagePhaseCheckpointLoad,
    SayaImagePhaseCheckpointStop,
    SayaImagePhaseController,
    SayaImageVAERouteSettings,
    SayaLazyCheckpointLoader,
)
from .nodes.regional_attention.node import RegionalAttentionNode
from .nodes.sampling_config import SayaKSamplerConfig
from .nodes.saya_resolution_scale import (
    SayaNear4KTargetCalculator,
    SayaResolutionScaleCalculator,
    SayaUpscalePresetModelLoader,
    SayaUpscaleTargetCalculator,
)
from .nodes.text_and_model_routing import DualClipTextEncoderNode, HiresModelRouterNode

NODE_CLASS_MAPPINGS: dict[str, type[Any]] = {
    "SayaImageGenerationReview": SayaImageGenerationReview,
    "SayaImagePhaseController": SayaImagePhaseController,
    "SayaImageModelHubSettings": SayaImageModelHubSettings,
    "SayaImageVAERouteSettings": SayaImageVAERouteSettings,
    "SayaLazyCheckpointLoader": SayaLazyCheckpointLoader,
    "SayaImagePhaseCheckpointLoad": SayaImagePhaseCheckpointLoad,
    "SayaImagePhaseCheckpointStop": SayaImagePhaseCheckpointStop,
    "SayaImagePhase1Stop": SayaImagePhase1Stop,
    "SayaImagePhase2Load": SayaImagePhase2Load,
    "SayaImagePhase2Stop": SayaImagePhase2Stop,
    "SayaImagePhase3Load": SayaImagePhase3Load,
    "SayaImagePhase3Stop": SayaImagePhase3Stop,
    "SayaImagePhase4Load": SayaImagePhase4Load,
    "SayaImagePhase4Stop": SayaImagePhase4Stop,
    "SayaImagePhase5Load": SayaImagePhase5Load,
    "SayaImagePhase5Stop": SayaImagePhase5Stop,
    "SayaImagePhase6Load": SayaImagePhase6Load,
    "SayaImagePhase6Stop": SayaImagePhase6Stop,
    "Attention couple": RegionalAttentionNode,
    "SayaComfyCouple": CoupleConditioningNode,
    "SayaComfyCoupleCopy": CoupleConditioningCopyNode,
    "SayaCouplePromptBundlePack": SayaCouplePromptBundlePack,
    "SayaCouplePromptBundleUnpack": SayaCouplePromptBundleUnpack,
    "SayaLatentShapeFromImage": SayaLatentShapeFromImage,
    "SayaDualCLIPTextEncode": DualClipTextEncoderNode,
    "SayaHiresTrioRouterSharedPrompt": HiresModelRouterNode,
    "SayaKSamplerConfig": SayaKSamplerConfig,
    "SayaResolutionScaleCalculator": SayaResolutionScaleCalculator,
    "SayaResolutionScaleCalculator13Ratios": SayaResolutionScaleCalculator,
    "SayaNear4KTargetCalculator": SayaNear4KTargetCalculator,
    "SayaUpscalePresetModelLoader": SayaUpscalePresetModelLoader,
    "SayaUpscaleTargetCalculator": SayaUpscaleTargetCalculator,
    "SayaDetailerForEach": SayaDetailerForEach,
    "SayaDetailerForEachAutoRetry": SayaDetailerForEachAutoRetry,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "SayaImageGenerationReview": "Saya Image Review · Continue / Restart New Seed",
    "SayaImagePhaseController": "Saya Image Auto Phase Controller",
    "SayaImageModelHubSettings": "Saya Image Model Hub · Settings Only",
    "SayaImageVAERouteSettings": "Saya Image VAE Routes · Settings Only",
    "SayaLazyCheckpointLoader": "Saya Lazy Checkpoint Loader",
    "SayaImagePhaseCheckpointLoad": "Saya Image Phase LOAD Previous Checkpoint",
    "SayaImagePhaseCheckpointStop": "Saya Image Phase AUTO STOP / UNLOAD",
    "SayaImagePhase1Stop": "AUTO PASS 1 · STOP / UNLOAD",
    "SayaImagePhase2Load": "AUTO PASS 2 · LOAD",
    "SayaImagePhase2Stop": "AUTO PASS 2 · STOP / UNLOAD",
    "SayaImagePhase3Load": "AUTO PASS 3 · LOAD",
    "SayaImagePhase3Stop": "AUTO PASS 3 · STOP / UNLOAD",
    "SayaImagePhase4Load": "AUTO PASS 4 · LOAD",
    "SayaImagePhase4Stop": "AUTO PASS 4 · STOP / UNLOAD",
    "SayaImagePhase5Load": "AUTO PASS 5 · LOAD",
    "SayaImagePhase5Stop": "AUTO PASS 5 · STOP / UNLOAD",
    "SayaImagePhase6Load": "AUTO PASS 6 · LOAD",
    "SayaImagePhase6Stop": "AUTO PASS 6 · STOP / UNLOAD",
    "Attention couple": "Load Attention couple V3.3 Saya Listen Gate",
    "SayaComfyCouple": "Saya Comfy Couple",
    "SayaComfyCoupleCopy": "Saya Comfy Couple · COPY · No Settings",
    "SayaCouplePromptBundlePack": "Saya Couple Prompt Bundle PACK",
    "SayaCouplePromptBundleUnpack": "Saya Couple Prompt Bundle UNPACK",
    "SayaLatentShapeFromImage": "Saya Latent Shape From Image",
    "SayaDualCLIPTextEncode": "Saya Dual CLIP Text Encode",
    "SayaHiresTrioRouterSharedPrompt": "Saya Hires Trio Router Shared Prompt RESCUE",
    "SayaKSamplerConfig": "Saya Sampling Config · beta45 compatible",
    "SayaResolutionScaleCalculator": "Saya Resolution Scale Calculator · Existing + Added Ratios",
    "SayaResolutionScaleCalculator13Ratios": "Saya Resolution Scale Calculator · Existing + Added Ratios",
    "SayaNear4KTargetCalculator": "Saya Dynamic Near-4K Target · Preserve Ratio",
    "SayaUpscalePresetModelLoader": "Saya Final Upscale · Preset + Model",
    "SayaUpscaleTargetCalculator": "Saya Dynamic Upscale Target · Preserve Ratio",
    "SayaDetailerForEach": "Saya Detailer For Each · Couple Crop",
    "SayaDetailerForEachAutoRetry": "Saya Detailer For Each AutoRetry · Couple Crop",
}
