"""Conditioning encoding and bundle-normalization helpers."""

from __future__ import annotations

from typing import Any


def encode_text_conditioning(clip: Any, text: str) -> Any:
    """Encode prompt text into a ComfyUI conditioning structure."""
    text = str(text or "")
    tokens = clip.tokenize(text)
    try:
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return [[cond, {"pooled_output": pooled}]]
    except TypeError:
        return clip.encode_from_tokens_scheduled(tokens)


def extract_conditioning(bundle: Any, prefer: str = "sdxl") -> Any:
    """Extract conditioning from supported bundle shapes."""
    if isinstance(bundle, dict):
        for key in (prefer, "conditioning", "cond", "positive", "sdxl", "anima"):
            if key in bundle and bundle[key] is not None:
                return bundle[key]
        return None
    if isinstance(bundle, (list, tuple)):
        return bundle
    return bundle
