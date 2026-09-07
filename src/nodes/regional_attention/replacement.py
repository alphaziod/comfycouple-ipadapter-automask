"""Installation and aggregation of ComfyUI attention-block replacements.

ComfyUI lets an extension replace the forward pass of individual attention
blocks through ``model_options["transformer_options"]["patches_replace"]``.
:class:`RegionalAttentionReplacement` is the callable stored there. It wraps the
Couple patch and, optionally, extra sigma-gated callbacks (used by the legacy
"Attention couple" node, not by the Couple pipeline itself).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

_LOG_PREFIX = "[ComfyCouple]"

#: Sentinels used when a callback does not restrict its sigma range.
_SIGMA_ALWAYS_HIGH = 999999999.9
_SIGMA_ALWAYS_LOW = -999999999.9


class RegionalAttentionReplacement:
    """Callable installed on one attention block; runs the Couple patch (+ callbacks)."""

    def __init__(self, couple_patch: Callable[..., torch.Tensor]) -> None:
        """Store the wrapped Couple patch and prepare the (usually empty) callback list."""
        self.couple_patch = couple_patch
        self.callback: list[Callable[..., torch.Tensor]] = []
        self.kwargs: list[dict[str, Any]] = []
        self.multigpu_kwargs: dict[Any, list[dict[str, Any]]] = {}

    def add(self, callback: Callable[..., torch.Tensor], **kwargs: Any) -> None:
        """Register an extra sigma-gated callback and expose its kwargs as attributes."""
        self.callback.append(callback)
        self.kwargs.append(kwargs)
        self.multigpu_kwargs = {}
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_multigpu_kwargs(self, device: Any) -> list[dict[str, Any]]:
        """Return device-local callback kwargs, falling back to the originals."""
        return self.multigpu_kwargs.get(device, self.kwargs)

    def __call__(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, extra_options: Any) -> torch.Tensor:
        """Run the Couple patch, then any sigma-gated callbacks, without host syncs.

        The Couple pipeline registers no callbacks, so the common path avoids the
        per-block ``sigma.cpu().item()`` sync that older versions did on every
        patched attention block.
        """
        dtype = q.dtype
        out = self.couple_patch(q, k, v, extra_options)

        if not self.callback:
            return out.to(dtype=dtype)

        sigma = (
            extra_options["sigmas"].detach().cpu()[0].item()
            if "sigmas" in extra_options
            else _SIGMA_ALWAYS_HIGH
        )
        device_kwargs = self.get_multigpu_kwargs(q.device)
        for index, callback in enumerate(self.callback):
            kwargs = device_kwargs[index]
            sigma_start = kwargs.get("sigma_start", _SIGMA_ALWAYS_HIGH)
            sigma_end = kwargs.get("sigma_end", _SIGMA_ALWAYS_LOW)
            if sigma_end <= sigma <= sigma_start:
                out = out + callback(out, q, k, v, extra_options, **kwargs)
        return out.to(dtype=dtype)

    def to(self, device: Any, *args: Any, **kwargs: Any) -> "RegionalAttentionReplacement":
        """Follow the model's device moves for Couple-owned runtime tensors.

        Delegates to ``owner._regional_attention_to`` (the node keeps the real
        state) and rebuilds device-local callback kwargs on GPU targets. Moving
        to CPU only drops the device-specific clones.
        """
        try:
            target = device if isinstance(device, torch.device) else torch.device(device)
        except (TypeError, ValueError, RuntimeError):
            return self

        mover = getattr(self.couple_patch.owner, "_regional_attention_to", None)
        if callable(mover):
            mover(target)

        if target.type == "cpu":
            self.multigpu_kwargs.clear()
            return self

        if target in self.multigpu_kwargs and len(self.multigpu_kwargs[target]) == len(self.kwargs):
            return self

        moved: list[dict[str, Any]] = []
        for kwargs_dict in self.kwargs:
            new_dict = dict(kwargs_dict)
            for key, value in list(new_dict.items()):
                if key == "ipadapter" and hasattr(value, "create_multigpu_clone"):
                    value.create_multigpu_clone(target)
                elif isinstance(value, torch.Tensor):
                    new_dict[key] = value.to(target)
            moved.append(new_dict)
        self.multigpu_kwargs[target] = moved
        return self


def install_attention_replacement(
    model: Any, patch: Callable[..., torch.Tensor], key: Any, attn_name: str = "attn2"
) -> bool:
    """Install one regional attention replacement on an already-cloned model.

    ``attn_name`` selects the replace table: ``"attn2"`` for cross-attention,
    ``"attn1"`` for the experimental self-attention path.

    The install is idempotent for Saya's own replacements (re-patching a model
    replaces the previous Saya entry). A slot already taken by *another*
    extension is left untouched and ``False`` is returned.
    """
    transformer_options = model.model_options["transformer_options"]
    replace_tables = transformer_options.setdefault("patches_replace", {})
    table = replace_tables.setdefault(attn_name, {})

    existing = table.get(key)
    if existing is not None and not isinstance(existing, RegionalAttentionReplacement):
        print(
            f"{_LOG_PREFIX} {attn_name} {key} already replaced by another patch; "
            "skipping to avoid a conflict"
        )
        return False

    table[key] = RegionalAttentionReplacement(patch)
    return True
