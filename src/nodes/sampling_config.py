"""Sampler configuration whose COMBO output types match ComfyUI exactly."""

from __future__ import annotations

from typing import Any, Self


def _runtime_sampler_lists() -> tuple[list[str], list[str]]:
    """Read the lists after scheduler extensions (beta45/beta57) are loaded."""
    try:
        from comfy.samplers import KSampler  # type: ignore

        samplers = list(KSampler.SAMPLERS)
        schedulers = list(KSampler.SCHEDULERS)
    except (ImportError, AttributeError):
        # Unit-test fallback only. A real ComfyUI process provides KSampler.
        samplers = ["euler"]
        schedulers = ["simple", "linear_quadratic", "beta57", "beta45"]
    return samplers, schedulers


_SAMPLER_NAMES, _SCHEDULER_NAMES = _runtime_sampler_lists()


class SayaKSamplerConfig:
    """Expose sampling controls with exact runtime COMBO types.

    rgthree's config node can keep an older scheduler list. When another custom
    node adds beta45/beta57 to ``KSampler.SCHEDULERS``, ComfyUI rejects links
    between the two unequal COMBO types. This node sources both its input and
    output types from the live KSampler lists, so the types remain identical.
    """

    @classmethod
    def INPUT_TYPES(cls: type[Self]) -> dict[str, dict[str, Any]]:
        """Return the ComfyUI input schema, with COMBO types taken from the live KSampler."""
        return {
            "required": {
                "steps_total": (
                    "INT",
                    {"default": 4, "min": 1, "max": 10000, "step": 1},
                ),
                "refiner_step": (
                    "INT",
                    {"default": 2, "min": 0, "max": 10000, "step": 1},
                ),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.01,
                    },
                ),
                "sampler_name": (_SAMPLER_NAMES,),
                "scheduler": (_SCHEDULER_NAMES,),
            }
        }

    RETURN_TYPES = (
        "INT",
        "INT",
        "FLOAT",
        _SAMPLER_NAMES,
        _SCHEDULER_NAMES,
    )
    RETURN_NAMES = (
        "steps",
        "refiner_step",
        "cfg",
        "sampler_name",
        "scheduler",
    )
    FUNCTION = "configure"
    CATEGORY = "Saya/Sampling"

    def configure(
        self: Self,
        steps_total: int,
        refiner_step: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
    ) -> tuple[int, int, float, str, str]:
        """Return the coerced ``(steps, refiner_step, cfg, sampler_name, scheduler)``."""
        return (
            int(steps_total),
            int(refiner_step),
            float(cfg),
            str(sampler_name),
            str(scheduler),
        )
