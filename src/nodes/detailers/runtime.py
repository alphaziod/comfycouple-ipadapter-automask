"""Impact Pack discovery and runtime dependency initialization."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

_CUSTOM_NODES_DIRECTORY = Path(__file__).resolve().parents[4]
for _folder_name in ("comfyui-impact-pack", "ComfyUI-Impact-Pack"):
    _modules_directory = _CUSTOM_NODES_DIRECTORY / _folder_name / "modules"
    if _modules_directory.is_dir() and str(_modules_directory) not in sys.path:
        sys.path.insert(0, str(_modules_directory))
_IMPACT_READY = False
_impact_pack = None


def find_loaded_impact_pack_module() -> Any:
    """Locate the Impact Pack module already loaded by ComfyUI."""
    module = sys.modules.get("impact.impact_pack")
    if module is not None:
        return module
    for candidate in tuple(sys.modules.values()):
        filename = getattr(candidate, "__file__", None)
        if not filename:
            continue
        normalized = str(filename).replace("\\\\", "/")
        if normalized.endswith("/modules/impact/impact_pack.py"):
            return candidate
    return None


def initialize_impact_pack_runtime() -> None:
    """Import Impact Pack once and re-export its ``core``/``utils``/``wildcards`` here.

    Idempotent (guarded by ``_IMPACT_READY``). Raises ``RuntimeError`` if Impact
    Pack is present but not fully loaded yet.
    """
    global _IMPACT_READY, _impact_pack
    if _IMPACT_READY:
        return
    module = find_loaded_impact_pack_module()
    if module is None:
        import importlib

        module = importlib.import_module("impact.impact_pack")
    required = ("core", "utils", "wildcards")
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise RuntimeError(
            "Impact Pack n'est pas encore complètement chargé : " + ", ".join(missing)
        )
    for name, value in vars(module).items():
        if name not in globals():
            globals()[name] = value
    _impact_pack = module
    _IMPACT_READY = True


def detailer_debug_enabled() -> bool:
    """Return whether crop-aware detailer debug logging is enabled."""
    return str(os.environ.get("SAYA_COUPLE_DEBUG", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
        "debug",
    }
