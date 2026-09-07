"""ComfyUI entry point for the Saya Couple / image-phase extension.

ComfyUI imports this package and reads ``NODE_CLASS_MAPPINGS``,
``NODE_DISPLAY_NAME_MAPPINGS`` and ``WEB_DIRECTORY``. Importing also registers
the image-phase HTTP routes used by the ``web/`` frontend.
"""

from __future__ import annotations

from .src.registry import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .src.routes.image_phases import register_image_phase_routes

register_image_phase_routes()

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
