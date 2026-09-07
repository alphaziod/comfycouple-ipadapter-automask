"""HTTP endpoints used by the image-phase controller frontend."""

from __future__ import annotations

import logging
from typing import Any

from ..services.image_phases import (
    discard_candidate,
    normalize_detailer,
    parse_phase,
    phase_status,
    promote_candidate,
    unload_everything,
)

LOGGER = logging.getLogger(__name__)


def register_image_phase_routes() -> None:
    """Register validation, redo, status, and emergency-unload endpoints."""
    from aiohttp import web
    from server import PromptServer

    routes = PromptServer.instance.routes

    @routes.post("/saya/image-phases/validate")
    async def validate_phase(request: Any) -> Any:
        """POST: promote the current phase candidate to a validated checkpoint."""
        try:
            payload = await request.json()
            phase = parse_phase(payload.get("phase"))
            detailer = normalize_detailer(payload.get("detailer", "none"))
            root = str(payload.get("checkpoint_root", "image/checkpoints"))
            manifest = promote_candidate(phase, detailer, root)
            return web.json_response({"ok": True, "manifest": manifest})
        except (ValueError, OSError) as error:
            LOGGER.exception("Image phase validation failed")
            return web.json_response({"ok": False, "error": str(error)}, status=400)

    @routes.post("/saya/image-phases/redo")
    async def redo_phase(request: Any) -> Any:
        """POST: discard the current phase candidate so it can be re-generated."""
        try:
            payload = await request.json()
            phase = parse_phase(payload.get("phase"))
            root = str(payload.get("checkpoint_root", "image/checkpoints"))
            return web.json_response({"ok": True, **discard_candidate(phase, root)})
        except (ValueError, OSError) as error:
            return web.json_response({"ok": False, "error": str(error)}, status=400)

    @routes.post("/saya/image-phases/unload")
    async def unload_phase(request: Any) -> Any:
        """POST: free every loaded model and cache (emergency VRAM/RAM unload)."""
        del request
        return web.json_response({"ok": True, "unload": unload_everything()})

    @routes.post("/saya/image-phases/status")
    async def status_phase(request: Any) -> Any:
        """POST: return the on-disk + memory status for one phase."""
        try:
            payload = await request.json()
            phase = parse_phase(payload.get("phase"))
            detailer = normalize_detailer(payload.get("detailer", "none"))
            root = str(payload.get("checkpoint_root", "image/checkpoints"))
            return web.json_response({"ok": True, **phase_status(phase, detailer, root)})
        except (ValueError, OSError) as error:
            return web.json_response({"ok": False, "error": str(error)}, status=400)
