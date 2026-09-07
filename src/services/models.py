"""Model discovery and loading services used by Saya nodes."""

from __future__ import annotations

from typing import Any
from urllib.parse import unquote


def list_registered_model_files(kind: str) -> list[str]:
    """Return the ComfyUI filename list for one model category (``[]`` on failure)."""
    try:
        import folder_paths

        return folder_paths.get_filename_list(kind)
    except Exception:
        return []


def build_model_choice_list(kind: str, extras: Any = ()) -> list[str]:
    """Return ``extras`` then the registered files, de-duplicated by string value.

    Order is preserved; ``None`` entries are dropped; an empty result becomes
    ``["none"]`` so the widget always has a choice.
    """
    candidates = list(extras) + list_registered_model_files(kind)
    seen: set[str] = set()
    choices: list[str] = []
    for candidate in candidates:
        if candidate is None:
            continue
        text = str(candidate)
        if text not in seen:
            seen.add(text)
            choices.append(text)
    return choices or ["none"]


def resolve_registered_model_path(kind: str, name: str) -> Any:
    """Resolve exact or URL-encoded ComfyUI model names safely.

    Some saved workflows contain literal ``%20`` sequences while other
    installations expose the same filename with normal spaces.  Try both
    forms, then fall back to a case-insensitive match against ComfyUI's
    registered filenames.
    """
    raw_name = str(name or "").strip()
    if not raw_name or raw_name == "none":
        return None

    candidates: list[str] = []
    for candidate in (unquote(raw_name), raw_name):
        candidate = candidate.replace("\\", "/")
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    try:
        import folder_paths

        for candidate in candidates:
            try:
                return folder_paths.get_full_path_or_raise(kind, candidate)
            except Exception:
                try:
                    path = folder_paths.get_full_path(kind, candidate)
                except Exception:
                    path = None
                if path:
                    return path

        wanted = {unquote(value).casefold() for value in candidates}
        wanted_basenames = {value.rsplit("/", 1)[-1] for value in wanted}
        for registered in folder_paths.get_filename_list(kind):
            registered_text = str(registered).replace("\\", "/")
            normalized = unquote(registered_text).casefold()
            if (
                normalized in wanted
                or normalized.rsplit("/", 1)[-1] in wanted_basenames
            ):
                try:
                    return folder_paths.get_full_path_or_raise(kind, registered_text)
                except Exception:
                    path = folder_paths.get_full_path(kind, registered_text)
                    if path:
                        return path
    except Exception:
        return None
    return None


def load_checkpoint_bundle(name: str) -> Any:
    """Load a checkpoint and return its model, CLIP encoder, and VAE."""
    path = resolve_registered_model_path("checkpoints", name)
    if not path:
        raise RuntimeError(f"Checkpoint introuvable: {name}")
    import comfy.sd
    import folder_paths

    model, clip, vae, _ = comfy.sd.load_checkpoint_guess_config(
        path,
        output_vae=True,
        output_clip=True,
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
    )
    return (model, clip, vae)


def load_vae_or_fallback(name: str, fallback: Any = None) -> Any:
    """Load a standalone VAE or return the supplied fallback when unavailable."""
    if not name or str(name) == "none":
        return fallback
    path = resolve_registered_model_path("vae", name)
    if not path:
        return fallback
    try:
        import comfy.sd
        import comfy.utils

        sd = comfy.utils.load_torch_file(path)
        return comfy.sd.VAE(sd=sd)
    except Exception:
        return fallback


def load_selected_model_family(
    kind: str, sdxl_name: str, anima_name: str, fallback: Any = None
) -> Any:
    """Load the model selected for one model family while preserving a fallback."""
    kind = str(kind or "none")
    if kind == "none":
        return fallback
    name = anima_name if kind == "anima" else sdxl_name
    try:
        return load_checkpoint_bundle(name)
    except Exception:
        if fallback is not None:
            return fallback
        raise
