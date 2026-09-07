"""Filesystem, image checkpoint, and memory helpers for isolated image phases."""

from __future__ import annotations

import gc
import json
import logging
import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

PHASE_LABELS: dict[int, str] = {
    0: "idle",
    1: "base",
    2: "hires",
    3: "refiner",
    4: "predetail",
    5: "detailers",
    6: "final",
}

PHASE_STEMS: dict[int, str] = {
    1: "phase_1_base",
    2: "phase_2_hires",
    3: "phase_3_refiner",
    4: "phase_4_predetail",
    5: "phase_5_detailers",
    6: "phase_6_final",
}

DETAILERS: tuple[str, ...] = (
    "body",
    "hair",
    "face",
    "full_eyes",
    "eyes",
    "mouth",
    "lips",
    "breast",
    "nipples",
    "hands",
    "feet",
    "bra",
    "panties",
    "ass",
    "anus",
    "pussy",
    "penis",
    "nsfw",
)


@dataclass(frozen=True)
class CheckpointPaths:
    """Paths for one phase's candidate and validated artifacts."""

    directory: Path
    candidate_image: Path
    candidate_manifest: Path
    validated_image: Path
    validated_manifest: Path


def parse_phase(value: Any) -> int:
    """Parse a phase selector value such as ``2 — HIRES / USDU``."""
    text = str(value or "").strip()
    if not text:
        return 0
    head = text.split(maxsplit=1)[0]
    try:
        phase = int(head)
    except ValueError as error:
        raise ValueError(f"Phase invalide: {value!r}") from error
    if phase not in PHASE_LABELS:
        raise ValueError(f"Phase hors plage: {phase}")
    return phase


def normalize_detailer(value: Any) -> str:
    """Return a validated detailer slug or ``none``."""
    detailer = str(value or "none").strip().lower().replace(" ", "_")
    if detailer in {"", "none", "__shared__"}:
        return "none"
    if detailer not in DETAILERS:
        raise ValueError(f"Detailer inconnu: {value!r}")
    return detailer


def output_directory() -> Path:
    """Return ComfyUI's output directory with a safe local fallback."""
    try:
        import folder_paths

        return Path(folder_paths.get_output_directory()).resolve()
    except Exception:
        return (Path.cwd() / "output").resolve()


def checkpoint_directory(checkpoint_root: str) -> Path:
    """Resolve a user-visible relative checkpoint directory under output."""
    raw = str(checkpoint_root or "image/checkpoints").strip().replace("\\", "/")
    if not raw:
        raw = "image/checkpoints"
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("checkpoint_root doit rester relatif au dossier output.")
    base = output_directory()
    target = (base / relative).resolve()
    if target != base and base not in target.parents:
        raise ValueError("checkpoint_root sort du dossier output.")
    target.mkdir(parents=True, exist_ok=True)
    return target


def checkpoint_paths(phase: int, checkpoint_root: str) -> CheckpointPaths:
    """Build candidate and validated paths for one phase."""
    if phase not in PHASE_STEMS:
        raise ValueError(f"Aucun checkpoint pour la phase {phase}.")
    directory = checkpoint_directory(checkpoint_root)
    stem = PHASE_STEMS[phase]
    return CheckpointPaths(
        directory=directory,
        candidate_image=directory / f"{stem}.candidate.png",
        candidate_manifest=directory / f"{stem}.candidate.json",
        validated_image=directory / f"{stem}.png",
        validated_manifest=directory / f"{stem}.json",
    )


def source_phase_for(phase: int) -> int:
    """Return the validated phase that must feed ``phase``."""
    if phase <= 1:
        return 0
    return phase - 1


def _fsync_directory(directory: Path) -> None:
    """Best-effort fsync of a directory after atomic renames."""
    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Write bytes through a same-directory temporary file and ``os.replace``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    """Atomically write a UTF-8 JSON document."""
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    _atomic_write_bytes(path, payload)


def image_tensor_to_png_bytes(images: Any, metadata: dict[str, Any] | None = None) -> bytes:
    """Convert the first ComfyUI image tensor to encoded PNG bytes."""
    import io

    import numpy as np
    from PIL import Image, PngImagePlugin

    tensor = images
    if hasattr(tensor, "detach"):
        tensor = tensor.detach()
    if hasattr(tensor, "cpu"):
        tensor = tensor.cpu()
    if hasattr(tensor, "numpy"):
        tensor = tensor.numpy()
    array = np.asarray(tensor)
    if array.ndim == 4:
        if array.shape[0] < 1:
            raise ValueError("Le batch IMAGE est vide.")
        array = array[0]
    if array.ndim != 3 or array.shape[-1] not in (3, 4):
        raise ValueError(f"Format IMAGE inattendu: shape={array.shape!r}")
    if array.dtype != np.uint8:
        array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    mode = "RGBA" if array.shape[-1] == 4 else "RGB"
    image = Image.fromarray(array, mode=mode)
    png_info = None
    if metadata:
        png_info = PngImagePlugin.PngInfo()
        png_info.add_text("saya_phase_manifest", json.dumps(metadata, ensure_ascii=False))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", pnginfo=png_info, compress_level=4)
    return buffer.getvalue()


def load_image_tensor(path: Path) -> Any:
    """Load a PNG checkpoint into a ComfyUI IMAGE tensor."""
    import numpy as np
    import torch
    from PIL import Image, ImageOps

    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint image introuvable: {path}")
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        array = np.asarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(array)[None, ...]


def parse_json_widget(value: Any, expected: type[Any], name: str) -> Any:
    """Parse metadata widgets without ever aborting a completed image phase.

    These widgets only describe the models/VAE/samplers written to the manifest;
    they do not control execution.  ComfyUI extensions may deserialize multiline
    widgets as strings, lists, dictionaries, or already-decoded values.  A shape
    mismatch must therefore be normalized instead of throwing after an expensive
    render has already completed.
    """
    if isinstance(value, expected):
        return value

    if value is None:
        return expected()

    parsed: Any = value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return expected()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            # Preserve a plain non-JSON string as useful manifest metadata.
            parsed = text

    if expected is list:
        if parsed is None:
            return []
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, tuple):
            return list(parsed)
        # A shifted or extension-decoded widget must never crash FORCE STOP.
        return [parsed]

    if expected is dict:
        if parsed is None:
            return {}
        if isinstance(parsed, dict):
            return parsed
        return {"value": parsed}

    if isinstance(parsed, expected):
        return parsed
    raise ValueError(f"{name} doit être de type {expected.__name__}.")


def build_manifest(
    *,
    phase: int,
    detailer: str,
    source_path: str,
    result_path: Path,
    seed: int,
    positive_prompt: str,
    negative_prompt: str,
    models: list[Any],
    vaes: list[Any],
    samplers: dict[str, Any],
    status: str,
    transaction_id: str | None = None,
) -> dict[str, Any]:
    """Build a phase manifest with the fields required by the workflow contract."""
    return {
        "phase": phase,
        "phase_name": PHASE_LABELS[phase],
        "detailer": normalize_detailer(detailer),
        "transaction_uuid": transaction_id or str(uuid.uuid4()),
        "source_file": str(source_path or ""),
        "result_file": str(result_path),
        "seed": int(seed),
        "positive_prompt": str(positive_prompt or ""),
        "negative_prompt": str(negative_prompt or ""),
        "models_used": models,
        "vaes_used": vaes,
        "samplers": samplers,
        "date": datetime.now(UTC).isoformat(),
        "status": str(status),
    }


def save_candidate(
    *,
    images: Any,
    phase: int,
    detailer: str,
    checkpoint_root: str,
    source_path: str,
    seed: int,
    positive_prompt: str,
    negative_prompt: str,
    models: list[Any],
    vaes: list[Any],
    samplers: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    """Atomically save one candidate image and its manifest."""
    paths = checkpoint_paths(phase, checkpoint_root)
    transaction_id = str(uuid.uuid4())
    manifest = build_manifest(
        phase=phase,
        detailer=detailer,
        source_path=source_path,
        result_path=paths.candidate_image,
        seed=seed,
        positive_prompt=positive_prompt,
        negative_prompt=negative_prompt,
        models=models,
        vaes=vaes,
        samplers=samplers,
        status="candidate",
        transaction_id=transaction_id,
    )
    image_bytes = image_tensor_to_png_bytes(images, manifest)
    _atomic_write_bytes(paths.candidate_image, image_bytes)
    try:
        atomic_write_json(paths.candidate_manifest, manifest)
    except Exception:
        paths.candidate_image.unlink(missing_ok=True)
        raise
    return paths.candidate_image, manifest


def read_manifest(path: Path) -> dict[str, Any]:
    """Read and validate one JSON manifest."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Manifeste illisible: {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Manifeste invalide: {path}")
    return value


def _backup_file(path: Path, transaction_id: str) -> Path | None:
    """Create a rollback copy of one existing validated artifact."""
    if not path.exists():
        return None
    backup = path.with_name(f".{path.name}.rollback.{transaction_id}")
    shutil.copy2(path, backup)
    with backup.open("rb") as handle:
        os.fsync(handle.fileno())
    return backup


def promote_candidate(phase: int, detailer: str, checkpoint_root: str) -> dict[str, Any]:
    """Promote a complete candidate pair to validated files with rollback."""
    paths = checkpoint_paths(phase, checkpoint_root)
    candidate = read_manifest(paths.candidate_manifest)
    if not paths.candidate_image.is_file():
        raise FileNotFoundError(f"Image candidate introuvable: {paths.candidate_image}")
    if int(candidate.get("phase", -1)) != phase:
        raise ValueError("La candidate ne correspond pas à la phase demandée.")
    requested_detailer = normalize_detailer(detailer)
    candidate_detailer = normalize_detailer(candidate.get("detailer", "none"))
    if phase == 5 and requested_detailer != "none" and candidate_detailer != requested_detailer:
        raise ValueError(
            f"Candidate detailer={candidate_detailer}, sélection={requested_detailer}."
        )

    transaction_id = str(candidate.get("transaction_uuid") or uuid.uuid4())
    image_backup = _backup_file(paths.validated_image, transaction_id)
    manifest_backup = _backup_file(paths.validated_manifest, transaction_id)
    image_temp = paths.validated_image.with_name(f".{paths.validated_image.name}.{transaction_id}.tmp")
    manifest_temp = paths.validated_manifest.with_name(
        f".{paths.validated_manifest.name}.{transaction_id}.tmp"
    )
    validated = dict(candidate)
    validated.update(
        {
            "status": "validated",
            "validated_date": datetime.now(UTC).isoformat(),
            "result_file": str(paths.validated_image),
        }
    )
    try:
        shutil.copy2(paths.candidate_image, image_temp)
        with image_temp.open("rb") as handle:
            os.fsync(handle.fileno())
        manifest_temp.write_text(
            json.dumps(validated, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        with manifest_temp.open("rb") as handle:
            os.fsync(handle.fileno())

        os.replace(image_temp, paths.validated_image)
        os.replace(manifest_temp, paths.validated_manifest)
        _fsync_directory(paths.directory)
    except Exception:
        image_temp.unlink(missing_ok=True)
        manifest_temp.unlink(missing_ok=True)
        if image_backup and image_backup.exists():
            os.replace(image_backup, paths.validated_image)
        elif paths.validated_image.exists():
            paths.validated_image.unlink(missing_ok=True)
        if manifest_backup and manifest_backup.exists():
            os.replace(manifest_backup, paths.validated_manifest)
        elif paths.validated_manifest.exists():
            paths.validated_manifest.unlink(missing_ok=True)
        _fsync_directory(paths.directory)
        raise
    finally:
        if image_backup:
            image_backup.unlink(missing_ok=True)
        if manifest_backup:
            manifest_backup.unlink(missing_ok=True)

    paths.candidate_image.unlink(missing_ok=True)
    paths.candidate_manifest.unlink(missing_ok=True)
    return validated


def discard_candidate(phase: int, checkpoint_root: str) -> dict[str, Any]:
    """Delete only the unvalidated candidate for one phase."""
    paths = checkpoint_paths(phase, checkpoint_root)
    removed: list[str] = []
    for path in (paths.candidate_image, paths.candidate_manifest):
        if path.exists():
            path.unlink()
            removed.append(str(path))
    _fsync_directory(paths.directory)
    return {"removed": removed}


def load_validated_source(
    phase: int, detailer: str, checkpoint_root: str
) -> tuple[Any, dict[str, Any], Path]:
    """Load only the immediately preceding validated phase checkpoint.

    Phase 5 deliberately restarts from phase 4 every time.  The detailers run as
    one normal chain inside a single phase; an older phase-5 result must never be
    reused as an implicit input.
    """
    del detailer
    if phase <= 1:
        raise ValueError("La phase 1 n'utilise pas de checkpoint précédent.")
    source_phase = source_phase_for(phase)
    paths = checkpoint_paths(source_phase, checkpoint_root)
    manifest = read_manifest(paths.validated_manifest)
    if manifest.get("status") != "validated":
        raise ValueError(f"La phase {source_phase} n'est pas validée.")
    image = load_image_tensor(paths.validated_image)
    return image, manifest, paths.validated_image


def memory_snapshot() -> dict[str, Any]:
    """Return best-effort CPU/GPU memory information without forcing GPU init."""
    snapshot: dict[str, Any] = {}
    try:
        import torch

        if torch.cuda.is_available():
            snapshot.update(
                {
                    "device": str(torch.cuda.get_device_name(torch.cuda.current_device())),
                    "allocated_bytes": int(torch.cuda.memory_allocated()),
                    "reserved_bytes": int(torch.cuda.memory_reserved()),
                }
            )
        else:
            snapshot["device"] = "cpu"
    except Exception as error:
        snapshot["memory_error"] = str(error)
    return snapshot


def unload_everything() -> dict[str, Any]:
    """Unload ComfyUI models and clear Python/Torch caches."""
    before = memory_snapshot()
    errors: list[str] = []
    try:
        import comfy.model_management as model_management

        model_management.unload_all_models()
        model_management.soft_empty_cache()
    except Exception as error:
        errors.append(f"comfy.model_management: {error}")
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
    except Exception as error:
        errors.append(f"torch cache: {error}")
    collected = gc.collect()
    after = memory_snapshot()
    result = {
        "before": before,
        "after": after,
        "gc_collected": int(collected),
        "errors": errors,
    }
    if errors:
        LOGGER.warning("Image phase unload completed with warnings: %s", errors)
    return result



def emit_phase_complete(
    *,
    phase: int,
    next_phase: int,
    manifest: dict[str, Any],
    unload_report: dict[str, Any],
) -> bool:
    """Notify the ComfyUI frontend that one isolated prompt run is finished.

    The browser uses this event to wait two seconds, activate only the next
    phase, and submit a brand-new prompt.  Failure to notify is non-fatal: the
    validated checkpoint remains safely stored and the user can resume later.
    """
    payload = {
        "phase": int(phase),
        "next_phase": int(next_phase),
        "transaction_uuid": str(manifest.get("transaction_uuid", "")),
        "checkpoint": str(manifest.get("result_file", "")),
        "unload": unload_report,
    }
    try:
        from server import PromptServer

        PromptServer.instance.send_sync("saya_image_phase_complete", payload)
        return True
    except Exception as error:
        LOGGER.warning("Unable to emit image-phase completion event: %s", error)
        return False

def phase_status(phase: int, detailer: str, checkpoint_root: str) -> dict[str, Any]:
    """Return checkpoint and memory status for the controller UI."""
    normalized_detailer = normalize_detailer(detailer)
    result: dict[str, Any] = {
        "phase": phase,
        "phase_name": PHASE_LABELS.get(phase, "unknown"),
        "detailer": normalized_detailer,
        "memory": memory_snapshot(),
    }
    if phase in PHASE_STEMS:
        paths = checkpoint_paths(phase, checkpoint_root)
        result["candidate"] = str(paths.candidate_image) if paths.candidate_image.exists() else ""
        result["validated"] = str(paths.validated_image) if paths.validated_image.exists() else ""
    return result
