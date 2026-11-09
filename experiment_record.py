"""NPZ-safe run metadata and independent, deterministic episode seed streams.

File fingerprints refer to working-tree bytes, including uncommitted changes.
This module imports no simulator and never changes global RNG state.
"""
from __future__ import annotations

import dataclasses
from datetime import datetime, timezone
import enum
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Any, Iterable, Mapping
import uuid

import numpy as np


# 新增流只能**追加在末尾**：stream_id 是 enumerate 下标，
# 插在中间会让后面所有流的种子改变，破坏既有批次的种子可复现性。
_STREAM_NAMES = ("env", "gyro", "reference", "dropout", "vx", "bias")
_EXCLUDED_SOURCE_DIRS = {
    ".git", ".venv", "venv", "__pycache__", "build", "dist", "logs",
    "traces", "doc", "docs", "tests", "node_modules", ".pytest_cache",
}


def _json_default(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported metadata value: {type(value).__name__}")


def _json(value: Any) -> str:
    return json.dumps(value, default=_json_default, ensure_ascii=False,
                      sort_keys=True, separators=(",", ":"))


def _file_digest(path: Path) -> tuple[str, str | None]:
    """Only return a digest if actual file bytes could be read."""
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return "ok", digest.hexdigest()
    except FileNotFoundError:
        return "missing", None
    except IsADirectoryError:
        return "not_a_file", None
    except OSError as exc:
        return f"unreadable:{type(exc).__name__}", None


def _git_metadata(project_dir: Path) -> dict[str, Any]:
    def git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(["git", "-C", str(project_dir), *args],
                              capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=10)
    result: dict[str, Any] = {"git_status": "unavailable"}
    try:
        revision = git("rev-parse", "HEAD")
        status = git("status", "--porcelain=v1", "--untracked-files=all")
    except (OSError, subprocess.TimeoutExpired) as exc:
        result["git_error"] = type(exc).__name__
        return result
    if revision.returncode == 0:
        result["git_revision"] = revision.stdout.strip()
    if status.returncode == 0:
        result["git_dirty"] = bool(status.stdout.strip())
        result["git_status"] = "ok" if revision.returncode == 0 else "no_revision"
        result["git_worktree_status"] = status.stdout.rstrip()
    else:
        result["git_error"] = status.stderr.strip()
    return result


def _python_files(directory: Path) -> Iterable[Path]:
    for current_dir, child_dirs, files in os.walk(directory):
        child_dirs[:] = sorted(name for name in child_dirs
                               if name not in _EXCLUDED_SOURCE_DIRS)
        for name in sorted(files):
            if name.endswith(".py"):
                yield Path(current_dir) / name


def _source_manifest(project_dir: Path,
                     source_paths: Iterable[str | os.PathLike[str]] | None
                     ) -> dict[str, dict[str, str]]:
    paths: set[Path] = set()
    selected_paths = [project_dir] if source_paths is None else source_paths
    for candidate in selected_paths:
        path = Path(candidate)
        if not path.is_absolute():
            path = project_dir / path
        path = path.resolve()
        if path.is_dir():
            paths.update(_python_files(path))
        else:
            # Preserve missing explicitly selected paths in the manifest.
            paths.add(path)
    manifest: dict[str, dict[str, str]] = {}
    for path in sorted(paths):
        try:
            key = path.relative_to(project_dir).as_posix()
        except ValueError:
            key = path.as_posix()
        status, digest = _file_digest(path)
        entry = {"status": status}
        if digest is not None:
            entry["sha256"] = digest
        manifest[key] = entry
    return manifest


def build_run_metadata(
    project_dir: str | os.PathLike[str],
    checkpoint: str | os.PathLike[str] | None,
    config: Mapping[str, Any],
    source_paths: Iterable[str | os.PathLike[str]] | None = None,
) -> dict[str, Any]:
    """Capture a run once before stepping; all returned values are scalars.

    ``source_paths`` accepts files and directories relative to ``project_dir``;
    directories recursively include Python source files. The default scans the
    project Python files excluding tests, docs, logs and build outputs. This is
    a code fingerprint, not a complete dependency/asset snapshot.

    Supply resolved experiment values in ``config`` (including actual gains),
    not only CLI overrides. Full nested content is saved as ``config_json``.
    Common analysis fields are also copied as scalars when provided. Absent
    configuration fields and missing-file hashes are never fabricated.
    """
    project_dir = Path(project_dir).resolve()
    now = datetime.now(timezone.utc)
    result: dict[str, Any] = {
        "metadata_schema_version": 1,
        "run_id": now.strftime("%Y%m%dT%H%M%S.%fZ_") + uuid.uuid4().hex,
        "run_started_utc": now.isoformat(),
        "project_dir": str(project_dir),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "config_json": _json(dict(config)),
    }
    for name in ("fixed_body", "pin_cmd", "task_id", "kp", "ki", "kd",
                 "feedback_mode"):
        if name in config and config[name] is not None:
            value = config[name]
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, enum.Enum):
                value = value.value
            result[name] = value if isinstance(value, (str, bool, int, float)) else _json(value)
    result.update(_git_metadata(project_dir))
    manifest = _source_manifest(project_dir, source_paths)
    manifest_json = _json(manifest)
    result["source_manifest_json"] = manifest_json
    result["source_sha256_json"] = _json({key: entry["sha256"]
        for key, entry in manifest.items() if "sha256" in entry})
    result["source_tree_sha256"] = hashlib.sha256(manifest_json.encode("utf-8")).hexdigest()
    result["source_file_count"] = len(manifest)
    result["source_snapshot_complete"] = bool(manifest) and all(
        entry["status"] == "ok" for entry in manifest.values())
    if checkpoint is None:
        result["checkpoint_status"] = "not_provided"
    else:
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = project_dir / checkpoint_path
        checkpoint_path = checkpoint_path.resolve()
        result["checkpoint"] = str(checkpoint_path)
        status, digest = _file_digest(checkpoint_path)
        result["checkpoint_status"] = status
        if digest is not None:
            result["checkpoint_sha256"] = digest
    return result


def episode_seed_metadata(base_seed: int, episode_index: int) -> dict[str, int]:
    """Derive streams using only the base seed and zero-based episode index.

    Calling episodes out of order or changing a filter/feedback mode cannot
    change the streams. Seeds fit NumPy's legacy seed API and PyTorch inputs.
    The stream IDs and deterministic collision handling define schema v1.
    """
    for name, value in (("base_seed", base_seed), ("episode_index", episode_index)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer")
        if not 0 <= value <= np.iinfo(np.int64).max:
            raise ValueError(f"{name} must be between 0 and 2**63 - 1")
    base_seed, episode_index = int(base_seed), int(episode_index)
    result = {"seed": base_seed, "seed_base": base_seed, "episode_index": episode_index}
    used: set[int] = set()
    for stream_id, name in enumerate(_STREAM_NAMES):
        sequence = np.random.SeedSequence(base_seed, spawn_key=(episode_index, stream_id))
        seed = int(sequence.generate_state(1, dtype=np.uint32)[0])
        # Guarantee distinct scalar seeds even in the unlikely collision case.
        while seed in used:
            seed = (seed + 1) % (2 ** 32)
        used.add(seed)
        result[f"seed_{name}"] = seed
    return result
