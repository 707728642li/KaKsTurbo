"""Atomic checkpoints and process-owned locks; kernel releases locks on death."""
from __future__ import annotations
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import uuid
from .errors import IntegrityError, RunLockedError


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_json(path: Path) -> dict:
    try:
        with Path(path).open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, ValueError) as exc:
        raise IntegrityError(f"Cannot read checkpoint or manifest: {path}") from exc
    if not isinstance(value, dict):
        raise IntegrityError(f"Expected an object: {path}")
    return value


def contained(directory: Path, relative: str) -> Path:
    path = (directory / relative).resolve()
    try:
        path.relative_to(directory.resolve())
    except ValueError as exc:
        raise IntegrityError(f"Checkpoint path escapes its run directory: {relative}") from exc
    return path


def verify_hashes(directory: Path, hashes: dict[str, str]):
    for relative, expected in hashes.items():
        path = contained(directory, relative)
        if not path.is_file() or file_sha256(path) != expected:
            raise IntegrityError(f"Saved artifact is missing or changed: {relative}")


@contextmanager
def run_lock(directory: Path):
    """Use an OS advisory byte/file lock, not a PID-based stale-lock guess."""
    path = directory / ".kaksturbo.lock"
    handle = path.open("a+b")
    acquired = False
    try:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RunLockedError(f"Another process holds this run: {directory}") from exc
        acquired = True
        # a+b always appends; write ownership through a separate seekable handle.
        owner = {"schema": "kaksturbo.run_lock/1", "pid": os.getpid(),
                 "host": socket.gethostname(), "acquired_at": utc_now()}
        with path.open("r+b") as metadata:
            metadata.seek(1)
            metadata.write(json.dumps(owner).encode("utf-8"))
            metadata.truncate()
            metadata.flush()
            os.fsync(metadata.fileno())
        yield
    finally:
        if acquired:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()
