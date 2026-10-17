"""Lightweight progress events; observers cannot change numerical execution."""
from __future__ import annotations
from datetime import datetime, timezone
import math
import os
from pathlib import Path
from ._paths import absolute_path
import queue
import sys
import threading
import time
from .checkpoint import atomic_json, load_json, utc_now


def _counts(state):
    chunks = state.get("chunks", [])
    done = [c for c in chunks if c["status"] == "complete"]
    active = next((c["index"] + 1 for c in chunks if c["status"] == "running"), None)
    return dict(completed_pairs=sum(c["count"] for c in done),
                total_pairs=state.get("preflight", {}).get("pairs"),
                completed_chunks=len(done), total_chunks=len(chunks), current_chunk=active)


def _duration(seconds):
    if seconds is None or not math.isfinite(seconds):
        return "?"
    seconds = max(0, int(seconds))
    return f"{seconds // 3600:d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


class TerminalProgress:
    """A dependency-free, optional ``progress=`` callback writing to stderr.

    TTY streams get an ASCII bar on one line; redirected streams get occasional
    plain lines. Counts represent fully committed chunks, never guessed work
    inside a chunk. Elapsed time and throughput use the current session. Use as
    a context manager (or call close) to end a live line on every exit path.
    Rendering failures disable this observer and never change numerical work.
    """
    def __init__(self, stream=None, *, min_interval=None, enabled=True, tty=None, width=20):
        self.stream = sys.stderr if stream is None else stream
        self.enabled = enabled
        try:
            self.tty = bool(self.stream.isatty()) if tty is None else bool(tty)
        except (AttributeError, OSError, ValueError):
            self.tty = False
        self.min_interval = (0.25 if self.tty else 30.0) if min_interval is None else max(0.0, min_interval)
        self.width = max(1, int(width))
        self.rendering_errors = 0
        self._lock = threading.Lock()
        self._closed = False
        self._last_time = None
        self._last_phase = None
        self._last_width = 0
        self._line_open = False
        self._initial_pairs = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _columns(self):
        try:
            return max(1, os.get_terminal_size(self.stream.fileno()).columns)
        except (AttributeError, OSError, ValueError):
            return 80

    def _line(self, event):
        done, total = event.get("completed_pairs", 0), event.get("total_pairs")
        phase = event.get("phase", "unknown")
        complete = phase == "complete"
        if self._initial_pairs is None:
            self._initial_pairs = done
        elapsed = event.get("session_elapsed_seconds", event.get("elapsed_seconds", 0.0))
        new_pairs = event.get("session_completed_pairs", max(0, done - self._initial_pairs))
        rate = event.get("pairs_per_second")
        if rate is None:
            rate = new_pairs / elapsed if elapsed and new_pairs > 0 else 0.0
        rate_text = "?" if new_pairs == 0 and not complete else f"{rate:.2f}"
        percentage, fraction = "--%", 0.0
        if total:
            fraction = min(1.0, max(0.0, done / total))
            percentage = f"{100 * fraction:.1f}%"
            if not complete:
                if percentage == "100.0%":
                    # All fits may be committed while aggregation/sealing remains.
                    # Keep the exact pair counts; do not invent an overall 99.x%.
                    percentage = "<100%"
        if complete:
            eta = "0:00:00"
        elif phase in {"failed", "interrupted"}:
            eta = "--"
        elif not complete and total is not None and done >= total:
            eta = "finalizing"
        else:
            eta = _duration(event.get("eta_seconds"))
        counts = f"{done}/{total if total is not None else '?'}"
        elapsed_text = _duration(elapsed)
        details = (f"{counts} pairs {percentage} | elapsed {elapsed_text}"
                   f" | {rate_text} pairs/s | ETA {eta} | phase={phase}")
        width = self.width
        if self.tty:
            # Leave the last physical column unused to avoid terminal autowrap.
            limit = self._columns() - 1
            layouts = [details,
                f"phase={phase} {counts} {percentage} elapsed={elapsed_text} {rate_text} pairs/s ETA={eta}",
                f"phase={phase} {counts} {percentage} e={elapsed_text} {rate_text}/s ETA={eta}",
                f"{phase} {counts} {percentage} e={elapsed_text} {rate_text}/s ETA={eta}",
                f"{phase} {counts} {percentage} {rate_text}/s", f"{phase} {counts}", phase]
            details = next((line for line in layouts if len(line) <= limit), phase)
            width = max(0, min(width, limit - len(details) - 3))
        if not width:
            return details
        filled = int(width * fraction)
        if not complete:
            filled = min(width - 1, filled)
        bar = "#" * filled + "." * (width - filled)
        return f"[{bar}] {details}"

    def __call__(self, event):
        with self._lock:
            if self._closed or not self.enabled:
                return
            try:
                now = time.monotonic()
                phase = event.get("phase", "unknown")
                final = phase in {"complete", "failed", "interrupted"}
                changed = phase != self._last_phase
                if self._last_time is not None and not changed and not final and now - self._last_time < self.min_interval:
                    return
                line = self._line(event)
                if self.tty:
                    erase_width = min(self._last_width, self._columns() - 1)
                    self.stream.write("\r" + line + " " * max(0, erase_width - len(line)))
                    self._line_open = True
                    self._last_width = len(line)
                    if final:
                        self.stream.write("\n")
                        self._line_open = False
                else:
                    self.stream.write(line + "\n")
                self.stream.flush()
                self._last_time, self._last_phase = now, phase
            except Exception:
                self.rendering_errors += 1
                self.enabled = False

    def close(self):
        """Finish a live TTY line, including when the caller raises or interrupts."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._line_open:
                try:
                    self.stream.write("\n")
                    self.stream.flush()
                except Exception:
                    self.rendering_errors += 1
                self._line_open = False


class RunProgress:
    """Callbacks receive snapshots, including heartbeats, on an observer thread.

    Callback exceptions are counted in progress.json and do not abort fitting.
    No scientific parameter or result is provided to an observer for mutation.
    """
    def __init__(self, output, state, callback=None, interval=5.0):
        self.output = Path(output)
        self.callback = callback
        self.interval = interval
        self.started = time.monotonic()
        self.initial_pairs = _counts(state)["completed_pairs"]
        self.prior_elapsed = state.get("elapsed_seconds_completed_sessions", 0.0)
        self.snapshot = {}
        self.callback_errors = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._events = queue.Queue(maxsize=128)
        self._callback_thread = None
        if callback is not None:
            self._callback_thread = threading.Thread(target=self._observe, name="kaksturbo-observer", daemon=True)
            self._callback_thread.start()
        self.update(state)
        self._thread = threading.Thread(target=self._heartbeat, name="kaksturbo-progress", daemon=True)
        self._thread.start()

    def update(self, state):
        with self._lock:
            config = state.get("config", {})
            self.snapshot = dict(schema="kaksturbo.progress/1", phase=state["status"],
                                 directory=str(self.output), started_at=state.get("created_at"),
                                 requested_backend=config.get("backend", "auto"),
                                 backend="cpu" if config.get("backend", "auto") == "auto" else config["backend"],
                                 threads=config.get("threads"), device=config.get("device"), **_counts(state))
            self._deliver()

    def _deliver(self, notify=True):
        duration = time.monotonic() - self.started
        new_pairs = max(0, self.snapshot["completed_pairs"] - self.initial_pairs)
        total = self.snapshot.get("total_pairs")
        eta = None
        remaining = None if total is None else max(0, total - self.snapshot["completed_pairs"])
        if remaining and new_pairs > 0 and self.snapshot["phase"] == "running":
            eta = remaining * duration / new_pairs
        if self.snapshot["phase"] == "complete":
            eta = 0.0
        event = dict(self.snapshot, updated_at=utc_now(), session_elapsed_seconds=duration,
                     session_completed_pairs=new_pairs, pairs_per_second=new_pairs / duration if duration > 0 else 0.0,
                     elapsed_seconds=self.prior_elapsed + duration, eta_seconds=eta,
                     eta_scope="Remaining fitting estimated from completed pairs in this session; excludes final aggregation; pair lengths and models vary",
                     progress_scope="Only committed fitted pairs; aggregation and sealing must finish before phase=complete",
                     callback_errors=self.callback_errors)
        atomic_json(self.output / "progress.json", event)
        if self.callback is not None and notify:
            if self._events.full():
                self._events.get_nowait()
            self._events.put_nowait(event)

    def _observe(self):
        while True:
            event = self._events.get()
            if event is None:
                return
            try:
                self.callback(dict(event))
            except Exception:
                with self._lock:
                    self.callback_errors += 1

    def _heartbeat(self):
        while not self._stop.wait(self.interval):
            with self._lock:
                try:
                    self._deliver()
                except OSError:
                    # The main execution still owns checkpoint/error handling.
                    return

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval + 1.0)
        if self._callback_thread is not None:
            if self._events.full():
                self._events.get_nowait()
            self._events.put_nowait(None)
            # A stalled external UI must not hold up completed numerical work.
            self._callback_thread.join(timeout=1.0)
        with self._lock:
            self._deliver(notify=False)


def _active(directory):
    """Probe the existing advisory lock without changing its owner metadata."""
    lock = directory / ".kaksturbo.lock"
    if not lock.exists():
        lock = directory / ".kaksforge.lock"
    if not lock.exists():
        return False
    with lock.open("r+b") as handle:
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            return True
    return False


def status(output_dir):
    """Read current checkpoint progress, including a dead/interrupted process."""
    directory = absolute_path(output_dir)
    state = load_json(directory / "checkpoint.json")
    if state.get("schema") not in {"kaksturbo.checkpoint/1", "kaksturbo.checkpoint/2", "kaksforge.checkpoint/1", "kaksforge.checkpoint/2"}:
        from .errors import IntegrityError
        raise IntegrityError("Unrecognized KaKsTurbo checkpoint")
    event = load_json(directory / "progress.json") if (directory / "progress.json").is_file() else {}
    active = _active(directory)
    stored = state["status"]
    phase = "interrupted" if not active and stored in {"preflight", "running", "aggregating"} else stored
    age = None
    if event.get("updated_at"):
        age = max(0, (datetime.now(timezone.utc) - datetime.fromisoformat(event["updated_at"])).total_seconds())
    if active and age is not None and stored not in {"complete", "failed", "interrupted"}:
        for field in ("elapsed_seconds", "session_elapsed_seconds"):
            if event.get(field) is not None:
                event[field] += age
    return dict(event, schema="kaksturbo.status/1", directory=str(directory),
                phase=phase, checkpoint_phase=stored, process_active=active,
                seconds_since_update=age, resumable=phase != "complete", **_counts(state))
