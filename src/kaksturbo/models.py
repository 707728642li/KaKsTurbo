"""Versioned public configuration and result interfaces."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
import csv
import os
from pathlib import Path
from ._paths import absolute_path
from typing import Iterator
from .errors import ConfigurationError

from ._version import VERSION
MODEL_ORDER = ("JC", "F81", "K2P", "HKY", "TNEF", "TN", "K3P", "K3PUF",
               "TIMEF", "TIM", "TVMEF", "TVM", "SYM", "GTR")

def default_threads():
    """Respect a small Linux VM/container's available CPU affinity by default."""
    available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    return max(1, min(16, available))


@dataclass(frozen=True)
class RunConfig:
    mode: str = "robust"
    backend: str = "auto"
    threads: int = field(default_factory=default_threads)
    device: str = "auto"
    chunk_pairs: int = 1000
    input_format: str = "axt"
    pairs_manifest: str | None = None
    input_metadata: str | None = None
    method: str = "MA"
    genetic_code: int = 1
    batch_tasks: int = 512
    gpu_cache_buckets: int = 256
    request_capacity: int = 16384
    max_iterations: int = 500
    max_seconds: float = 600.0

    def __post_init__(self):
        import math
        for name, values in (("mode", {"reproduce", "robust"}),
                             ("backend", {"cpu", "gpu", "auto"}),
                             ("input_format", {"axt", "fasta"})):
            if not isinstance(getattr(self, name), str) or getattr(self, name) not in values:
                raise ConfigurationError(f"Unsupported {name}: {getattr(self, name)!r}")
        if self.method != "MA" or isinstance(self.genetic_code, bool) or not isinstance(self.genetic_code, int) or self.genetic_code != 1:
            raise ConfigurationError("This release supports MA and genetic code 1 only; all 14 models are retained")
        import os
        thread_limit = 32 if os.name == "nt" else 128
        for name, minimum, maximum in (("threads", 1, thread_limit), ("chunk_pairs", 1, 1000),
                                       ("batch_tasks", 32, 512), ("gpu_cache_buckets", 1, 256),
                                       ("request_capacity", 1, 1000000)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
                raise ConfigurationError(f"{name} must be an integer in [{minimum}, {maximum}]")
        if self.batch_tasks & (self.batch_tasks - 1):
            raise ConfigurationError("batch_tasks must be a power of two in [32, 512]")
        if self.max_iterations != 500:
            raise ConfigurationError("This release fixes the validated fitting budget at 500 iterations")
        if isinstance(self.max_seconds, bool) or not isinstance(self.max_seconds, (int, float)) or not math.isfinite(self.max_seconds) or self.max_seconds <= 0:
            raise ConfigurationError("max_seconds must be finite and positive")
        if not isinstance(self.device, str) or not (self.device == "auto" or self.device.startswith("GPU-") or
                                                   self.device in ("cuda:0", "cuda:1")):
            raise ConfigurationError("device must be auto, a physical GPU UUID, cuda:0 or cuda:1")
        if self.mode == "reproduce" and self.backend == "gpu":
            raise ConfigurationError("Historical reproduction is qualified on CPU only; choose --backend cpu or auto")
        if self.input_format == "fasta" and not self.pairs_manifest:
            raise ConfigurationError("Aligned FASTA requires an explicit pairs TSV manifest")
        if self.input_format == "axt" and self.pairs_manifest is not None:
            raise ConfigurationError("pairs_manifest is only used with aligned FASTA")
        if self.pairs_manifest is not None:
            object.__setattr__(self, "pairs_manifest", str(absolute_path(self.pairs_manifest)))
        if self.input_metadata is not None:
            if self.input_format != "axt":
                raise ConfigurationError("input_metadata is only used with prepared AXT")
            object.__setattr__(self, "input_metadata", str(absolute_path(self.input_metadata)))

    @property
    def code(self):
        return self.genetic_code

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class AlignedPair:
    pair_id: str
    left: str
    right: str


@dataclass(frozen=True)
class RunResult:
    directory: Path
    manifest: dict

    def iter_results(self) -> Iterator[dict]:
        """Stream the stable core schema, with numeric values and None for missing data."""
        from .results import iter_core_rows
        yield from iter_core_rows(self.directory / "result.tsv")

    def iter_native_results(self) -> Iterator[dict[str, str]]:
        """Read original mode-specific columns as strings, including literal NA."""
        with (self.directory / "native_result.tsv").open(encoding="utf-8", newline="") as handle:
            yield from csv.DictReader(handle, delimiter="\t")

    def __iter__(self):
        return self.iter_results()
