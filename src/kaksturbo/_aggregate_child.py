"""Pure serialization worker. No numerical engine, fitting or GPU imports."""
from __future__ import annotations
import importlib.abc
import json
import os
from pathlib import Path
import sys


NUMERICAL_MODULES = frozenset(("numpy", "scipy", "torch", "cupy", "kakscuda", "cpu_backend",
    "cpu_gradient", "gradient_optimizer", "shared_core", "shared_statistics", "product_v1_engine_fit"))


class NoNumericalLibraries(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in NUMERICAL_MODULES or fullname.startswith("publication_gpu"):
            raise ImportError("Output serialization forbids numerical imports: " + fullname)
        return None


def main():
    job = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    sys.meta_path.insert(0, NoNumericalLibraries())
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from kaksturbo._parallel_aggregate import build_fragment, pin_output_cpu
    from kaksturbo import api
    from kaksturbo.checkpoint import atomic_json
    affinity = pin_output_cpu(job["cpu"])
    result = build_fragment(api, job)
    result["CPU_affinity"] = affinity
    blas = {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
    imported = sorted(name for name in sys.modules if name.split(".")[0] in NUMERICAL_MODULES
                      or name.startswith("publication_gpu"))
    if any(value != "1" for value in blas.values()) or imported:
        raise RuntimeError("Output worker did not retain its no-numerical-import/BLAS1 scope")
    result["runtime_scope"] = dict(BLAS_environment=blas, numerical_python_modules_loaded=imported,
                                   numerical_import_blocker_installed=True)
    atomic_json(Path(job["destination"]) / "completed.json", result)


if __name__ == "__main__":
    main()
