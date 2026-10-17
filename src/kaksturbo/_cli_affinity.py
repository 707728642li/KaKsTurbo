"""Bound the standalone CLI to its requested Windows CPU budget.

The Python API does not call this context: an embedding application's process
affinity must remain under that application's control. No numerical imports.
"""
from contextlib import contextmanager
import os

from .errors import ConfigurationError


@contextmanager
def temporary_budget(threads):
    if type(threads) is not int or threads < 1:
        raise ConfigurationError("CLI CPU budget must be a positive integer")
    if os.name != "nt":
        yield
        return
    from ._parallel_aggregate import windows_kernel, process_mask, windows_resources
    kernel = windows_kernel()
    original = process_mask(kernel)
    allowed, cores = windows_resources()
    if sum(1 << cpu for cpu in allowed) != original:
        raise ConfigurationError("CLI affinity changed while selecting its CPU budget")
    selected = []
    for core in cores:
        selected.extend(core[:max(0, threads - len(selected))])
        if len(selected) >= threads:
            break
    mask = sum(1 << cpu for cpu in selected)
    if not mask or mask & ~original:
        raise ConfigurationError("CLI CPU budget must remain within inherited affinity")
    if not kernel.SetProcessAffinityMask(kernel.GetCurrentProcess(), mask):
        raise ConfigurationError("Cannot apply the requested CLI CPU budget")
    try:
        if process_mask(kernel) != mask:
            raise ConfigurationError("CLI CPU affinity verification failed")
        yield
    finally:
        if (not kernel.SetProcessAffinityMask(kernel.GetCurrentProcess(), original)
                or process_mask(kernel) != original):
            raise ConfigurationError("Cannot restore the CLI's original CPU affinity")
