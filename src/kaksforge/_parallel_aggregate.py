"""Compatibility alias for kaksturbo._parallel_aggregate."""
import importlib
import sys
sys.modules[__name__] = importlib.import_module("kaksturbo._parallel_aggregate")
