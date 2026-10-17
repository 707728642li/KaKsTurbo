"""Compatibility alias for kaksturbo.results."""
import importlib
import sys
sys.modules[__name__] = importlib.import_module("kaksturbo.results")
