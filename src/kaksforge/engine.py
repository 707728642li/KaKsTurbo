"""Compatibility alias for kaksturbo.engine."""
import importlib
import sys
sys.modules[__name__] = importlib.import_module("kaksturbo.engine")
