"""Compatibility alias for kaksturbo.preparation."""
import importlib
import sys
sys.modules[__name__] = importlib.import_module("kaksturbo.preparation")
