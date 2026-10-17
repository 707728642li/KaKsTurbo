"""Compatibility alias for kaksturbo.quality."""
import importlib
import sys
sys.modules[__name__] = importlib.import_module("kaksturbo.quality")
