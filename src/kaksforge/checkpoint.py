"""Compatibility alias for kaksturbo.checkpoint."""
import importlib
import sys
sys.modules[__name__] = importlib.import_module("kaksturbo.checkpoint")
