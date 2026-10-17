"""Compatibility alias for kaksturbo.progress."""
import importlib
import sys
sys.modules[__name__] = importlib.import_module("kaksturbo.progress")
