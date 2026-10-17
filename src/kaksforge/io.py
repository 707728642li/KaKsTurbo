"""Compatibility alias for kaksturbo.io."""
import importlib
import sys
sys.modules[__name__] = importlib.import_module("kaksturbo.io")
