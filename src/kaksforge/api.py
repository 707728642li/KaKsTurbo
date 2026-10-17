"""Compatibility alias for kaksturbo.api."""
import importlib
import sys
sys.modules[__name__] = importlib.import_module("kaksturbo.api")
