"""Compatibility alias for kaksturbo.runtime."""
import importlib
import sys
sys.modules[__name__] = importlib.import_module("kaksturbo.runtime")
