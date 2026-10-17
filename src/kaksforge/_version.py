"""Compatibility alias for kaksturbo._version."""
import importlib
import sys
sys.modules[__name__] = importlib.import_module("kaksturbo._version")
