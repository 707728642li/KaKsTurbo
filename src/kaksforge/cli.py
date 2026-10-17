"""Compatibility alias for kaksturbo.cli."""
import importlib
import sys
sys.modules[__name__] = importlib.import_module("kaksturbo.cli")
