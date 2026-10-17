"""Compatibility alias for kaksturbo.report."""
import importlib
import sys
sys.modules[__name__] = importlib.import_module("kaksturbo.report")
