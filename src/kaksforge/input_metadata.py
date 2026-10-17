"""Compatibility alias for kaksturbo.input_metadata."""
import importlib
import sys
sys.modules[__name__] = importlib.import_module("kaksturbo.input_metadata")
