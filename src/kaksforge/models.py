"""Compatibility alias for kaksturbo.models."""
import importlib
import sys
sys.modules[__name__] = importlib.import_module("kaksturbo.models")
