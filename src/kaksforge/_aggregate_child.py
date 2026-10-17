"""Compatibility alias for kaksturbo._aggregate_child."""
import importlib
import sys
sys.modules[__name__] = importlib.import_module("kaksturbo._aggregate_child")
