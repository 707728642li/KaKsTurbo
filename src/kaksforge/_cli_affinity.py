"""Compatibility alias for kaksturbo._cli_affinity."""
import importlib
import sys
sys.modules[__name__] = importlib.import_module("kaksturbo._cli_affinity")
