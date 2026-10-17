"""Compatibility alias for kaksturbo.compat_export."""
import importlib
import sys
sys.modules[__name__] = importlib.import_module("kaksturbo.compat_export")
