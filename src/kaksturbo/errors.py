"""Stable public errors and command-line exit codes."""
from __future__ import annotations


class KaKsTurboError(Exception):
    code = "runtime_error"
    exit_code = 4

    def __init__(self, message: str, *, details: dict | None = None):
        super().__init__(message)
        self.details = details or {}

    def as_dict(self) -> dict:
        return {"schema": "kaksturbo.error/1", "error": self.code,
                "message": str(self), "details": self.details}


class InputError(KaKsTurboError):
    code = "invalid_input"
    exit_code = 2


class ConfigurationError(KaKsTurboError):
    code = "invalid_configuration"
    exit_code = 2


class IntegrityError(KaKsTurboError):
    code = "integrity_failure"
    exit_code = 3


class RunLockedError(KaKsTurboError):
    code = "run_locked"
    exit_code = 3


class EngineError(KaKsTurboError):
    code = "engine_failure"
    exit_code = 4

# Compatibility name for callers of the previous public package.
KaKsForgeError = KaKsTurboError
