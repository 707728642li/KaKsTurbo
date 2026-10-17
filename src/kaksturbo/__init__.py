"""KaKsTurbo public API. Importing this package does not initialize CUDA."""
from .models import AlignedPair, MODEL_ORDER, RunConfig, RunResult, VERSION
from .api import doctor, fit_file, fit_pairs, resume, status, validate
from .preparation import prepare_aligned, write_input_examples
from .quality import summarize_quality
from .compat_export import export_kaks_calculator
from .errors import KaKsTurboError, ConfigurationError, EngineError, InputError, IntegrityError, RunLockedError

__version__ = VERSION
__all__ = ["AlignedPair", "MODEL_ORDER", "RunConfig", "RunResult", "doctor", "fit_file", "fit_pairs",
           "resume", "status", "validate", "prepare_aligned", "write_input_examples", "summarize_quality", "export_kaks_calculator",
           "KaKsTurboError", "ConfigurationError", "EngineError", "InputError", "IntegrityError", "RunLockedError"]
