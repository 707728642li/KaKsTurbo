"""Standard-library ctypes binding for the native MA adapter.

Example:
    adapter = NativeAdapter()
    with adapter.prepare(pair_id, sequence_a, sequence_b) as pair:
        meta, values, z0, z1, weights, gc = pair.export(pattern_offset=0)
        cpu_result96, status4 = pair.evaluate(model_index=0, x=[.1, .3, 1, 1, 1, 1, 1])
        # GPU model results must contain 14 fitted results in canonical order.
        row, cpu_seconds, diagnostics = pair.finalize(gpu_results, gpu_statuses)

Arrays returned by export/evaluate are contiguous ctypes arrays. No NumPy,
CuPy, CUDA compiler, or CPU reference table is used by this module.
"""
from __future__ import annotations
from product_v1_platform import dll_directory, native_path
import ctypes as C
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
I32 = C.c_int32
U32 = C.c_uint32
D64 = C.c_double
I8 = C.c_int8
PI32 = C.POINTER(I32)
PU32 = C.POINTER(U32)
PD64 = C.POINTER(D64)
PI8 = C.POINTER(I8)


class NativeAdapterError(RuntimeError):
    def __init__(self, code: int, message: str):
        self.code = code
        super().__init__(f"Native adapter error {code}: {message}")


def _flat_buffer(value, ctype, count):
    if isinstance(value, C.Array):
        element_type = value._type_
        while isinstance(element_type, type) and issubclass(element_type, C.Array):
            element_type = element_type._type_
        if element_type is not ctype:
            raise TypeError(f"ctypes buffer element type must be {ctype.__name__}")
        if C.sizeof(value) != C.sizeof(ctype) * count:
            raise ValueError(f"Expected {count} values")
        return C.cast(value, C.POINTER(ctype)), value
    # Optional array-library interoperability, without importing a dependency.
    if hasattr(value, "ravel"):
        value = value.ravel()
    values = list(value)
    if len(values) != count:
        raise ValueError(f"Expected {count} values, got {len(values)}")
    array = (ctype * count)(*values)
    return C.cast(array, C.POINTER(ctype)), array


class NativeAdapter:
    def __init__(self, dll_path=None):
        self._directory_handles=[];self.closed=False
        try:self._initialize(dll_path)
        except BaseException:
            self.close();raise

    def _initialize(self,dll_path):
        if os.name not in ("nt", "posix"):
            raise RuntimeError("Windows or Linux is required")
        self.path = native_path(Path(dll_path or ROOT / "build/native-adapter-checked/kaks_adapter.dll"))
        library_dir = ROOT / "envs/cpu-reference/Library/bin"
        self._directory_handles = []
        if hasattr(os, "add_dll_directory"):
            for directory in (self.path.parent, library_dir):
                self._directory_handles.append(dll_directory(str(directory)))
        self.dll = C.CDLL(str(self.path))
        d = self.dll
        d.kaks_adapter_abi_version.argtypes = []
        d.kaks_adapter_abi_version.restype = U32
        if d.kaks_adapter_abi_version() != 1:
            raise RuntimeError("Unknown native adapter ABI")
        d.kaks_adapter_schema.argtypes = []
        d.kaks_adapter_schema.restype = C.c_char_p
        d.kaks_adapter_model_name.argtypes = [U32]
        d.kaks_adapter_model_name.restype = C.c_char_p
        d.kaks_prepare_pair.argtypes = [C.c_char_p, C.c_char_p, C.c_char_p,
                                       C.POINTER(C.c_void_p), C.c_char_p, U32]
        d.kaks_get_pair_sizes.argtypes = [C.c_void_p, PU32, PU32, PU32, C.c_char_p, U32]
        d.kaks_export_pair.argtypes = [C.c_void_p, I32, PI32, PD64, PI8, PI8,
                                      PD64, U32, PD64, C.c_char_p, U32]
        d.kaks_evaluate_pair.argtypes = [C.c_void_p, U32, PD64, PD64, PI32, C.c_char_p, U32]
        d.kaks_finalize_pair.argtypes = [C.c_void_p, PD64, U32, PI32, C.c_char_p,
                                        U32, PU32, PD64, C.c_char_p, U32]
        d.kaks_get_postprocess_diagnostics.argtypes = [C.c_void_p, PI32, C.c_char_p, U32]
        d.kaks_free_pair.argtypes = [C.c_void_p]
        d.kaks_free_pair.restype = None
        for name in ("kaks_prepare_pair", "kaks_get_pair_sizes", "kaks_export_pair",
                     "kaks_evaluate_pair", "kaks_finalize_pair", "kaks_get_postprocess_diagnostics"):
            getattr(d, name).restype = C.c_int

    @property
    def schema(self):
        return self.dll.kaks_adapter_schema().decode("ascii")

    @property
    def model_names(self):
        return [self.dll.kaks_adapter_model_name(i).decode("ascii") for i in range(14)]

    @staticmethod
    def _check(code, error):
        if code:
            raise NativeAdapterError(code, error.value.decode("utf-8", errors="replace"))

    def prepare(self, pair_id, sequence_a, sequence_b):
        if self.closed:raise RuntimeError('Native adapter binding is closed')
        def encoded(value):
            return value.encode("ascii") if isinstance(value, str) else bytes(value)
        error = C.create_string_buffer(1024)
        context = C.c_void_p()
        code = self.dll.kaks_prepare_pair(encoded(pair_id), encoded(sequence_a),
                                          encoded(sequence_b), C.byref(context), error, len(error))
        self._check(code, error)
        return NativePair(self, context)

    def close(self):
        """Call after all pair readers; release only DLL directory cookies."""
        handles,self._directory_handles=self._directory_handles,[];self.closed=True
        for directory in handles:directory.close()
    def __enter__(self):
        if self.closed:raise RuntimeError('Native adapter binding is closed')
        return self
    def __exit__(self,*_):self.close()
    def __del__(self):
        try:self.close()
        except Exception:pass


class NativePair:
    def __init__(self, adapter, context):
        self.adapter, self.context = adapter, context
        self.closed = False
        raw, cleaned, count = U32(), U32(), U32()
        error = C.create_string_buffer(1024)
        code = adapter.dll.kaks_get_pair_sizes(context, C.byref(raw), C.byref(cleaned),
                                              C.byref(count), error, len(error))
        adapter._check(code, error)
        self.raw_nt_length = raw.value
        self.cleaned_nt_length = cleaned.value
        self.pattern_count = count.value

    def _open(self):
        if self.closed:
            raise RuntimeError("Native pair context is already released")

    def export(self, pattern_offset=0):
        self._open()
        meta, values, gc = (I32 * 4)(), (D64 * 66)(), (D64 * 4)()
        z0, z1 = (I8 * self.pattern_count)(), (I8 * self.pattern_count)()
        weights = (D64 * self.pattern_count)()
        error = C.create_string_buffer(1024)
        code = self.adapter.dll.kaks_export_pair(self.context, pattern_offset, meta,
                                                 values, z0, z1, weights, self.pattern_count,
                                                 gc, error, len(error))
        self.adapter._check(code, error)
        return meta, values, z0, z1, weights, gc

    def evaluate(self, model_index, x):
        self._open()
        parameters, owner = _flat_buffer(x, D64, 7)
        results, status = (D64 * 96)(), (I32 * 4)()
        error = C.create_string_buffer(1024)
        code = self.adapter.dll.kaks_evaluate_pair(self.context, model_index, parameters,
                                                   results, status, error, len(error))
        self.adapter._check(code, error)
        return results, status

    def finalize(self, results, statuses):
        self._open()
        results_ptr, results_owner = _flat_buffer(results, D64, 14 * 96)
        status_ptr, status_owner = _flat_buffer(statuses, I32, 14 * 4)
        output, error = C.create_string_buffer(65536), C.create_string_buffer(1024)
        required, cpu_seconds = U32(), D64()
        code = self.adapter.dll.kaks_finalize_pair(self.context, results_ptr, 14, status_ptr,
                                                   output, len(output), C.byref(required),
                                                   C.byref(cpu_seconds), error, len(error))
        self.adapter._check(code, error)
        diagnostics = (I32 * 4)()
        code = self.adapter.dll.kaks_get_postprocess_diagnostics(self.context, diagnostics,
                                                               error, len(error))
        self.adapter._check(code, error)
        return output.value.decode("ascii"), cpu_seconds.value, {
            "negative_ming2_count": diagnostics[0],
            "iteration_limit_count": diagnostics[1],
            "gpu_eigen_warning_count": diagnostics[2],
            "cpu_MA_eigen_warning_count": diagnostics[3],
        }

    def close(self):
        if not self.closed:
            self.adapter.dll.kaks_free_pair(self.context)
            self.closed = True
            self.context = C.c_void_p()

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
