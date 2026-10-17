"""Minimal native Windows NVRTC / CUDA Driver API bindings (no GPU framework).

NVRTC compilation uses the project runtime; device contexts are chosen by UUID.
All kernels synchronize explicitly. No system settings are changed.
"""
from __future__ import annotations
from product_v1_platform import cuda_driver, dll_directory
import ctypes as C
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_DLL_DIRS = []
_LOADED_LIBRARIES = []


def _dll_paths():
    for relative in ('envs/cpu-reference/Library/bin', 'envs/cuda-runtime/Library/bin'):
        path = ROOT / relative
        if path.is_dir():
            _DLL_DIRS.append(dll_directory(str(path)))


def _bind(lib, name, argtypes):
    fn = getattr(lib, name)
    fn.argtypes = argtypes
    fn.restype = C.c_int
    return fn


def compile_ptx(source: str, options=None):
    """Compile without initializing CUDA or creating a GPU context."""
    raise RuntimeError('KaKsForge release inference requires installed cached PTX; runtime compilation is disabled')
    _dll_paths()
    # NVRTC loads its builtins with an internal LoadLibrary call that does not
    # inherit Python's DLL search flags; preload the exact project copy.
    _LOADED_LIBRARIES.append(C.CDLL(str(ROOT / 'envs/cuda-runtime/Library/bin/nvrtc-builtins64_129.dll')))
    lib = C.CDLL(str(ROOT / 'envs/cuda-runtime/Library/bin/nvrtc64_120_0.dll'))
    ptr = C.c_void_p
    create = _bind(lib, 'nvrtcCreateProgram', [C.POINTER(ptr), C.c_char_p, C.c_char_p, C.c_int, ptr, ptr])
    compile_program = _bind(lib, 'nvrtcCompileProgram', [ptr, C.c_int, C.POINTER(C.c_char_p)])
    log_size = _bind(lib, 'nvrtcGetProgramLogSize', [ptr, C.POINTER(C.c_size_t)])
    get_log = _bind(lib, 'nvrtcGetProgramLog', [ptr, ptr])
    ptx_size = _bind(lib, 'nvrtcGetPTXSize', [ptr, C.POINTER(C.c_size_t)])
    get_ptx = _bind(lib, 'nvrtcGetPTX', [ptr, ptr])
    destroy = _bind(lib, 'nvrtcDestroyProgram', [C.POINTER(ptr)])
    opts = options or ['--std=c++17', '--gpu-architecture=compute_86', '--fmad=false']
    encoded = (C.c_char_p * len(opts))(*(o.encode('utf-8') for o in opts))
    program = ptr()
    code = create(C.byref(program), source.encode('utf-8'), b'gy94_fp64.cu', 0, None, None)
    if code:
        raise RuntimeError(f'nvrtcCreateProgram failed: {code}')
    try:
        code = compile_program(program, len(opts), encoded)
        size = C.c_size_t()
        if log_size(program, C.byref(size)):
            raise RuntimeError('nvrtcGetProgramLogSize failed')
        log = C.create_string_buffer(max(1, size.value))
        get_log(program, log)
        message = log.value.decode('utf-8', errors='replace')
        if code:
            raise RuntimeError(f'NVRTC compilation failed ({code}):\n{message}')
        if ptx_size(program, C.byref(size)):
            raise RuntimeError('nvrtcGetPTXSize failed')
        data = C.create_string_buffer(size.value)
        if get_ptx(program, data):
            raise RuntimeError('nvrtcGetPTX failed')
        return data.raw, message
    finally:
        destroy(C.byref(program))


class DeviceBuffer:
    def __init__(self, pointer, size):
        self.pointer = int(pointer)
        self.size = int(size)


class CudaContext:
    def __init__(self, uuid: str, reserve_bytes=2*1024**3):
        _dll_paths()
        self.lib = cuda_driver()
        self.context = C.c_void_p()
        self.buffers = []
        self.modules = []
        self.reserve_bytes = reserve_bytes
        self.allocated_bytes = 0
        self._check(_bind(self.lib, 'cuInit', [C.c_uint])(0), 'cuInit')
        count = C.c_int()
        self._check(_bind(self.lib, 'cuDeviceGetCount', [C.POINTER(C.c_int)])(C.byref(count)), 'cuDeviceGetCount')
        get_device = _bind(self.lib, 'cuDeviceGet', [C.POINTER(C.c_int), C.c_int])
        get_uuid = _bind(self.lib, 'cuDeviceGetUuid_v2', [C.c_void_p, C.c_int])
        matched = None
        for ordinal in range(count.value):
            device = C.c_int()
            self._check(get_device(C.byref(device), ordinal), 'cuDeviceGet')
            raw = (C.c_ubyte*16)()
            self._check(get_uuid(raw, device), 'cuDeviceGetUuid_v2')
            hexed = bytes(raw).hex()
            found = 'GPU-' + '-'.join((hexed[:8], hexed[8:12], hexed[12:16], hexed[16:20], hexed[20:]))
            if found.lower() == uuid.lower():
                matched = device
                self.ordinal = ordinal
                self.uuid = found
                break
        if matched is None:
            raise RuntimeError(f'Selected GPU UUID not found by CUDA: {uuid}')
        name = C.create_string_buffer(256)
        self._check(_bind(self.lib, 'cuDeviceGetName', [C.c_void_p, C.c_int, C.c_int])(name, 256, matched), 'cuDeviceGetName')
        self.name = name.value.decode()
        self._check(_bind(self.lib, 'cuCtxCreate_v2', [C.POINTER(C.c_void_p), C.c_uint, C.c_int])(C.byref(self.context), 0, matched), 'cuCtxCreate_v2')

    def _check(self, code, name):
        if code:
            error = C.c_char_p()
            fn = _bind(self.lib, 'cuGetErrorString', [C.c_int, C.POINTER(C.c_char_p)])
            fn(code, C.byref(error))
            raise RuntimeError(f'{name} failed ({code}): {error.value.decode() if error.value else "unknown"}')

    def memory_info(self):
        free, total = C.c_size_t(), C.c_size_t()
        self._check(_bind(self.lib, 'cuMemGetInfo_v2', [C.POINTER(C.c_size_t), C.POINTER(C.c_size_t)])(C.byref(free), C.byref(total)), 'cuMemGetInfo_v2')
        return free.value, total.value

    def allocate(self, size):
        size = int(size)
        if size <= 0:
            raise ValueError('Device allocation must be positive')
        free, total = self.memory_info()
        if size > free - self.reserve_bytes:
            raise RuntimeError(f'Allocation {size} exceeds available memory with 2 GiB reserve ({free}/{total})')
        pointer = C.c_uint64()
        self._check(_bind(self.lib, 'cuMemAlloc_v2', [C.POINTER(C.c_uint64), C.c_size_t])(C.byref(pointer), size), 'cuMemAlloc_v2')
        buf = DeviceBuffer(pointer.value, size)
        self.buffers.append(buf)
        self.allocated_bytes += size
        return buf

    def upload(self, data):
        raw = bytes(data)
        buf = self.allocate(len(raw))
        self.write(buf,raw)
        return buf

    def write(self, buf, data):
        raw = bytes(data)
        if len(raw)>buf.size:
            raise ValueError('Copy would exceed device buffer')
        host = C.create_string_buffer(raw, len(raw))
        self._check(_bind(self.lib, 'cuMemcpyHtoD_v2', [C.c_uint64, C.c_void_p, C.c_size_t])(buf.pointer, host, len(raw)), 'cuMemcpyHtoD_v2')

    def download(self, buf, size=None):
        size = buf.size if size is None else int(size)
        if not 0 <= size <= buf.size:
            raise ValueError('Copy would exceed device buffer')
        host = C.create_string_buffer(size)
        self._check(_bind(self.lib, 'cuMemcpyDtoH_v2', [C.c_void_p, C.c_uint64, C.c_size_t])(host, buf.pointer, size), 'cuMemcpyDtoH_v2')
        return host.raw

    def load_module(self, ptx):
        module = C.c_void_p()
        data = C.create_string_buffer(ptx)
        self._check(_bind(self.lib, 'cuModuleLoadData', [C.POINTER(C.c_void_p), C.c_void_p])(C.byref(module), data), 'cuModuleLoadData')
        self.modules.append(module)
        return module

    def function_info(self, module, function_name):
        function = C.c_void_p()
        self._check(_bind(self.lib, 'cuModuleGetFunction', [C.POINTER(C.c_void_p), C.c_void_p, C.c_char_p])(C.byref(function), module, function_name.encode()), 'cuModuleGetFunction')
        get = _bind(self.lib, 'cuFuncGetAttribute', [C.POINTER(C.c_int), C.c_int, C.c_void_p])
        result = {}
        for name, code in [('max_threads_per_block',0),('shared_bytes',1),('constant_bytes',2),
                           ('local_bytes_per_thread',3),('registers_per_thread',4),('ptx_version',5),('binary_version',6)]:
            value = C.c_int()
            self._check(get(C.byref(value),code,function),'cuFuncGetAttribute')
            result[name]=value.value
        stack = C.c_size_t()
        self._check(_bind(self.lib,'cuCtxGetLimit',[C.POINTER(C.c_size_t),C.c_int])(C.byref(stack),0),'cuCtxGetLimit')
        result['context_stack_limit_bytes']=stack.value
        return result

    def launch(self, module, function_name, args, grid, block=32, shared_bytes=0):
        function = C.c_void_p()
        self._check(_bind(self.lib, 'cuModuleGetFunction', [C.POINTER(C.c_void_p), C.c_void_p, C.c_char_p])(C.byref(function), module, function_name.encode()), 'cuModuleGetFunction')
        if shared_bytes > 48*1024:
            self._check(_bind(self.lib,'cuFuncSetAttribute',[C.c_void_p,C.c_int,C.c_int])(function,8,shared_bytes),'cuFuncSetAttribute(max_dynamic_shared_bytes)')
        values = [C.c_uint64(a.pointer) if isinstance(a, DeviceBuffer) else a for a in args]
        if any(not isinstance(a, (C.c_int, C.c_uint, C.c_uint64, C.c_double, C.c_float)) for a in values):
            raise TypeError('Kernel scalar args must use explicit ctypes types')
        pointers = (C.c_void_p*len(values))(*(C.cast(C.byref(a), C.c_void_p) for a in values))
        launch = _bind(self.lib, 'cuLaunchKernel', [C.c_void_p]+[C.c_uint]*7+[C.c_void_p,C.c_void_p,C.c_void_p])
        self._check(launch(function, grid, 1, 1, block, 1, 1, shared_bytes, None, pointers, None), function_name)
        self._check(_bind(self.lib, 'cuCtxSynchronize', [])(), 'cuCtxSynchronize')

    def close(self):
        if self.context:
            # Destroying this private context releases only its own allocations.
            self._check(_bind(self.lib, 'cuCtxDestroy_v2', [C.c_void_p])(self.context), 'cuCtxDestroy_v2')
            self.context = C.c_void_p()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
