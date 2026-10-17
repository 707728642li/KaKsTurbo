"""Platform paths and dynamic loaders for the packaged, unchanged estimator.

Importing this module performs no library load or calculation. Platform
records remain distinct so Windows DLL provenance never describes Linux SOs.
"""
from __future__ import annotations
import ctypes as C
import os
from pathlib import Path

class _NoDirectory:
    def close(self):pass
    def __enter__(self):return self
    def __exit__(self,*args):self.close()

def native_path(path):
    path=Path(path)
    if os.name!='nt':
        if path.suffix.lower()=='.dll':path=path.with_name('lib'+path.stem+'.so')
        elif path.suffix.lower()=='.exe':path=path.with_suffix('')
    return path.resolve()

def build_record(path):
    """Return the record Path for an actual native binary (no JSON parsing)."""
    path=Path(path)
    return path.parent/('build.json' if os.name=='nt' else 'linux-build.json')

def snapshot_directory(path):
    path=Path(path)
    return path.parent/('compiled_source_snapshot' if os.name=='nt' else 'linux_compiled_source_snapshot')

def dll_directory(path):
    path=Path(path)
    root=Path(__file__).resolve().parents[1]
    if path==root/'envs/cpu-reference/Library/bin':path=root/'envs-runtime/windows/native'
    return os.add_dll_directory(str(path)) if os.name=='nt' and path.is_dir() else _NoDirectory()

def cuda_driver():
    return C.WinDLL('nvcuda.dll') if os.name=='nt' else C.CDLL('libcuda.so.1')

def blas_runtime(root,explicit=None):
    selected=explicit or os.environ.get('KAKSFORGE_MKL_RUNTIME')
    if selected:return Path(selected).resolve()
    root=Path(root)
    return (root/('envs-runtime/windows/blas/mkl_rt.2.dll' if os.name=='nt' else 'envs-runtime/linux/blas/libkaks_blas64.so')).resolve()

def linux_blas_modules(root):
    root=Path(root).resolve();result={}
    for line in Path('/proc/self/maps').read_text(encoding='utf-8').splitlines():
        fields=line.split(maxsplit=5)
        if len(fields)<6 or not fields[5].startswith('/'):continue
        path=Path(fields[5].removesuffix(' (deleted)')).resolve()
        if any(marker in path.name.lower() for marker in ('openblas','kaks_blas','gfortran','quadmath','libiomp','mkl_')):
            import hashlib
            in_project=path.is_relative_to(root)
            result[str(path)]={'path':str(path),'in_project':in_project,'sha256':hashlib.sha256(path.read_bytes()).hexdigest() if in_project else None}
    return sorted(result.values(),key=lambda item:item['path'])
