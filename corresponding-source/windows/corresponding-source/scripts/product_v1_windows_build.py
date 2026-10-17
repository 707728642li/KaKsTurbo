"""Build distributable x86-64 CPU libraries from exact previously compiled sources.

No historical build/source is modified. Scientific validation is a separate gate.
"""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
COMPILER = ROOT / 'envs/cpu-reference/Library/bin/g++.exe'
SYSLIB = ROOT / 'envs/cpu-reference/Library/x86_64-w64-mingw32/sysroot/usr/lib'
JOBS = (
    ('cpu-gradient-v2', 'product-v1-windows-cpu-gradient', 'kaks_cpu_gradient.dll', ('shared_core.cpp','cpu_gradient.cpp'), True),
    ('cpu-backend-v1-reviewed', 'product-v1-windows-cpu-backend', 'kaks_cpu_backend.dll', ('shared_core.cpp','cpu_backend.cpp'), True),
    ('gradient-optimizer-stagnation-v4b', 'product-v1-windows-optimizer', 'kaks_optimizer.dll', ('batched_optimizer.cpp',), False),
    ('shared-statistics-v1-checked', 'product-v1-windows-statistics', 'kaks_statistics.dll', ('statistics.cpp','shared_core.cpp'), False),
)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    env = dict(os.environ, PATH=str(COMPILER.parent)+';'+os.environ['PATH'])
    version = subprocess.check_output([str(COMPILER),'--version'],env=env,text=True)
    outcomes = []
    for old, new, binary, units, threaded in JOBS:
        source = ROOT/'build'/old
        destination = ROOT/'build'/new
        destination.mkdir(exist_ok=False)
        copied = destination/'compiled_source_snapshot'
        shutil.copytree(source/'compiled_source_snapshot',copied)
        old_record = json.loads((source/'build.json').read_text(encoding='utf-8-sig'))
        for name, expected in old_record['source_sha256'].items():
            if sha(copied/name) != expected:
                raise ValueError('Source snapshot hash mismatch: '+name)
        flags = ['-O3','-std=c++17','-march=x86-64','-mtune=generic','-fno-fast-math',
                 '-ffp-contract=off','-Wall','-Wextra','-static-libgcc','-static-libstdc++',
                 '-B'+str(SYSLIB)+'\\','-L'+str(SYSLIB),'-shared']
        if threaded:
            flags.append('-pthread')
        output = destination/binary
        command = [str(COMPILER),*flags,*(str(copied/unit) for unit in units),'-o',str(output)]
        start = time.perf_counter()
        with (destination/'build.log').open('wb') as log:
            done = subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        record = dict(old_record, schema='kaksforge.windows_portable_build/1',
                      compiler=str(COMPILER),compiler_version=version,compiler_sha256=sha(COMPILER),
                      flags=flags,dll=str(output),dll_sha256=sha(output) if output.exists() else None,
                      source_build=old,source_build_sha256=sha(source/'build.json'),
                      compiled_translation_units=list(units),target='Windows x86-64 baseline; runtime-dispatched MKL',
                      source_bytes_unchanged=True,numerically_validated=False,
                      build_script_sha256=sha(Path(__file__)),returncode=done.returncode,
                      build_seconds=time.perf_counter()-start)
        (destination/'build.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
        if done.returncode:
            raise RuntimeError('Portable CPU build failed: '+str(destination))
        outcomes.append(dict(component=old,output=str(output),sha256=sha(output)))
        print(json.dumps(outcomes[-1]),flush=True)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
