"""Build the remaining portable Windows artifacts from pinned historical bytes.

This performs compilation and PE inspection only. It never runs an estimator,
loads a CUDA driver, replaces a historical file, or changes product resources.
Numerical equivalence remains a separate release gate.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
COMPILER = ROOT / 'envs/cpu-reference/Library/bin/g++.exe'
OBJDUMP = COMPILER.with_name('x86_64-w64-mingw32-objdump.exe')
SYSLIB = ROOT / 'envs/cpu-reference/Library/x86_64-w64-mingw32/sysroot/usr/lib'
SNAPSHOT_JOBS = (
    ('publication-graph-host-v1', 'torch_graph_host.dll', ('publication_graph_host.cpp',)),
    ('driver-staging-round3-v1', 'driver_staging.dll', ('request_staging.cpp',)),
    ('publication-tight-round2-v1', 'tight_bound.dll', ('tight_bound.cpp',)),
)
CLI_UNITS = ('KaKs_Calculator.cpp', 'KaKs.cpp', 'MSMA.cpp', 'MYN.cpp',
             'base.cpp', 'NG86.cpp', 'LWL85.cpp', 'LPB93.cpp', 'GY94.cpp', 'YN00.cpp')


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def verify(directory, expected):
    actual = {name: sha(directory / name) for name in expected}
    if actual != expected:
        raise ValueError('Historical source bytes differ: ' + str(directory))
    return actual


def pe_dependencies(path, env):
    output = subprocess.check_output([str(OBJDUMP), '-p', str(path)], env=env, text=True)
    imports = re.findall(r'DLL Name:\s*(\S+)', output)
    if any('libstdc++' in name.lower() or 'libgcc' in name.lower() for name in imports):
        raise ValueError('Compiler runtime was not statically linked: ' + str(path))
    return {'format_x86_64': 'file format pei-x86-64' in output,
            'imported_DLLs': imports, 'compiler_runtimes_statically_linked': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=ROOT/'build/product-v1-windows-remaining-v1')
    arguments = parser.parse_args()
    if os.name != 'nt':
        parser.error('Use native Windows; Linux builds have a separate script')
    destination_root = arguments.output_root.resolve()
    destination_root.relative_to(ROOT/'build')
    destination_root.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PATH=str(COMPILER.parent)+';'+os.environ.get('PATH', ''))
    compiler_sha = sha(COMPILER)
    compiler_version = subprocess.check_output([str(COMPILER), '--version'], env=env, text=True)
    default_target = subprocess.check_output([str(COMPILER), '-Q', '--help=target'], env=env, text=True)
    (destination_root/'compiler_default_target.txt').write_text(default_target, encoding='utf-8')
    compiler_identity = dict(compiler=str(COMPILER), compiler_sha256=compiler_sha,
                             compiler_version=compiler_version)
    base_flags = ['-O3', '-march=x86-64', '-mtune=generic', '-fno-fast-math', '-ffp-contract=off',
                  '-Wall', '-Wextra', '-static-libgcc', '-static-libstdc++',
                  '-B'+str(SYSLIB)+'\\', '-L'+str(SYSLIB)]
    tracked = {str(COMPILER): compiler_sha}
    artifacts = []
    report = dict(schema='kaksforge.windows_remaining_baseline_build/1', PASS=False,
                  complete=False, scientific_execution=False, CUDA_execution=False,
                  resource_replacement=False, numerical_equivalence_validated=False,
                  created_at=datetime.now(timezone.utc).isoformat(),
                  builder_sha256=sha(Path(__file__)), artifacts=artifacts)

    def tracked_source(path, expected):
        actual = verify(path, expected)
        tracked.update({str(path/name): value for name, value in actual.items()})
        return actual

    def compile_job(component, binary, snapshot_source, expected, units, standard, shared=True):
        historical = ROOT/'build'/component
        old_record_path = historical/'build.json'
        old_record = read(old_record_path)
        tracked[str(old_record_path)] = sha(old_record_path)
        tracked[str(historical/binary)] = sha(historical/binary)
        if tracked[str(historical/binary)] != old_record.get('dll_sha256', old_record.get('executable_sha256')):
            raise ValueError('Historical binary does not match record: '+component)
        source_before = tracked_source(snapshot_source, expected)
        destination = destination_root/component
        destination.mkdir()
        snapshot = destination/'compiled_source_snapshot'
        shutil.copytree(snapshot_source, snapshot)
        if verify(snapshot, expected) != source_before:
            raise ValueError('Copied source mismatch: '+component)
        output = destination/binary
        flags = [*base_flags, '-std='+standard]
        if shared:
            flags.append('-shared')
        command = [str(COMPILER), *flags, *(str(snapshot/name) for name in units), '-o', str(output)]
        started = time.perf_counter()
        with (destination/'build.log').open('wb') as stream:
            done = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, check=False)
        record = dict(old_record, **compiler_identity, schema='kaksforge.windows_portable_build/1',
                      flags=flags, command=command, source_sha256=source_before,
                      compiled_translation_units=list(units),
                      source_build=component, source_build_sha256=sha(old_record_path),
                      source_snapshot_origin=str(snapshot_source), source_bytes_unchanged=True,
                      target='Windows x86-64 baseline', numerically_validated=False,
                      CUDA_execution=False, estimator_execution=False,
                      returncode=done.returncode, build_seconds=time.perf_counter()-started,
                      build_script_sha256=sha(Path(__file__)), built_at=datetime.now(timezone.utc).isoformat())
        key = 'dll' if shared else 'executable'
        record[key] = str(output)
        record[key+'_sha256'] = sha(output) if output.is_file() else None
        if not shared:
            record['optimization'] = 'O3-x86-64-generic-no-fast-math-no-FMA'
            shutil.copy2(historical/'source_manifest.json', destination/'source_manifest.json')
            record['source_manifest'] = str(destination/'source_manifest.json')
            record['source_manifest_sha256'] = sha(destination/'source_manifest.json')
        dump(destination/'build.json', record)
        if done.returncode:
            raise RuntimeError('Portable build failed, evidence retained: '+str(destination))
        verified = pe_dependencies(output, env)
        if not verified['format_x86_64']:
            raise ValueError('Wrong executable architecture: '+str(output))
        verify(snapshot_source, expected)
        item = dict(component=component, decision='rebuilt_from_exact_source_bytes',
                    historical_binary=str(historical/binary), historical_sha256=tracked[str(historical/binary)],
                    output=str(output), sha256=record[key+'_sha256'],
                    build_record=str(destination/'build.json'), build_record_sha256=sha(destination/'build.json'),
                    source_sha256=source_before, PE=verified, numerical_equivalence_validated=False)
        artifacts.append(item)
        print(json.dumps({'component': component, 'decision': item['decision'], 'output': str(output)}), flush=True)

    try:
        adapter = ROOT/'build/native-adapter-checked'
        adapter_record = read(adapter/'build.json')
        if adapter_record['compiler_sha256'] != compiler_sha:
            raise ValueError('Historical adapter compiler identity differs')
        if not re.search(r'^\s*-march=\s+x86-64\s*$', default_target, re.M):
            raise ValueError('Compiler default ISA is not baseline x86-64')
        if any(flag.startswith(('-march=', '-mtune=', '-mavx', '-msse3', '-msse4', '-mfma')) for flag in adapter_record['flags']):
            raise ValueError('Adapter is no longer an already-baseline build')
        if sha(adapter/'kaks_adapter.dll') != adapter_record['executable_sha256']:
            raise ValueError('Historical adapter binary differs')
        tracked[str(adapter/'build.json')] = sha(adapter/'build.json')
        tracked[str(adapter/'kaks_adapter.dll')] = sha(adapter/'kaks_adapter.dll')
        staged = tracked_source(ROOT/'build/native-adapter-src', adapter_record['staged_source_sha256'])
        native = tracked_source(ROOT/'src/native', adapter_record['native_source_sha256'])
        artifacts.append(dict(component='native-adapter-checked', decision='reuse_verified_already_baseline',
                              output=str(adapter/'kaks_adapter.dll'), sha256=adapter_record['executable_sha256'],
                              build_record=str(adapter/'build.json'), build_record_sha256=sha(adapter/'build.json'),
                              compiler_default_target_sha256=sha(destination_root/'compiler_default_target.txt'),
                              staged_source_sha256=staged, native_source_sha256=native,
                              PE=pe_dependencies(adapter/'kaks_adapter.dll', env),
                              scientific_arithmetic_unchanged=True, numerical_equivalence_validated=False))
        for component, binary, units in SNAPSHOT_JOBS:
            record = read(ROOT/'build'/component/'build.json')
            compile_job(component, binary, ROOT/'build'/component/'compiled_source_snapshot',
                        record['source_sha256'], units, 'c++17')
        cached_source_record = ROOT/'build/cpu-reference-cached/source_manifest.json'
        tracked[str(cached_source_record)] = sha(cached_source_record)
        compile_job('cpu-reference-cached', 'KaKs_Calculator.exe', ROOT/'build/cpu-reference-cached-src',
                    read(cached_source_record)['cached_source_sha256'], CLI_UNITS, 'gnu++98', shared=False)
        report.update(complete=True, PASS=True)
    except BaseException as error:
        report['error'] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        after = {path: sha(path) for path in tracked}
        report['historical_files_before_sha256'] = tracked
        report['historical_files_after_sha256'] = after
        report['all_historical_files_unchanged'] = tracked == after
        report['PASS'] = report['PASS'] and report['all_historical_files_unchanged']
        dump(destination_root/'map-proof.json', report)
    return 0 if report['PASS'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
