"""Build isolated Linux native overlays from verified Windows source snapshots.

Run with an existing Linux Python/compiler. This script does not install,
download, execute fitting, initialize CUDA or modify the historical sources.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[1]
INHERITED_WINDOWS_FIELDS=('compiler_executable_sha256','compiled_translation_units','build_script_sha256',
    'frozen_shared_build_sha256','source_hash_scope','MKL_interface','compilation_source',
    'live_snapshot_match_after_build','base_build','base_dll_sha256','prepared_at')

def normalize_native_build(record,upstream,compiler_sha256,builder_sha256):
    """Separate historical Windows evidence from actual Linux build facts."""
    result=dict(record);moved={k:result.pop(k) for k in INHERITED_WINDOWS_FIELDS if k in result}
    result['upstream_windows_build_metadata']=dict(upstream)
    result['inherited_windows_fields_removed_from_current_scope']=moved
    result['compiler_sha256']=compiler_sha256
    result['compiler_executable_sha256']=compiler_sha256
    result['compiled_translation_units']=[v for v in result['command'] if str(v).endswith(('.cpp','.cc','.cxx','.c'))]
    result['build_script_sha256']=builder_sha256
    result['source_hash_scope']='Actual Linux compiled source snapshot listed by source_sha256; packaged as linux_compiled_source_snapshot'
    result['compiled_from']='Linux compiled source snapshot; exact compiler inputs in command/compiled_translation_units'
    cpu_blas='kaks_cpu_gradient' in str(result.get('dll','')) or 'kaks_cpu_backend' in str(result.get('dll',''))
    result['BLAS_runtime_used_by_this_component']=cpu_blas
    result['BLAS_implementation']=('Bundled OpenBLAS64, int64 CBLAS bridge with true per-thread one-thread setting restored after each evaluation'
        if cpu_blas else 'This component performs no BLAS calls; the packaged CPU objective/gradient providers use bundled OpenBLAS64')
    if cpu_blas:
        result['MKL_interface']='Compatibility symbol names only: cblas_dgemm_64 and MKL thread-setting exports forward to bundled OpenBLAS64, not Intel MKL'
        result['BLAS_runtime_legacy_field_names']='mkl_runtime/mkl_runtime_sha256 retain the Python loader ABI; their values identify the Linux OpenBLAS64 bridge'
    result['upstream_metadata_scope']='Historical Windows input build only; actual Linux compiler/flags/command/source hashes/binary hashes are top-level'
    return result
OVERLAY=ROOT/'src/platform_v1_linux'
ORIGINAL_ARCHIVE_SHA='d3f321647c262aaab5560fdc7fe886af11d3570fc27340d81ce5a757dcc2f0af'
OLD_PATH_GUARD="if(!path||!*path||!(path[0]==L'\\\\'||(path[0]&&path[1]==L':'&&path[2]==L'\\\\')))"
NEW_PATH_GUARD='if(!kaksforge_absolute_blas_path(path))'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'build/product-v1-linux-v1')
    p.add_argument('--openblas-source-dir',type=Path,required=True)
    p.add_argument('--numpy-license-dir',type=Path,required=True)
    a=p.parse_args()
    if os.name!='posix':p.error('Run this builder under Linux/WSL.')
    output=a.output.resolve();output.relative_to(ROOT)
    if output.exists():raise FileExistsError(f'Preserve earlier builds: {output}')
    if sha(ROOT/'software/cpu_kaks_reference/kakscalculator2-v2.0.1.tar.gz')!=ORIGINAL_ARCHIVE_SHA:raise RuntimeError('Historical archive changed')
    compiler=Path(shutil.which('g++') or '').resolve()
    if not compiler.is_file():raise FileNotFoundError('Existing g++ required')
    compiler_version=subprocess.run([str(compiler),'--version'],text=True,capture_output=True,check=True).stdout.strip()
    output.mkdir(parents=True)
    compiler_overlay=output/'platform_overlay';shutil.copytree(OVERLAY,compiler_overlay)
    platform_identity={str(f.relative_to(compiler_overlay)):sha(f) for f in sorted(compiler_overlay.rglob('*')) if f.is_file()}
    runtime=output/'envs-runtime/blas';runtime.mkdir(parents=True)
    libnames=('libscipy_openblas64_-32a4b2a6.so','libgfortran-040039e1-0352e75f.so.5.0.0','libquadmath-96973f99-934c22de.so.0.0.0')
    runtime_origin={}
    for name in libnames:
        source=a.openblas_source_dir/name
        if not source.is_file():raise FileNotFoundError(source)
        shutil.copy2(source,runtime/name)
        runtime_origin[name]={'source':str(source),'source_sha256':sha(source),'copied_sha256':sha(runtime/name)}
    shutil.copytree(a.numpy_license_dir,runtime/'licenses')
    common=['-O3','-std=c++17','-fno-fast-math','-ffp-contract=off','-Wall','-Wextra','-fPIC','-shared','-pthread','-static-libgcc','-static-libstdc++']
    bridge=runtime/'libkaks_blas64.so'
    bridge_command=[str(compiler),*common,str(compiler_overlay/'openblas64_bridge.cpp'),'-L'+str(runtime),'-l:'+libnames[0],'-Wl,-rpath,$ORIGIN','-o',str(bridge)]
    result=subprocess.run(bridge_command,text=True,capture_output=True)
    (runtime/'build.log').write_text(result.stdout+result.stderr,encoding='utf-8')
    if result.returncode:raise RuntimeError('OpenBLAS64 bridge compilation failed')
    bridge_metadata=dict(schema='kaksforge.linux_openblas64_bridge_build/1',platform='Linux x86_64',compiler=str(compiler),compiler_version=compiler_version,flags=common,command=bridge_command,
        arithmetic='Direct forwarding to existing scipy_cblas_dgemm64_; true openblas_set_num_threads_local; numerical fitting not executed',runtime_origin=runtime_origin,
        bridge_sha256=sha(bridge),source_sha256=sha(compiler_overlay/'openblas64_bridge.cpp'),licenses_sha256={str(f.relative_to(runtime/'licenses')):sha(f) for f in sorted((runtime/'licenses').rglob('*')) if f.is_file()})
    (runtime/'build.json').write_text(json.dumps(bridge_metadata,indent=2)+'\n',encoding='utf-8')
    specs=[
      ('cpu-gradient-v2','kaks_cpu_gradient.dll',['shared_core.cpp','cpu_gradient.cpp']),
      ('cpu-backend-v1-reviewed','kaks_cpu_backend.dll',['shared_core.cpp','cpu_backend.cpp']),
      ('shared-statistics-v1-checked','kaks_statistics.dll',['statistics.cpp','shared_core.cpp']),
      ('gradient-optimizer-stagnation-v4b','kaks_optimizer.dll',['batched_optimizer.cpp']),
      ('publication-graph-host-v1','torch_graph_host.dll',['publication_graph_host.cpp']),
      ('driver-staging-round3-v1','driver_staging.dll',['request_staging.cpp']),
      ('publication-tight-round2-v1','tight_bound.dll',['tight_bound.cpp']),
    ]
    mapping={};identities={};all_source_before={};transform_records=[]
    for name,windows_dll,units in specs:
        source_build=ROOT/'build'/name;metadata=read_json(source_build/'build.json')
        before={};dest=output/'build'/name;dest.mkdir(parents=True);snapshot=dest/'compiled_source_snapshot';snapshot.mkdir()
        for filename,expected in metadata['source_sha256'].items():
            source=source_build/'compiled_source_snapshot'/filename
            if sha(source)!=expected:raise RuntimeError(f'Frozen source snapshot changed: {source}')
            all_source_before[str(source.relative_to(ROOT))]=sha(source);before[filename]=sha(source)
            data=source.read_bytes()
            if filename in ('cpu_gradient.cpp','cpu_backend.cpp'):
                text=data.decode('latin1')
                if text.count(OLD_PATH_GUARD)!=1:raise RuntimeError(f'Expected exact absolute-path guard once: {filename}')
                transformed=text.replace(OLD_PATH_GUARD,NEW_PATH_GUARD)
                transformed=transformed.replace('MKL path must be an absolute native Windows path','BLAS runtime path must be an absolute Linux path')
                data=transformed.encode('latin1')
                transform_records.append(dict(build=name,file=filename,before_sha256=sha(source),after_sha256=hashlib.sha256(data).hexdigest(),numerical_body_unchanged=True,change='Platform absolute-path guard and corresponding diagnostic only; dlopen/locking via compiler overlay'))
            (snapshot/filename).write_bytes(data)
        library=dest/('lib'+windows_dll.removesuffix('.dll')+'.so')
        flags=[*common,'-include',str(compiler_overlay/'windows.h'),'-I'+str(compiler_overlay)]
        if name=='publication-graph-host-v1':flags+=['-DKAKSFORGE_HOST_FINITE_RENAME=1']
        command=[str(compiler),*flags,*(str(snapshot/u) for u in units),'-ldl','-o',str(library)]
        result=subprocess.run(command,text=True,capture_output=True);(dest/'build.log').write_text(result.stdout+result.stderr,encoding='utf-8')
        if result.returncode:raise RuntimeError(f'Linux build failed, inspect {dest}/build.log')
        copied={f.name:sha(f) for f in sorted(snapshot.iterdir()) if f.is_file()}
        new=dict(metadata,platform='Linux x86_64',compiler=str(compiler),compiler_version=compiler_version,compiler_sha256=sha(compiler),flags=flags,command=command,
            source_sha256=copied,original_Windows_source_sha256=before,dll=str(library),dll_sha256=sha(library),
            platform_overlay_sha256=platform_identity,platform_overlay_location=str(compiler_overlay),mkl_runtime=str(bridge),mkl_runtime_sha256=sha(bridge),
            BLAS_implementation='Bundled OpenBLAS64 via explicit int64 CBLAS and true thread-local one-thread bridge',compiled_from='compiled_source_snapshot',
            Windows_numerical_source_bodies_retained=True,Windows_numerical_equivalence_requires_separate_validation=True,CUDA_execution=False,CPU_optimizer_called=False,built_at=datetime.now(timezone.utc).isoformat())
        new=normalize_native_build(new,metadata,sha(compiler),sha(__file__))
        (dest/'build.json').write_text(json.dumps(new,indent=2)+'\n',encoding='utf-8')
        mapping[f'build/{name}/{windows_dll}']=str(library.relative_to(output));identities[str(library.relative_to(output))]=sha(library)
        print('compiled',name,flush=True)
    # Original prepared-data adapter: staged upstream code and hooks are retained.
    source_meta=read_json(ROOT/'build/native-adapter-checked/build.json')
    adapter=output/'build/native-adapter-checked';adapter.mkdir(parents=True);snapshot=adapter/'compiled_source_snapshot';snapshot.mkdir()
    for filename,expected in source_meta['staged_source_sha256'].items():
        source=ROOT/'build/native-adapter-src'/filename
        if sha(source)!=expected:raise RuntimeError(f'Frozen staged adapter source changed: {source}')
        all_source_before[str(source.relative_to(ROOT))]=sha(source);shutil.copy2(source,snapshot/filename)
    for filename in ('kaks_adapter.cpp','kaks_adapter.h','adapter_bridge.h'):
        source=ROOT/'src/native'/filename
        if sha(source)!=source_meta['native_source_sha256'][filename]:raise RuntimeError(f'Frozen native adapter changed: {source}')
        all_source_before[str(source.relative_to(ROOT))]=sha(source);shutil.copy2(source,snapshot/filename)
    library=adapter/'libkaks_adapter.so'
    adapter_flags=[f for f in common if f!='-O3'];adapter_flags.insert(0,'-O2');adapter_flags+=['-DKAKS_CUDA_HOST_ADAPTER=1','-include',str(compiler_overlay/'windows.h'),'-I'+str(compiler_overlay),'-I'+str(snapshot)]
    command=[str(compiler),*adapter_flags,*(str(snapshot/u) for u in ('base.cpp','GY94.cpp','MSMA.cpp','kaks_adapter.cpp')),'-ldl','-o',str(library)]
    result=subprocess.run(command,text=True,capture_output=True);(adapter/'build.log').write_text(result.stdout+result.stderr,encoding='utf-8')
    if result.returncode:raise RuntimeError(f'Linux adapter build failed: {adapter}/build.log')
    adapter_meta=dict(source_meta,platform='Linux x86_64',compiler=str(compiler),compiler_version=compiler_version,flags=adapter_flags,command=command,
        executable=str(library),executable_sha256=sha(library),dll=str(library),dll_sha256=sha(library),
        source_sha256={f.name:sha(f) for f in sorted(snapshot.iterdir()) if f.is_file()},platform_overlay_sha256=platform_identity,
        native_source_numerical_bodies_unchanged=True,thread_safety='Original recursive critical section mapped to pthread recursive mutex',state='compiled_not_run')
    adapter_meta=normalize_native_build(adapter_meta,source_meta,sha(compiler),sha(__file__))
    (adapter/'build.json').write_text(json.dumps(adapter_meta,indent=2)+'\n',encoding='utf-8')
    mapping['build/native-adapter-checked/kaks_adapter.dll']='build/native-adapter-checked/libkaks_adapter.so';identities[str(library.relative_to(output))]=sha(library)
    print('compiled native-adapter-checked',flush=True)
    # Historical reproduce estimator: original upstream source with no edits.
    historical=output/'build/cpu-reference';historical.mkdir(parents=True);snapshot=historical/'compiled_source_snapshot';snapshot.mkdir()
    original=ROOT/'tools/kakscalculator2-2.0.1/src';original_hashes={}
    for source in sorted(original.iterdir()):
        if source.is_file():original_hashes[source.name]=sha(source);all_source_before[str(source.relative_to(ROOT))]=sha(source);shutil.copy2(source,snapshot/source.name)
    executable=historical/'KaKs_Calculator'
    units=('KaKs_Calculator.cpp','KaKs.cpp','MSMA.cpp','MYN.cpp','base.cpp','NG86.cpp','LWL85.cpp','LPB93.cpp','GY94.cpp','YN00.cpp')
    historical_flags=['-O2','-std=gnu++98','-fno-fast-math','-ffp-contract=off','-static-libgcc','-static-libstdc++']
    command=[str(compiler),*historical_flags,*(str(snapshot/u) for u in units),'-o',str(executable)]
    result=subprocess.run(command,text=True,capture_output=True);(historical/'build.log').write_text(result.stdout+result.stderr,encoding='utf-8')
    if result.returncode:raise RuntimeError(f'Historical Linux build failed: {historical}/build.log')
    history_meta=dict(schema='kaksforge.linux_original_reference_build/1',platform='Linux x86_64',compiler=str(compiler),compiler_version=compiler_version,flags=historical_flags,command=command,
        executable=str(executable),executable_sha256=sha(executable),source_archive_sha256=ORIGINAL_ARCHIVE_SHA,source_sha256=original_hashes,source_modifications=[],original_source_unchanged=True,numerical_validation_executed=False)
    history_meta=normalize_native_build(history_meta,read_json(ROOT/'build/cpu-reference/build.json'),sha(compiler),sha(__file__))
    (historical/'build.json').write_text(json.dumps(history_meta,indent=2)+'\n',encoding='utf-8');mapping['build/cpu-reference/KaKs_Calculator.exe']='build/cpu-reference/KaKs_Calculator';identities[str(executable.relative_to(output))]=sha(executable)
    print('compiled cpu-reference',flush=True)
    all_source_after={p:sha(ROOT/p) for p in all_source_before}
    if all_source_after!=all_source_before:raise RuntimeError('Historical source changed during build')
    manifest=dict(schema='kaksforge.linux_native_overlay_build/1',PASS=True,platform='Linux x86_64',compiler=str(compiler),compiler_version=compiler_version,compiler_sha256=sha(compiler),
        output=str(output),library_map=mapping,library_sha256=identities,BLAS_runtime='envs-runtime/blas/libkaks_blas64.so',BLAS_runtime_sha256=sha(bridge),runtime_origin=runtime_origin,
        transformations=transform_records,platform_overlay_sha256=platform_identity,original_source_before=all_source_before,original_source_after=all_source_after,original_sources_unchanged=True,
        numerical_validation_executed=False,CUDA_execution=False,CPU_fits_executed=False,original_archive_sha256=ORIGINAL_ARCHIVE_SHA,builder_sha256=sha(__file__),built_at=datetime.now(timezone.utc).isoformat())
    (output/'linux-native-map.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8');print(json.dumps({'PASS':True,'compiled_native_units':len(mapping),'output':str(output)}),flush=True)

if __name__=='__main__':main()
