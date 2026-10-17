"""Apply platform-only path/loader substitutions to bundled Python copies.

The original scientific scripts remain untouched. This is an assembly step,
not an inference-time patch. Every substituted source is pinned by the final
product manifest after assembly.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import shutil

ROOT=Path(__file__).resolve().parents[1]

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def add_import(text,names):
    line='from product_v1_platform import '+', '.join(sorted(set(names)))+'\n'
    if 'from product_v1_platform import ' in text:raise RuntimeError('Preserve applied platform overlay; refuse reapplication')
    future='from __future__ import annotations\n'
    if text.count(future)!=1:raise RuntimeError('Expected one annotations future import')
    return text.replace(future,future+line,1)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--resource-root',type=Path,default=ROOT/'src/kaksforge/_engine_root')
    p.add_argument('--linux-build',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();resource=a.resource_root.resolve();resource.relative_to(ROOT);build=a.linux_build.resolve();build.relative_to(ROOT);output=a.output.resolve();output.relative_to(ROOT)
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True);changes=[]
    for file in sorted((resource/'scripts').glob('*.py')):
        if file.name in ('product_v1_platform.py','product_v1_engine_fit.py','product_v1_engine_runtime.py','product_v1_owned_job.py','product_v1_engine_reproduce.py','publication_provenance.py'):continue
        before=file.read_text(encoding='utf-8');text=before;imports=set()
        if "C.WinDLL('nvcuda.dll')" in text:text=text.replace("C.WinDLL('nvcuda.dll')",'cuda_driver()');imports.add('cuda_driver')
        if 'os.add_dll_directory(' in text:text=text.replace('os.add_dll_directory(','dll_directory(');imports.add('dll_directory')
        if file.name in ('native_adapter.py','batched_optimizer.py','shared_core.py','cpu_backend.py','cpu_gradient.py','torch_graph_likelihood_fast.py'):
            lines=text.splitlines(keepends=True);found=0
            for i,line in enumerate(lines):
                # All these native constructors keep the source's mathematical
                # method body. Only the Path wrapper chooses .dll versus .so.
                match=re.search(r'(self\.path\s*=\s*)Path\((.+)\)\.resolve\(\)',line)
                if match:
                    lines[i]=line[:match.start()]+match.group(1)+'native_path(Path('+match.group(2)+'))'+line[match.end():];found+=1
            if found!=1:raise RuntimeError(f'Expected one native path assignment: {file.name} ({found})')
            text=''.join(lines);imports.add('native_path')
        if file.name=='native_adapter.py':
            text=text.replace('if os.name != "nt":\n            raise RuntimeError("Native Windows is required")',
                              'if os.name not in ("nt", "posix"):\n            raise RuntimeError("Windows or Linux is required")')
        if file.name in ('cpu_backend.py','cpu_gradient.py'):
            pattern=r"self\.mkl_path=Path\(mkl or os\.environ\.get\('KAKSFORGE_MKL_RUNTIME'\) or ROOT/'envs-runtime/windows/blas/mkl_rt\.2\.dll'\)\.resolve\(\)"
            text,count=re.subn(pattern,'self.mkl_path=blas_runtime(ROOT,mkl)',text)
            if count!=1:raise RuntimeError(f'Expected existing product BLAS default once: {file.name}')
            text=text.replace("(self.path.parent/'build.json')",'build_record(self.path)');imports.update(('blas_runtime','build_record'))
            if file.name=='cpu_backend.py':
                needle='    """Actually loaded MKL implementation/OpenMP DLLs, not only the dispatcher."""\n'
                if text.count(needle)!=1:raise RuntimeError('Expected exact loaded-module function marker')
                text=text.replace(needle,needle+"    if os.name!='nt':return linux_blas_modules(ROOT)\n",1);imports.add('linux_blas_modules')
        if file.name=='publication_gpu_native_driver_round3.py':
            text=text.replace("HOST_DLL = ROOT / 'build/publication-graph-host-v1/torch_graph_host.dll'","HOST_DLL = native_path(ROOT / 'build/publication-graph-host-v1/torch_graph_host.dll')")
            text=text.replace("STAGING_DLL = ROOT / 'build/driver-staging-round3-v1/driver_staging.dll'","STAGING_DLL = native_path(ROOT / 'build/driver-staging-round3-v1/driver_staging.dll')")
            text=text.replace("self.path.parent / 'build.json'",'build_record(self.path)').replace("HOST_DLL.parent / 'build.json'",'build_record(HOST_DLL)')
            text=text.replace("self.path.parent / 'compiled_source_snapshot'",'snapshot_directory(self.path)').replace("HOST_DLL.parent / 'compiled_source_snapshot'",'snapshot_directory(HOST_DLL)')
            imports.update(('native_path','build_record','snapshot_directory'))
        if file.name=='publication_gpu_tight_round2.py':
            text=text.replace('self.path=Path(dll_path).resolve()','self.path=native_path(dll_path)')
            text=text.replace("self.path.parent/'build.json'",'build_record(self.path)').replace("self.path.parent/'compiled_source_snapshot'",'snapshot_directory(self.path)')
            imports.update(('native_path','build_record','snapshot_directory'))
        if text!=before:
            text=add_import(text,imports);ast.parse(text,filename=str(file))
            changes.append(dict(path=str(file.relative_to(resource)),before_sha256=sha(file),after_sha256=hashlib.sha256(text.encode()).hexdigest(),substitutions='Platform dynamic library/directory path, native metadata and CPU loaded-library inventory only'))
            file.write_text(text,encoding='utf-8',newline='\n')
    # Preserve both platform build records and snapshots in the resource root.
    native_map=json.loads((build/'linux-native-map.json').read_text())
    copied={}
    for windows_name,linux_name in native_map['library_map'].items():
        source=build/linux_name;target=resource/linux_name
        if target.exists():raise FileExistsError(target)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        shutil.copy2(source.parent/'build.json',target.parent/'linux-build.json')
        shutil.copytree(source.parent/'compiled_source_snapshot',target.parent/'linux_compiled_source_snapshot')
        copied[linux_name]=sha(target)
    runtime=resource/'envs-runtime/linux/blas';shutil.copytree(build/'envs-runtime/blas',runtime)
    shutil.copytree(build/'platform_overlay',resource/'src/platform_v1_linux')
    shutil.copy2(build/'linux-native-map.json',resource/'linux-native-build.json')
    report=dict(schema='kaksforge.product_linux_python_platform_overlay/1',PASS=True,changes=changes,copied_native_sha256=copied,native_map_sha256=sha(build/'linux-native-map.json'),
        helper_sha256=sha(resource/'scripts/product_v1_platform.py'),builder_sha256=sha(__file__),CPU_evaluation=False,GPU_execution=False,original_scientific_scripts_modified=False)
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(dict(PASS=True,patched_copied_scripts=len(changes),native_binaries=len(copied),output=str(output))))
if __name__=='__main__':main()
