"""Assemble a relocatable private engine; never mutate historical artifacts.

Static closure is intentionally conservative, including dormant comparison code.
Cached PTX remains byte-identical; build/compiler identities are historical data.
"""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import re
import shutil

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'src/kaksforge/_engine_root'
PORTABLE={
    'cpu-gradient-v2':'product-v1-windows-cpu-gradient',
    'cpu-backend-v1-reviewed':'product-v1-windows-cpu-backend',
    'gradient-optimizer-stagnation-v4b':'product-v1-windows-optimizer',
    'shared-statistics-v1-checked':'product-v1-windows-statistics',
}
NATIVE=('native-adapter-checked','publication-graph-host-v1','publication-tight-round2-v1',
        'driver-staging-round3-v1','cpu-reference-cached','cpu-reference',
        'batched-optimizer-v1-reviewed','shared-core-v1-reviewed')

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

def copy(source,destination):
    if destination.exists():
        if sha(source)!=sha(destination):raise FileExistsError('Preserve different destination: '+str(destination))
    else:
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,destination)

def closure():
    pending=['product_v1_engine_fit.py','product_v1_engine_reproduce.py'];seen=set()
    while pending:
        name=pending.pop()
        if name in seen:continue
        target=DEST/'scripts'/name;source=target if target.is_file() else ROOT/'scripts'/name
        if not source.is_file():raise FileNotFoundError(source)
        if source!=target:copy(source,target)
        seen.add(name)
        for node in ast.walk(ast.parse(source.read_bytes())):
            modules=[]
            if isinstance(node,ast.Import):modules=[a.name.split('.')[0] for a in node.names]
            if isinstance(node,ast.ImportFrom) and node.module:modules=[node.module.split('.')[0]]
            for module in modules:
                child=module+'.py'
                if (ROOT/'scripts'/child).is_file() or (DEST/'scripts'/child).is_file():pending.append(child)
    return sorted(seen)

def assemble():
    if (DEST/'bundle-proof.json').exists():raise FileExistsError('Preserve assembled engine; use manifest command after platform overlays')
    names=closure()
    builds=list(NATIVE)+list(PORTABLE)+[p.name for p in (ROOT/'build').iterdir() if p.is_dir() and p.name.startswith('gpu-')]
    for logical in sorted(set(builds)):
        source=ROOT/'build'/PORTABLE.get(logical,logical)
        target=DEST/'build'/logical
        if target.exists():raise FileExistsError(target)
        shutil.copytree(source,target,ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.pdb','*.log'))
    # Actual source/header snapshots are retained for build audit and source delivery.
    for relative in ('src/native','src/core','src/gpu_round2','src/gpu_round3','src/gpu_round4','src/gpu_round5','src/gpu_round6',
                     'tools/kakscalculator2-2.0.1','build/native-adapter-src'):
        source=ROOT/relative
        if source.exists():shutil.copytree(source,DEST/relative,ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.pdb','*.log'))
    compiler_files={name:sha(ROOT/'envs/cuda-runtime/Library/bin'/name) for name in ('nvrtc64_120_0.dll','nvrtc-builtins64_129.dll')}
    dump(DEST/'compiler-provenance.json',{'schema':'kaksforge.historical_compiler_identity/1',
         'scope':'Recorded compiler used to create immutable PTX; compiler is not required or loaded for inference', 'files':compiler_files})
    helper='''"""Historical build identities; manifest-pinned data, no compiler loading."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
def historical_compiler_sha(path):
    name=Path(path).name
    records=json.loads((ROOT/'compiler-provenance.json').read_text(encoding='utf-8'))['files']
    if name not in records:raise ValueError('Unknown historical compiler identity: '+name)
    return records[name]
'''
    (DEST/'scripts/product_v1_cached_assets.py').write_text(helper,encoding='utf-8')
    transforms={}
    for name in names:
        target=DEST/'scripts'/name;before=target.read_text(encoding='utf-8-sig');after=before
        if name.startswith('gpu_'):
            after=re.sub(r'\bsha\((compiler|builtins)\)',r'historical_compiler_sha(\1)',after)
            after=after.replace("hashlib.sha256((ROOT/'envs/cuda-runtime/Library/bin/nvrtc64_120_0.dll').read_bytes()).hexdigest()",
                                "historical_compiler_sha(ROOT/'envs/cuda-runtime/Library/bin/nvrtc64_120_0.dll')")
            if after!=before:after=after.replace('from __future__ import annotations','from __future__ import annotations\nfrom product_v1_cached_assets import historical_compiler_sha')
        if name=='publication_gpu_native_driver_round3.py':
            after=after.replace("            ROOT / 'envs/cuda-runtime/Library/bin/nvrtc64_120_0.dll',\n",'').replace("            ROOT / 'envs/cuda-runtime/Library/bin/nvrtc-builtins64_129.dll',\n",'')
        if name=='cuda_driver.py':
            marker='    _dll_paths()\n    # NVRTC loads its builtins'
            if marker not in after:raise ValueError('Compiler guard marker missing')
            after=after.replace(marker,"    raise RuntimeError('KaKsForge release inference requires installed cached PTX; runtime compilation is disabled')\n    _dll_paths()\n    # NVRTC loads its builtins",1)
        if after!=before:
            target.write_text(after,encoding='utf-8',newline='\n')
            transforms[name]={'before_sha256':hashlib.sha256(before.encode()).hexdigest(),'after_sha256':sha(target),
                              'purpose':'Historical compiler hash lookup or compiler prohibition only; numerical kernels unchanged'}
    copy(ROOT/'envs/cuda-torch/Lib/site-packages/torch/include/ATen/native/cuda/Reduce.cuh',
         DEST/'envs/cuda-torch/Lib/site-packages/torch/include/ATen/native/cuda/Reduce.cuh')
    copy(ROOT/'envs/cuda-torch/include/cuda.h',DEST/'envs/cuda-torch/include/cuda.h')
    # MKL dispatcher dynamically chooses CPU architecture; keep its complete DLL set.
    blas=DEST/'envs-runtime/windows/blas'
    for record_name in ('mkl-2023.1.0-h6a75c08_48682.json','intel-openmp-2025.0.0-haa95532_1165.json'):
        record=json.loads((ROOT/'envs/cuda-torch/conda-meta'/record_name).read_text())
        dump(DEST/'third-party'/record_name,record)
        for relative in record['files']:
            source=ROOT/'envs/cuda-torch'/relative
            if source.is_file() and source.suffix.lower()=='.dll':copy(source,blas/source.name)
            elif source.is_file() and any(word in relative.lower() for word in ('license','eula','notice')):
                copy(source,DEST/'third-party'/record['name']/relative)
    for source in (ROOT/'envs/cuda-torch/Library/bin').glob('tbb*.dll'):copy(source,blas/source.name)
    for filename in ('libwinpthread-1.dll',):copy(ROOT/'envs/cpu-reference/Library/bin'/filename,DEST/'envs-runtime/windows/native'/filename)
    copy(ROOT/'tools/kakscalculator2-2.0.1/LICENSE',DEST/'third-party/KaKs_Calculator_2_LICENSE')
    torch_license=ROOT/'envs/cuda-torch/Lib/site-packages/torch-2.5.1.dist-info/LICENSE'
    if torch_license.exists():copy(torch_license,DEST/'third-party/PyTorch_LICENSE')
    dump(DEST/'bundle-proof.json',{'schema':'kaksforge.engine_bundle_proof/1','static_closure':names,
         'source_transforms':transforms,'portable_CPU_build_map':PORTABLE,
         'PTX_bytes_unchanged':True,'compiler_not_packaged':True,'historical_sources_modified':False})
    repin()
    print(json.dumps({'scripts':len(names),'builds':len(set(builds)),'resource_root':str(DEST)}))

def repin():
    # Product-only records bind portable Python loader text without changing PTX.
    mapping={}
    for old in (ROOT/'scripts').glob('*.py'):
        copied=DEST/'scripts'/old.name
        if copied.exists():mapping[sha(old)]=sha(copied)
    for record in (DEST/'build').glob('gpu-*/build.json'):
        data=json.loads(record.read_text(encoding='utf-8-sig'));changed={}
        originals=data.get('product_original_runtime_identities',{})
        for key,value in list(data.items()):
            original=originals.get(key,value)
            if 'runtime' in key and key.endswith('sha256') and isinstance(original,str) and original in mapping and mapping[original]!=value:
                changed[key]=original;data[key]=mapping[original]
        if changed:
            data.setdefault('product_original_runtime_identities',{}).update(changed)
            dump(record,data)

def manifest():
    repin()
    files={p.relative_to(DEST).as_posix():{'sha256':sha(p),'bytes':p.stat().st_size}
           for p in sorted(DEST.rglob('*')) if p.is_file() and p.name!='engine-assets.json' and '__pycache__' not in p.parts and p.suffix!='.pyc'}
    dump(DEST/'engine-assets.json',{'schema':'kaksforge.engine_assets/1','version':'1.0.0',
         'platform':'Windows x86-64 and Linux x86-64','files':files,
         'science_contract':{'models':14,'genetic_code':1,'precision':'FP64','method':'MA',
             'robust':'conditional empirical/F3x4 pi; fixed bounds; analytic v4b optimizer; full61 states',
             'reproduce':'historical upstream estimator; platform-specific equivalence validation required'}})
    print(json.dumps({'manifest_sha256':sha(DEST/'engine-assets.json'),'assets':len(files),'bytes':sum(v['bytes'] for v in files.values())}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('assemble','manifest'));a=p.parse_args()
    assemble() if a.command=='assemble' else manifest()
