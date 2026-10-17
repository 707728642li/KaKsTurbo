"""Recompile exact pinned Linux translation units in a source build workspace.

This is a build-time operation only. Runtime integrity verification never creates
or trusts a new manifest. Original source/record hashes are retained as provenance.
Third-party OpenBLAS is the licensed, pinned binary distribution in engine/.
"""
from pathlib import Path
import hashlib,json,os,shlex,subprocess
R=Path(__file__).resolve().parents[1];E=R/'engine'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
original=json.loads((E/'engine-assets.json').read_text());original_pin=sha(E/'engine-assets.json')
# Input hashes establish exact source provenance, including platform glue.
for n,record in original['files'].items():
    if sha(E/n)!=record['sha256']:raise RuntimeError('Changed source/runtime input: '+n)
cxx=shlex.split(os.environ.get('CXX','g++'));records=[]
for recordfile in sorted(E.glob('build/*/linux-build.json')):
    record=json.loads(recordfile.read_text());oldcmd=record['command'];cmd=list(cxx)
    for arg in oldcmd[1:]:
        marker = next((s for s in ('/engine/', '/_engine_root/') if s in arg), None)
        if marker:
            suffix=arg.split(marker,1)[1]
            mapped=(E/suffix).resolve();mapped.relative_to(E)
            arg=('-I' if arg.startswith('-I') else '')+str(mapped)
        elif arg.startswith(('/data/', '/home/', '/usr/bin/')):
            raise ValueError('Unmapped historical build path: '+arg)
        cmd.append(arg)
    with (recordfile.parent/'release-build.log').open('wb') as log:
        subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    target=Path(cmd[cmd.index('-o')+1]);previous=sha(recordfile)
    record.update(command=cmd,compiler=cxx,compiler_version=subprocess.check_output(cxx+['--version'],text=True).splitlines()[0],
                  dll=str(target),dll_sha256=sha(target),packaging_source_record_sha256=previous,
                  scientific_source_unchanged=True,release_rebuild=True)
    recordfile.write_text(json.dumps(record,indent=2)+'\n');records.append(str(target.relative_to(E)))
blas=E/'envs-runtime/linux/blas';source=E/'src/platform_v1_linux/openblas64_bridge.cpp'
if not source.is_file():raise RuntimeError('Missing corresponding OpenBLAS bridge source')
cmd=cxx+['-O3','-std=c++17','-march=x86-64','-mtune=generic','-fno-fast-math','-ffp-contract=off','-shared','-fPIC','-static-libgcc','-static-libstdc++',str(source),'-L'+str(blas),'-l:libscipy_openblas64_-32a4b2a6.so','-Wl,-rpath,$ORIGIN','-o',str(blas/'libkaks_blas64.so')]
subprocess.run(cmd,check=True)
build={'source_manifest_sha256':original_pin,'rebuilt_components':records+['envs-runtime/linux/blas/libkaks_blas64.so'],
       'algorithm_sources_changed':False,'third_party_BLAS':'Pinned BSD-3-Clause OpenBLAS binary; same as qualified Linux distribution',
       'precision':'FP64','fast_math':False,'max_iterations':500,'model_count':14,'gradient_threshold':1e-8}
(E/'release-native-build.json').write_text(json.dumps(build,indent=2)+'\n')
original['files']={p.relative_to(E).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(E.rglob('*')) if p.is_file() and p.name!='engine-assets.json'}
original['packaging_source_manifest_sha256']=original_pin
(E/'engine-assets.json').write_text(json.dumps(original,indent=2)+'\n');pin=sha(E/'engine-assets.json')
pins=json.loads((R/'engine_pins.json').read_text());old=pins['linux_x86_64'];pins['linux_x86_64']=pin
(R/'engine_pins.json').write_text(json.dumps(pins,indent=2)+'\n')
p=R/'src/kaksturbo/_version.py';p.write_text(p.read_text().replace(old,pin),encoding='utf8')
identity=R/'release_identity.json'
if identity.is_file():
    data=json.loads(identity.read_text());data['pins']=pins;data['native_source_rebuilt']=True
    identity.write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps(build))
