"""Offline PEP 517 packaging of pinned native engines; no download or fitting."""
from pathlib import Path
import base64,csv,hashlib,io,json,os,sys,tarfile,zipfile
ROOT=Path(__file__).resolve().parents[1]
VERSION='1.1.0'
DIST=f'kaksturbo-{VERSION}.dist-info'
def wheel_tag(platform):return 'manylinux_2_17_x86_64' if platform=='linux_x86_64' else platform
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def platform_name(settings=None):
 value=(settings or {}).get('platform',os.environ.get('KAKSTURBO_BUILD_PLATFORM','win_amd64' if os.name=='nt' else 'linux_x86_64'))
 if value not in ('win_amd64','linux_x86_64'):raise ValueError('Unsupported native platform')
 return value
def engine(platform):
 path=Path(os.environ.get('KAKSTURBO_ENGINE_DIR',str(ROOT/'engine'))).resolve()
 pins=json.loads((ROOT/'engine_pins.json').read_text())
 if not (path/'engine-assets.json').is_file() or sha(path/'engine-assets.json')!=pins[platform]:
  raise ValueError('A matching pinned engine is required. Set KAKSTURBO_ENGINE_DIR; no runtime is downloaded.')
 manifest=json.loads((path/'engine-assets.json').read_text())
 for name,record in manifest['files'].items():
  source=(path/name).resolve();source.relative_to(path)
  if source.stat().st_size!=record['bytes'] or sha(source)!=record['sha256']:raise ValueError('Changed scientific asset '+name)
 return path,manifest
def get_requires_for_build_wheel(config_settings=None):return []
def get_requires_for_build_sdist(config_settings=None):return []

def license_entries():
 # Short, content-addressed paths avoid Windows MAX_PATH failures in deep
 # installation prefixes. The mapping preserves component/source filenames.
 entries={p:'thirdparty/'+sha(p)[:16]+p.suffix for p in sorted((ROOT/'licenses').rglob('*')) if p.is_file()}
 seen={}
 for p,n in entries.items():
  if n in seen and sha(p)!=seen[n]:raise ValueError('Short license hash collision')
  seen[n]=sha(p)
 return entries
def metadata(platform,sdist=False):
 license_names=(['licenses/'+p.relative_to(ROOT/'licenses').as_posix() for p in license_entries()]
                if sdist else list(license_entries().values()))
 text=('Metadata-Version: 2.4\nName: kaksturbo\nVersion: '+VERSION+'\n'
 'Summary: Codon model averaging with CPU/GPU execution and numerical diagnostics\n'
 'Requires-Python: >=3.11\nMaintainer: Taishan Li\nMaintainer-email: litaishan@caf.ac.cn\n'
 'License-Expression: GPL-3.0-only\nLicense-File: LICENSE\n'+
 ''.join('License-File: '+n+'\n' for n in sorted(set(license_names)))+
 'Classifier: Development Status :: 5 - Production/Stable\nClassifier: Programming Language :: Python :: 3\n'
 'Project-URL: Source, https://github.com/707728642li/KaKsTurbo\n'
 'Project-URL: Issues, https://github.com/707728642li/KaKsTurbo/issues\n'
 'Description-Content-Type: text/markdown\n\n'+(ROOT/('README_PYPI.md' if (ROOT/'README_PYPI.md').is_file() else 'README.md')).read_text(encoding='utf8'))
 return {DIST+'/METADATA':text.encode(),DIST+'/WHEEL':f'Wheel-Version: 1.0\nGenerator: KaKsTurbo offline pinned-engine builder\nRoot-Is-Purelib: false\nTag: py3-none-{wheel_tag(platform)}\n'.encode(),DIST+'/entry_points.txt':b'[console_scripts]\nkaksturbo = kaksturbo.cli:main\n'}
def prepare_metadata_for_build_wheel(metadata_directory,config_settings=None):
 for n,data in metadata(platform_name(config_settings)).items():
  p=Path(metadata_directory)/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
 return DIST
def build_wheel(wheel_directory,config_settings=None,metadata_directory=None):
 platform=platform_name(config_settings);runtime,manifest=engine(platform)
 if platform=='linux_x86_64':
  required=('build/cpu-backend-v1-reviewed/libkaks_cpu_backend.so',
            'build/cpu-gradient-v2/libkaks_cpu_gradient.so',
            'build/gradient-optimizer-schur-v1/libkaks_optimizer.so',
            'envs-runtime/linux/blas/libkaks_blas64.so')
  missing=[n for n in required if n not in manifest['files'] or not (runtime/n).is_file()]
  if missing:raise ValueError('Linux native engine is not built; use the Conda build or an official prebuilt package. Missing: '+', '.join(missing))
 files={}
 for pkg in ('kaksturbo','kaksforge'):
  for p in sorted((ROOT/'src'/pkg).glob('*.py')):files[pkg+'/'+p.name]=p
 files.update({'kaksturbo/_engine_root/'+n:runtime/n for n in manifest['files']})
 files['kaksturbo/_engine_root/engine-assets.json']=runtime/'engine-assets.json'
 files[DIST+'/licenses/LICENSE']=ROOT/'LICENSE'
 mapping={}
 for p,n in license_entries().items():
  files[DIST+'/licenses/'+n]=p
  mapping[p.relative_to(ROOT/'licenses').as_posix()]=n
 files[DIST+'/licenses/COMPONENT_PATHS.json']=json.dumps(mapping,indent=2).encode()
 files['kaksturbo/_release_identity.json']=ROOT/'release_identity.json'
 out=Path(wheel_directory);out.mkdir(parents=True,exist_ok=True);name=f'kaksturbo-{VERSION}-py3-none-{wheel_tag(platform)}.whl'
 entries=[]
 with zipfile.ZipFile(out/name,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for n,v in sorted({**files,**metadata(platform)}.items()):
   data=v.read_bytes() if isinstance(v,Path) else v
   info=zipfile.ZipInfo(n,(2026,10,17,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=(0o755 if n.endswith('/KaKs_Calculator') else 0o644)<<16
   z.writestr(info,data);entries.append([n,'sha256='+base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode(),len(data)])
  entries.append([DIST+'/RECORD','','']);s=io.StringIO();csv.writer(s,lineterminator='\n').writerows(entries);z.writestr(DIST+'/RECORD',s.getvalue())
 return name
def build_sdist(sdist_directory,config_settings=None):
 # The pinned Linux runtime is an explicit offline build input, never fetched.
 runtime,manifest=engine('linux_x86_64');out=Path(sdist_directory);out.mkdir(parents=True,exist_ok=True)
 name=f'kaksturbo-{VERSION}.tar.gz'
 with tarfile.open(out/name,'x:gz') as t:
  for p in sorted(ROOT.rglob('*')):
   if not p.is_file() or any(v in p.relative_to(ROOT).parts for v in ('.git','engine','dist','__pycache__','.test-tmp')) or p.suffix=='.pyc':continue
   t.add(p,arcname=f'kaksturbo-{VERSION}/'+p.relative_to(ROOT).as_posix())
  for n in [*manifest['files'],'engine-assets.json']:t.add(runtime/n,arcname=f'kaksturbo-{VERSION}/engine/'+n)
  data=metadata('linux_x86_64',sdist=True)[DIST+'/METADATA'];info=tarfile.TarInfo(f'kaksturbo-{VERSION}/PKG-INFO');info.size=len(data);info.mode=0o644;t.addfile(info,io.BytesIO(data))
 return name
