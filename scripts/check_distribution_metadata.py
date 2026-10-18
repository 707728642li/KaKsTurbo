"""PEP 639 License-File paths resolve differently in wheels and source archives."""
from pathlib import Path
import argparse,email.parser,hashlib,json,tarfile,zipfile
p=argparse.ArgumentParser();p.add_argument('files',nargs='+');p.add_argument('--output',required=True);a=p.parse_args()
checks=[]
for arg in a.files:
    path=Path(arg)
    if path.suffix=='.whl':
        with zipfile.ZipFile(path) as z:
            names=z.namelist();meta=next(n for n in names if n.endswith('.dist-info/METADATA'));data=z.read(meta).decode();prefix=meta.rsplit('/',1)[0]+'/licenses/'
    else:
        with tarfile.open(path) as t:
            names=t.getnames();meta=next(n for n in names if n.count('/')==1 and n.endswith('/PKG-INFO'));data=t.extractfile(meta).read().decode();prefix=meta.rsplit('/',1)[0]+'/'
    parsed=email.parser.Parser().parsestr(data);declared=parsed.get_all('License-File');assert declared
    missing=[prefix+n for n in declared if prefix+n not in names]
    rec=dict(name=path.name,license_paths=len(declared),missing=missing,PASS=not missing,sha256=hashlib.sha256(path.read_bytes()).hexdigest());checks.append(rec)
Path(a.output).write_text(json.dumps(dict(PASS=all(x['PASS'] for x in checks),artifacts=checks),indent=2)+'\n')
print(json.dumps(checks));assert all(x['PASS'] for x in checks)
