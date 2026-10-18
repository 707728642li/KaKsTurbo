"""Fetch the exact qualified distributions; no build, credentials or fitting."""
from pathlib import Path
import hashlib,json,urllib.request
root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'packaging/pypi_artifacts.json').read_text())
out=root/'dist';out.mkdir(exist_ok=False)
assert manifest['version']=='1.1.0rc10'
assert len(manifest['artifacts'])==3
for item in manifest['artifacts']:
    name=item['name'];assert Path(name).name==name
    assert name.startswith('kaksturbo-'+manifest['version'])
    asset=item.get('asset_name',name);assert Path(asset).name==asset
    assert asset.startswith('kaksturbo-'+manifest['version'])
    assert item['url']=='https://github.com/707728642li/KaKsTurbo/releases/download/v1.1.0rc10/'+asset
    with urllib.request.urlopen(item['url'],timeout=120) as response:data=response.read()
    assert len(data)==item['bytes'],name
    assert hashlib.sha256(data).hexdigest()==item['sha256'],name
    (out/name).write_bytes(data)
    print(name+': SHA256 and size verified')
