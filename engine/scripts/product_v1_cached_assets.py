"""Historical build identities; manifest-pinned data, no compiler loading."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
def historical_compiler_sha(path):
    name=Path(path).name
    records=json.loads((ROOT/'compiler-provenance.json').read_text(encoding='utf-8'))['files']
    if name not in records:raise ValueError('Unknown historical compiler identity: '+name)
    return records[name]
