"""Freeze local Python import closure and used binary build records before runs."""
from __future__ import annotations
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
from product_v1_platform import build_record, native_path, snapshot_directory
from product_v1_engine_runtime import platform_build_metadata
ROOT=Path(__file__).resolve().parents[1]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def source_snapshot(destination,seeds):
    destination=Path(destination);destination.mkdir();pending=list(seeds);seen=set();sources={}
    while pending:
        name=pending.pop()
        if name in seen:continue
        path=ROOT/'scripts'/name
        if not path.is_file():raise FileNotFoundError(path)
        seen.add(name);data=path.read_bytes();(destination/name).write_bytes(data);sources[name]=sha(path)
        tree=ast.parse(data,filename=str(path))
        for node in ast.walk(tree):
            modules=[]
            if isinstance(node,ast.Import):modules=[a.name.split('.')[0] for a in node.names]
            if isinstance(node,ast.ImportFrom) and node.module:modules=[node.module.split('.')[0]]
            for module in modules:
                filename=module+'.py'
                if (ROOT/'scripts'/filename).is_file():pending.append(filename)
    return {'Python_static_local_import_closure_sha256':sources,'dynamic_local_imports_executed':False,
        'closure_scope':'All local static imports, including function-local imports; external dependencies recorded by runtime/binary metadata'}
def binary_snapshot(destination,path,label):
    path=native_path(path).resolve();metadata=platform_build_metadata(path)
    expected=metadata.get('dll_sha256',metadata.get('executable_sha256'))
    if expected is not None and sha(path)!=expected:raise ValueError('Binary differs from build manifest: '+str(path))
    record=build_record(path)
    output=Path(destination)/label;output.mkdir();shutil.copy2(record,output/record.name)
    snapshot=snapshot_directory(path)
    if snapshot.is_dir():shutil.copytree(snapshot,output/'compiled_source_snapshot')
    return {'path':str(path),'sha256':sha(path),'build':metadata,'build_record_filename':record.name,'snapshot_present':snapshot.is_dir()}

def verify_source_snapshot(snapshot):
    changes=[]
    for name,expected in snapshot['Python_static_local_import_closure_sha256'].items():
        path=ROOT/'scripts'/name
        actual=sha(path) if path.is_file() else None
        if actual!=expected:changes.append({'source':name,'before':expected,'after':actual})
    return {'PASS':not changes,'changed_sources':changes,'scope':'Compare current source bytes to pre-run captured static local import closure; no retroactive guarantee for earlier runs'}
