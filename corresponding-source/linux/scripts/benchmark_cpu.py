"""Frozen MA sampling, native Windows CPU benchmark, and 22-column validation.

Only Python's standard library is required. Original data are read-only.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import ctypes
from ctypes import wintypes
from decimal import Decimal, InvalidOperation
import hashlib
import io
import itertools
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import struct
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data/Prunus_Anthocyanin_KaKs_30662_20261010_190300"
DEFAULT_SAMPLE = ROOT / "Codex_test/benchmark_1000"
SCHEMA = [
    "Sequence", "Method", "Ka", "Ks", "Ka/Ks", "P-Value(Fisher)", "Length",
    "S-Sites", "N-Sites", "Fold-Sites(0:2:4)", "Substitutions",
    "S-Substitutions", "N-Substitutions", "Fold-S-Substitutions(0:2:4)",
    "Fold-N-Substitutions(0:2:4)", "Divergence-Time",
    "Substitution-Rate-Ratio(rTC:rAG:rTA:rCG:rTG:rCA/rCA)", "GC(1:2:3)",
    "ML-Score", "AICc", "Akaike-Weight", "Model",
]
NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")
POLICY = {
    "schema_columns": SCHEMA,
    "exact_fields": ["Sequence", "Method", "Length", "Model"],
    "na": "NA must match NA exactly; missing, failure, and nonfinite are distinct.",
    "numeric_rule": "Every numeric token, including composite fields, is compared. "
                    "Tolerance = 1.000001 * 10^(floor(log10(max(abs(reference),abs(candidate))))-5). "
                    "This is one unit of the printed six-significant-digit precision. "
                    "Zero versus zero has tolerance zero; no absolute tolerance floor.",
    "composite_rule": "Punctuation and numeric token count must match exactly.",
    "threshold_rule": "Any change in Ka/Ks classification (<1, =1, >1) is reported and fails acceptance.",
    "missing_duplicate_extra": "Every requested pair must occur exactly once; any mismatch fails.",
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def dump(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def tsv_bytes(rows, fields) -> bytes:
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue().encode("utf-8")


def table(path: Path, schema=None):
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if schema is not None and reader.fieldnames != schema:
            raise ValueError(f"Schema differs from the required 22 columns: {path}")
        rows = list(reader)
        if any(None in r or any(v is None for v in r.values()) for r in rows):
            raise ValueError(f"Malformed TSV row: {path}")
        return rows


def axt_records(data: bytes):
    if b"\r" in data or not data.endswith(b"\n\n"):
        raise ValueError("AXT must preserve ASCII/LF and a final empty separator line")
    records = {}
    for part in data.split(b"\n\n"):
        if not part:
            continue
        header, sep, body = part.partition(b"\n")
        pair = header.decode("ascii")
        if not sep or pair in records:
            raise ValueError(f"Malformed or duplicated AXT pair: {pair}")
        lines = body.splitlines()
        if len(lines) % 2 or not lines:
            raise ValueError(f"Unbalanced wrapped AXT sequences: {pair}")
        a = b"".join(lines[:len(lines)//2])
        b = b"".join(lines[len(lines)//2:])
        if len(a) != len(b) or len(a) % 3 or set(a+b) - set(b"ACGTN"):
            raise ValueError(f"Invalid codon-aligned AXT: {pair}")
        records[pair] = part + b"\n\n"
    return records


def keyed(seed, pair):
    return sha(f"KaKsCUDA-sample-v1:{seed}:{pair}".encode("ascii"))


def prepare(args):
    data, dest = args.data.resolve(), args.sample.resolve()
    if dest.exists():
        raise ValueError(f"Sample destination exists; use its manifest or choose a new directory: {dest}")
    index = table(data / "input/pair_index.tsv")
    refs = table(data / "cpu_reference/MA_snapshot.tsv", SCHEMA)
    byref = {r["Sequence"]: r for r in refs}
    byindex = {r["pair_id"]: r for r in index}
    if len(byref) != len(refs) or len(byindex) != len(index) or set(byref) != set(byindex):
        raise ValueError("Source input/reference coverage is not one-to-one")
    records = axt_records((data / "input/all_pairs.axt").read_bytes())
    if set(records) != set(byindex):
        raise ValueError("Merged AXT differs from the pair index")
    if not 16 <= args.count <= len(index):
        raise ValueError("Prepare requires 16 <= count <= number of source pairs")
    ordered = sorted(index, key=lambda r: (int(r["aligned_nt_per_sequence"]), int(r["pair_index"])))
    cells, cellof = {}, {}
    for rank, row in enumerate(ordered):
        p = row["pair_id"]
        status = "NA" if byref[p]["Ka/Ks"] == "NA" else ("gt1" if Decimal(byref[p]["Ka/Ks"]) > 1 else "finite_le1")
        cell = f"length_decile_{min(9,rank*10//len(index))}:{status}"
        cells.setdefault(cell, []).append(p)
        cellof[p] = cell
    mandatory, priority, reasons = set(), [], {}

    def force(p, reason, reverse=True):
        if p not in mandatory:
            mandatory.add(p)
            priority.append(p)
        reasons.setdefault(p, []).append(reason)
        if reverse:
            row = byindex[p]
            rev = f"{row['gene_b']}-{row['gene_a']}"
            if rev != p and rev in byindex:
                force(rev, f"reverse_of:{p}", False)

    # Priority ensures a 16-pair smoke run contains the rare input ambiguity cases.
    n_pairs = [p for p in records if b"N" in records[p].partition(b"\n")[2]]
    for p in sorted(n_pairs, key=lambda p: keyed(args.seed, p)):
        force(p, "contains_N")
    force(ordered[0]["pair_id"], "shortest_alignment")
    force(ordered[-1]["pair_id"], "longest_alignment")
    nas = [p for p in byref if byref[p]["Ka/Ks"] == "NA"]
    if nas:
        force(min(nas, key=lambda p: keyed(args.seed,p)), "NA_result")
    finite = sorted((p for p in byref if byref[p]["Ka/Ks"] != "NA"), key=lambda p: (Decimal(byref[p]["Ka/Ks"]), p))
    force(finite[0], "minimum_finite_ratio")
    force(finite[-1], "maximum_finite_ratio")
    below = [p for p in finite if Decimal(byref[p]["Ka/Ks"]) < 1]
    above = [p for p in finite if Decimal(byref[p]["Ka/Ks"]) > 1]
    for p in below[-2:] + above[:2]:
        force(p, "nearest_ratio_to_1")
    reverse_groups = sorted((p for p,r in byindex.items() if p < f"{r['gene_b']}-{r['gene_a']}" and f"{r['gene_b']}-{r['gene_a']}" in byindex), key=lambda p:keyed(args.seed,p))
    for p in reverse_groups[:8]:
        force(p, "reverse_group_coverage")
    if len(mandatory) > args.count:
        raise ValueError(f"Mandatory coverage needs {len(mandatory)} pairs; prepare the full 1000 and use run --count 16 for smoke")
    quotas = {c: sum(p in mandatory for p in members) for c,members in cells.items()}
    expected = {c: args.count*len(members)/len(index) for c,members in cells.items()}
    # Largest remaining population deficit, with mandatory minima. Deterministic ties.
    for _ in range(args.count-len(mandatory)):
        c = min((c for c in cells if quotas[c] < len(cells[c])), key=lambda c: (-(expected[c]-quotas[c]), c))
        quotas[c] += 1
    selected = set(mandatory)
    for c,members in cells.items():
        candidates = sorted((p for p in members if p not in mandatory), key=lambda p:keyed(args.seed,p))
        selected.update(candidates[:quotas[c]-sum(p in mandatory for p in members)])
    selected_rows = [r for r in index if r["pair_id"] in selected]
    if len(selected_rows) != args.count:
        raise AssertionError("Sample allocation failed")
    for r in selected_rows:
        if sha(records[r["pair_id"]]) != r["axt_sha256"]:
            raise ValueError(f"Original pair bytes/hash differ: {r['pair_id']}")
    dest.mkdir(parents=True)
    (dest/"benchmark.axt").write_bytes(b"".join(records[r["pair_id"]] for r in selected_rows))
    (dest/"reference.tsv").write_bytes(tsv_bytes([byref[r["pair_id"]] for r in selected_rows],SCHEMA))
    (dest/"pairs.tsv").write_bytes(tsv_bytes(selected_rows,list(index[0])))
    priority += [r["pair_id"] for r in selected_rows if r["pair_id"] not in mandatory]
    manifest = {
        "schema": "kakscuda.frozen_benchmark/1", "created_utc":datetime.now(timezone.utc).isoformat(),
        "source_dataset":str(data), "count":args.count, "seed":args.seed,
        "method":"MA", "genetic_code":1, "direction":"Original directed requests, never deduplicated",
        "sampling_rule":"Rank by original aligned length, break ties by original request index, form ten equal population deciles; cross with NA, Ka/Ks>1, other. Allocate proportional population quotas by largest remaining deficit with mandatory coverage minima. Within cells select the smallest SHA256(KaKsCUDA-sample-v1:seed:pair_id). Output retains original request order.",
        "mandatory_rule":"All N-bearing inputs and their available reverse requests; shortest/longest alignment; one NA; minimum/maximum finite Ka/Ks; two finite ratios on each side nearest 1; eight deterministic reverse groups. Mandatory minima can slightly alter proportional quotas, reported below.",
        "mandatory_reasons":reasons, "smoke_priority_ids":priority,
        "strata":{c:{"population":len(cells[c]),"expected":expected[c],"selected":quotas[c]} for c in sorted(cells)},
        "population_statistics":{"pairs":len(index), "N_pairs":len(n_pairs),"N_characters":sum(r.partition(b"\n")[2].count(b"N") for r in records.values()),"NA_pairs":len(nas),"gt1_pairs":len(above),"reverse_requests":2*len(reverse_groups)},
        "sample_statistics":{"N_pairs":sum(p in selected for p in n_pairs),"NA_pairs":sum(p in selected for p in nas),"gt1_pairs":sum(p in selected for p in above),"reverse_requests":sum(f"{byindex[p]['gene_b']}-{byindex[p]['gene_a']}" in selected for p in selected),"mandatory_pairs":len(mandatory)},
        "validation_policy":POLICY,
        "source_files":{str(p.relative_to(data)):file_sha(p) for p in [data/"input/all_pairs.axt",data/"input/pair_index.tsv",data/"cpu_reference/MA_snapshot.tsv",data/"manifest.json"]},
        "files":{name:{"bytes":(dest/name).stat().st_size,"sha256":file_sha(dest/name)} for name in ("benchmark.axt","reference.tsv","pairs.tsv")},
    }
    dump(dest/"manifest.json",manifest)
    print(json.dumps({"sample":str(dest),"count":args.count,"statistics":manifest["sample_statistics"]},ensure_ascii=False))


def kernel32():
    if sys.platform != "win32" or ctypes.sizeof(ctypes.c_void_p) != 8:
        raise RuntimeError("This benchmark requires native 64-bit Windows Python")
    k = ctypes.WinDLL("kernel32",use_last_error=True)
    k.GetLogicalProcessorInformationEx.argtypes = [wintypes.DWORD,ctypes.c_void_p,ctypes.POINTER(wintypes.DWORD)]
    k.GetLogicalProcessorInformationEx.restype = wintypes.BOOL
    k.GetCurrentProcess.restype = wintypes.HANDLE
    k.GetProcessAffinityMask.argtypes = [wintypes.HANDLE,ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t)]
    k.GetProcessAffinityMask.restype = wintypes.BOOL
    k.SetProcessAffinityMask.argtypes = [wintypes.HANDLE,ctypes.c_size_t]
    k.SetProcessAffinityMask.restype = wintypes.BOOL
    return k


def affinity(k, handle):
    process, system = ctypes.c_size_t(),ctypes.c_size_t()
    if not k.GetProcessAffinityMask(handle,ctypes.byref(process),ctypes.byref(system)):
        raise ctypes.WinError(ctypes.get_last_error())
    return process.value,system.value


def topology(k):
    size=wintypes.DWORD()
    k.GetLogicalProcessorInformationEx(0,None,ctypes.byref(size))
    buf=ctypes.create_string_buffer(size.value)
    if not k.GetLogicalProcessorInformationEx(0,buf,ctypes.byref(size)):
        raise ctypes.WinError(ctypes.get_last_error())
    raw, offset, cores = buf.raw,0,[]
    while offset < size.value:
        relationship, entry_size=struct.unpack_from("<II",raw,offset)
        if entry_size < 32 or offset+entry_size > size.value:
            raise RuntimeError("Malformed Windows processor topology")
        groups=struct.unpack_from("<H",raw,offset+30)[0]
        masks=[]
        for j in range(groups):
            mask, group=struct.unpack_from("<QH",raw,offset+32+16*j)
            masks.append({"group":group,"mask":mask,"logical_processors":[i for i in range(64) if mask&(1<<i)]})
        cores.append({"physical_core_index":len(cores),"smt":bool(raw[offset+8]&1),"efficiency_class":raw[offset+9],"groups":masks})
        offset+=entry_size
    inherited,system=affinity(k,k.GetCurrentProcess())
    eligible=[c for c in cores if len(c["groups"])==1 and c["groups"][0]["group"]==0 and len(c["groups"][0]["logical_processors"])==2 and (c["groups"][0]["mask"]&inherited)==c["groups"][0]["mask"]]
    if len(eligible)<8:
        raise RuntimeError(f"Only {len(eligible)} available physical cores with both SMT threads in processor group 0; cannot claim 8 cores/16 threads")
    chosen=eligible[:8]
    mask=sum(c["groups"][0]["mask"] for c in chosen)
    if mask.bit_count()!=16:
        raise RuntimeError("Selected topology is not 16 distinct logical processors")
    return {"all_physical_cores":cores,"selected_physical_cores":chosen,"selected_mask":mask,"selected_mask_hex":hex(mask),"original_parent_mask":inherited,"system_mask":system,"physical_core_count":8,"logical_processor_count":16}


def validate(reference, candidate_paths, requested):
    issues, issue_count, seen, rows = [],0,{},[]
    errors, na_mismatches, threshold_changes = {},0,[]
    def issue(**record):
        nonlocal issue_count
        issue_count+=1
        if len(issues)<5000:
            issues.append(record)
    for path in candidate_paths:
        if not path.exists():
            issue(kind="missing_output",path=str(path)); continue
        try:
            loaded=table(path,SCHEMA)
        except (ValueError,UnicodeError) as e:
            issue(kind="schema_or_format",path=str(path),detail=str(e)); continue
        for r in loaded:
            p=r["Sequence"]
            if p in seen:
                issue(kind="duplicate_pair",pair=p)
            seen[p]=seen.get(p,0)+1
            rows.append(r)
    required=set(requested)
    for p in sorted(required-set(seen)): issue(kind="missing_pair",pair=p)
    for p in sorted(set(seen)-required): issue(kind="unexpected_pair",pair=p)
    matched={r["Sequence"]:r for r in rows if r["Sequence"] in required}
    for p in requested:
        if p not in matched: continue
        expected,observed=reference[p],matched[p]
        for field in SCHEMA:
            x,y=expected[field],observed[field]
            if x=="NA" or y=="NA":
                if x!=y:
                    na_mismatches+=1; issue(kind="NA_mismatch",pair=p,column=field,reference=x,candidate=y)
                continue
            if field in POLICY["exact_fields"]:
                if x!=y: issue(kind="exact_mismatch",pair=p,column=field,reference=x,candidate=y)
                continue
            xs,ys=NUMBER.findall(x),NUMBER.findall(y)
            if NUMBER.sub("#",x)!=NUMBER.sub("#",y) or len(xs)!=len(ys) or not xs:
                issue(kind="numeric_structure_or_nonfinite",pair=p,column=field,reference=x,candidate=y); continue
            for token,(a,b) in enumerate(zip(xs,ys)):
                try:
                    da,db=Decimal(a),Decimal(b)
                except InvalidOperation:
                    issue(kind="invalid_number",pair=p,column=field); continue
                delta=abs(da-db); scale=max(abs(da),abs(db))
                tolerance=Decimal("1.000001")*Decimal(10)**(scale.adjusted()-5) if scale else Decimal(0)
                relative=delta/abs(da) if da else (Decimal(0) if not db else None)
                key=f"{field}[{token}]" if len(xs)>1 else field
                stats=errors.setdefault(key,{"compared":0,"max_absolute_error":"0","max_relative_error":"0","max_absolute_pair":None,"max_relative_pair":None})
                stats["compared"]+=1
                if delta>Decimal(stats["max_absolute_error"]):
                    stats["max_absolute_error"]=str(delta); stats["max_absolute_pair"]=p
                if relative is not None and relative>Decimal(stats["max_relative_error"]):
                    stats["max_relative_error"]=str(relative); stats["max_relative_pair"]=p
                if delta>tolerance:
                    issue(kind="numeric_mismatch",pair=p,column=field,token=token,reference=a,candidate=b,absolute_error=str(delta),tolerance=str(tolerance))
            if field=="Ka/Ks":
                a,b=Decimal(x),Decimal(y)
                if (a>1)-(a<1)!=(b>1)-(b<1):
                    threshold_changes.append({"pair":p,"reference":x,"candidate":y})
                    issue(kind="KaKs_threshold_classification_change",pair=p,reference=x,candidate=y)
    return {"passed":issue_count==0,"requested_pairs":len(requested),"output_rows":len(rows),"unique_output_pairs":len(seen),"issue_count":issue_count,"issues":issues,"issues_truncated":issue_count>len(issues),"NA_mismatches":na_mismatches,"KaKs_threshold_changes":threshold_changes,"numeric_errors":errors},matched


def schedule_shards(requested, byindex, mode="fixed", chunks=64):
    """Balance estimated aligned length; dynamic mode dispatches extra chunks.

    The fixed default is exactly the original 16-shard allocation. Dynamic
    chunks use the same inputs but a free CLI slot takes the next chunk. The
    estimator does not know optimizer duration or read any reference values.
    """
    if mode not in ("fixed", "dynamic") or chunks < 1:
        raise ValueError("Invalid scheduling mode/chunk count")
    shard_count=16 if mode=="fixed" else min(chunks,len(requested))
    shards=[[] for _ in range(shard_count)]; loads=[0]*shard_count
    for p in sorted(requested,key=lambda p:(-int(byindex[p]["aligned_nt_per_sequence"]),p)):
        slot=min(range(shard_count),key=lambda i:(loads[i],i))
        shards[slot].append(p); loads[slot]+=int(byindex[p]["aligned_nt_per_sequence"])
    return shards


def run(args):
    sample=args.sample.resolve(); manifest=json.loads((sample/"manifest.json").read_text(encoding="utf-8"))
    if manifest["validation_policy"]!=POLICY:
        raise RuntimeError("Sample validation policy differs from the fixed harness policy")
    for name,meta in manifest["files"].items():
        if file_sha(sample/name)!=meta["sha256"]:
            raise RuntimeError(f"Frozen sample hash mismatch: {name}")
    index=table(sample/"pairs.tsv"); refrows=table(sample/"reference.tsv",SCHEMA)
    refs={r["Sequence"]:r for r in refrows}
    count=args.count or manifest["count"]
    if not 1<=count<=manifest["count"] or args.repeats<1 or args.chunks<1:
        raise ValueError("Invalid count or repeats")
    chosen=set(manifest["smoke_priority_ids"][:count]) if count<manifest["count"] else set(refs)
    requested=[r["pair_id"] for r in index if r["pair_id"] in chosen]
    byindex={r["pair_id"]:r for r in index}
    exe=args.exe.resolve()
    if not exe.is_file(): raise FileNotFoundError(exe)
    dest=args.output.resolve() if args.output else ROOT/"Codex_test"/("cpu_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    if dest.exists(): raise ValueError(f"Run output already exists: {dest}")
    dest.mkdir(parents=True)
    k=kernel32(); topo=topology(k); handle=k.GetCurrentProcess(); mask=topo["selected_mask"]
    if not k.SetProcessAffinityMask(handle,mask): raise ctypes.WinError(ctypes.get_last_error())
    if affinity(k,handle)[0]!=mask: raise RuntimeError("Parent affinity verification failed")
    info={"schema":"kakscuda.cpu_benchmark/1","created_utc":datetime.now(timezone.utc).isoformat(),"platform":platform.platform(),"python":sys.version,"native_windows":True,"executable":str(exe),"executable_sha256":file_sha(exe),"workers":16,"method":"MA","genetic_code":1,"topology":topo,"sample_manifest":str(sample/"manifest.json"),"sample_manifest_sha256":file_sha(sample/"manifest.json"),"count":count,"repeats":args.repeats,"timing_scope":"From reading frozen input to final merged output write: parsing, shard writing, dispatch/process startup, original CLI parsing/computation/output, output read and final merge/write. Reference loading, selection, hash verification and numerical validation are outside the timer.","validation_policy":POLICY,"runs":[]}
    info["scheduling_mode"]=args.scheduling
    info["requested_chunks"]=args.chunks if args.scheduling=="dynamic" else 16
    info["maximum_parallel_cli_processes"]=16
    info["scheduling_rule"]="Longest aligned input first, greedily balance aligned length across chunks; 16 process slots take queued chunks when free. Each chunk starts an independent original CLI." if args.scheduling=="dynamic" else "Original fixed 16 length-balanced shards, one independent original CLI per nonempty shard."
    build=exe.parent/"build.json"
    if build.exists(): info["build_provenance"]=json.loads(build.read_text(encoding="utf-8-sig"))
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            info["cpu_model"]=winreg.QueryValueEx(key,"ProcessorNameString")[0].strip()
    except (ImportError,OSError): pass
    dump(dest/"run_manifest.json",info)
    overall_pass=True
    try:
        for repeat in range(args.repeats):
            rd=dest/f"repeat_{repeat+1:02d}"; rd.mkdir()
            start=time.perf_counter()
            records=axt_records((sample/"benchmark.axt").read_bytes())
            chosen_input=b"".join(records[p] for p in requested)
            (rd/"requested.axt").write_bytes(chosen_input)
            shards=schedule_shards(requested,byindex,args.scheduling,args.chunks)
            jobs=[]
            for slot,pairs in enumerate(shards):
                if not pairs: continue
                prefix=f"chunk_{slot:04d}" if args.scheduling=="dynamic" else f"worker_{slot:02d}"
                inp=rd/(prefix+".axt"); out=rd/(prefix+".tsv"); log=rd/(prefix+".log")
                inp.write_bytes(b"".join(records[p] for p in pairs))
                jobs.append((slot,pairs,inp,out,log))
            dispatch=time.perf_counter()
            worker_local=threading.local()
            slot_counter=itertools.count()
            def initialize_slot():
                worker_local.slot=next(slot_counter)
            def worker(job):
                slot,pairs,inp,out,log=job
                begin=time.perf_counter()
                with log.open("wb") as stream:
                    proc=subprocess.Popen([str(exe),"-i",str(inp),"-o",str(out),"-m","MA","-c","1"],stdout=stream,stderr=subprocess.STDOUT)
                    inherited,system=affinity(k,int(proc._handle))
                    if inherited!=mask:
                        # Parent was restricted before spawning; fail rather than hide an affinity error.
                        proc.wait()
                        raise RuntimeError(f"Worker {slot} inherited unexpected affinity {hex(inherited)}")
                    code=proc.wait()
                return {"worker":slot,"task":slot,"process_slot":worker_local.slot,"pid":proc.pid,"pairs":len(pairs),"affinity_mask":inherited,"affinity_hex":hex(inherited),"affinity_verified":True,"exit_code":code,"wall_seconds":time.perf_counter()-begin,"queue_wait_seconds":begin-dispatch,"output":str(out),"log":str(log)}
            with concurrent.futures.ThreadPoolExecutor(max_workers=16,initializer=initialize_slot) as pool:
                workers=list(pool.map(worker,jobs))
            computation_done=time.perf_counter()
            # Parsing and merging the CLI output are included in the benchmark wall time.
            merged=[]; formats=[]
            for _,_,_,out,_ in jobs:
                try: merged.extend(table(out,SCHEMA))
                except (OSError,ValueError,UnicodeError) as e: formats.append(str(e))
            order={p:i for i,p in enumerate(requested)}
            merged.sort(key=lambda r:(order.get(r["Sequence"],len(order)),r["Sequence"]))
            output=rd/"merged.MA.tsv"; output.write_bytes(tsv_bytes(merged,SCHEMA))
            finish=time.perf_counter()
            validation,_=validate(refs,[job[3] for job in jobs],requested)
            if any(w["exit_code"]!=0 for w in workers) or formats:
                validation["passed"]=False
                validation["process_or_format_errors"]=formats+[w for w in workers if w["exit_code"]!=0]
            (rd/"reference.tsv").write_bytes(tsv_bytes([refs[p] for p in requested],SCHEMA))
            dump(rd/"validation.json",validation)
            report={"repeat":repeat+1,"first_run":repeat==0,"wall_seconds":finish-start,"read_parse_shard_seconds":dispatch-start,"process_seconds":computation_done-dispatch,"merge_write_seconds":finish-computation_done,"pairs":count,"pairs_per_second":count/(finish-start),"input_sha256":sha(chosen_input),"output_sha256":file_sha(output),"workers":workers,"validation_passed":validation["passed"],"validation_report":str(rd/"validation.json")}
            report["scheduling_mode"]=args.scheduling
            report["task_count"]=len(jobs)
            report["process_slots_used"]=len({w["process_slot"] for w in workers})
            dump(rd/"timing.json",report); info["runs"].append(report)
            overall_pass &= validation["passed"]
            dump(dest/"run_manifest.json",info)
            print(json.dumps({"repeat":repeat+1,"pairs":count,"wall_seconds":report["wall_seconds"],"validation_passed":validation["passed"],"directory":str(rd)},ensure_ascii=False),flush=True)
    finally:
        if not k.SetProcessAffinityMask(handle,topo["original_parent_mask"]):
            raise ctypes.WinError(ctypes.get_last_error())
    durations=[r["wall_seconds"] for r in info["runs"]]
    info["all_validation_passed"]=overall_pass
    info["median_wall_seconds"]=statistics.median(durations)
    info["first_wall_seconds"]=durations[0]
    info["subsequent_median_wall_seconds"]=statistics.median(durations[1:]) if len(durations)>1 else None
    info["warmup_note"]="Every CLI is a fresh process. No unmeasured warm-up; first and subsequent runs are separate. File cache state is not forcibly reset."
    dump(dest/"run_manifest.json",info)
    print(json.dumps({"directory":str(dest),"passed":overall_pass,"median_wall_seconds":info["median_wall_seconds"]},ensure_ascii=False))
    return 0 if overall_pass else 2


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    subs=parser.add_subparsers(dest="command",required=True)
    prep=subs.add_parser("prepare",help="Freeze representative sample and all validation rules before timing")
    prep.add_argument("--data",type=Path,default=DEFAULT_DATA)
    prep.add_argument("--sample",type=Path,default=DEFAULT_SAMPLE)
    prep.add_argument("--count",type=int,default=1000)
    prep.add_argument("--seed",type=int,default=20261010)
    bench=subs.add_parser("run",help="Run original Windows CLI on verified 8-core/16-thread affinity")
    bench.add_argument("--sample",type=Path,default=DEFAULT_SAMPLE)
    bench.add_argument("--exe",type=Path,default=ROOT/"build/cpu-reference/KaKs_Calculator.exe")
    bench.add_argument("--count",type=int,default=None)
    bench.add_argument("--repeats",type=int,default=1)
    bench.add_argument("--scheduling",choices=("fixed","dynamic"),default="fixed",help="Default retains original fixed 16 shards; dynamic dispatches queued chunks to 16 CLI slots")
    bench.add_argument("--chunks",type=int,default=64,help="Number of length-balanced chunks for dynamic mode, capped at pair count")
    bench.add_argument("--output",type=Path)
    args=parser.parse_args()
    try:
        return prepare(args) or 0 if args.command=="prepare" else run(args)
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}",file=sys.stderr)
        return 1


if __name__=="__main__":
    sys.exit(main())
