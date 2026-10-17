"""Audit fixture hashes/input truths and optional native Q; never fit or use GPU."""
from __future__ import annotations
import argparse
from collections import Counter
import ctypes as C
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASES = "TCAG"
CODE1 = "FFLLSSSSYY!!CC!WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
CODONS64 = [a+b+c for a in BASES for b in BASES for c in BASES]
SENSE = [c for c, aa in zip(CODONS64, CODE1) if aa != "!"]
AA = [aa for aa in CODE1 if aa != "!"]
INDEX = {c: i for i, c in enumerate(SENSE)}
MODELS = ["JC", "F81", "K2P", "HKY", "TNEF", "TN", "K3P", "K3PUF", "TIMEF", "TIM", "TVMEF", "TVM", "SYM", "GTR"]
RATE_INDEX = {frozenset("TC"): 0, frozenset("AG"): 1, frozenset("TA"): 2,
              frozenset("CG"): 3, frozenset("TG"): 4, frozenset("CA"): 5}
ABS_TOL, REL_TOL = 1e-10, 1e-12


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expand(model, raw):
    family = MODELS.index(model) // 2
    if family == 0: return [1.] * 6
    if family == 1: return [raw[0], raw[0], 1., 1., 1., 1.]
    if family == 2: return [raw[0], raw[1], 1., 1., 1., 1.]
    if family == 3: return [raw[0], raw[0], raw[1], raw[1], 1., 1.]
    if family == 4: return [raw[0], raw[1], raw[2], raw[2], 1., 1.]
    if family == 5: return [raw[0], raw[0], raw[1], raw[2], raw[3], 1.]
    return list(raw) + [1.]


def independent_q(pi, rates, omega):
    """Directed full-matrix mathematical reconstruction, not source edge loop."""
    q, rs, ra0 = [[0.] * 61 for _ in range(61)], 0., 0.
    for i, ci in enumerate(SENSE):
        for j, cj in enumerate(SENSE):
            changed = [p for p in range(3) if ci[p] != cj[p]]
            if len(changed) != 1:
                continue
            p = changed[0]
            v = rates[RATE_INDEX[frozenset((ci[p], cj[p]))]] * pi[j]
            if AA[i] == AA[j]:
                rs += pi[i] * v
            else:
                ra0 += pi[i] * v
                v *= omega
            q[i][j] = v
        q[i][i] = -math.fsum(q[i])
    return q, rs, ra0, rs + omega * ra0


def inputs(a, b):
    if len(a) != len(b) or len(a) % 3:
        raise ValueError("Bad fixture codon input dimensions")
    clean_a, clean_b, masked = [], [], []
    for p in range(0, len(a), 3):
        x, y = a[p:p+3].upper(), b[p:p+3].upper()
        if x not in INDEX or y not in INDEX:
            masked.append(p // 3)
        else:
            clean_a.append(x); clean_b.append(y)
    patterns = Counter((max(INDEX[x], INDEX[y]), min(INDEX[x], INDEX[y])) for x, y in zip(clean_a, clean_b))
    snp = sum(x != y for a3, b3 in zip(clean_a, clean_b) for x, y in zip(a3, b3))
    joined = a+b
    gc = [sum(c in "GC" for c in joined) / len(joined)]
    gc += [sum(c in "GC" for c in joined[p::3]) / (len(joined)/3) for p in range(3)]
    return 3*len(clean_a), snp, masked, [[i,j,n] for (i,j), n in sorted(patterns.items())], gc


class Core:
    WIDTH = 11305
    def __init__(self, path):
        self.path = path.resolve()
        self.build_path = self.path.with_name("build.json")
        self.build = json.loads(self.build_path.read_text(encoding="utf-8-sig"))
        if sha(self.path) != self.build["dll_sha256"]:
            raise ValueError("Shared DLL differs from build fingerprint")
        self.dll = C.CDLL(str(self.path))
        I, D, B, V = C.POINTER(C.c_int32), C.POINTER(C.c_double), C.POINTER(C.c_int8), C.c_void_p
        self.dll.kaks_core_abi_version.restype = C.c_uint32
        self.dll.kaks_core_create_prepared_pair.argtypes = [C.c_char_p,I,D,B,B,D,C.c_uint32,D,C.POINTER(V),C.c_char_p,C.c_uint32]
        self.dll.kaks_core_create_prepared_pair.restype = C.c_int
        self.dll.kaks_core_q_snapshot.argtypes = [V,C.c_uint32,C.c_uint32,D,D,C.c_uint32,C.c_char_p,C.c_uint32]
        self.dll.kaks_core_q_snapshot.restype = C.c_int
        self.dll.kaks_core_free_prepared_pair.argtypes = [V]
        self.dll.kaks_core_free_prepared_pair.restype = None
        if self.dll.kaks_core_abi_version() != 1:
            raise ValueError("Unknown shared core ABI")

    def snapshot(self, case):
        s, truth = case["input_statistics"], case["truth"]
        patterns = s["patterns"]
        pi = [float.fromhex(v) for v in truth["stationary_pi61_hex"]]
        values = (C.c_double*66)(s["cleaned_nt_length"],s["observed_nucleotide_differences"],*pi,0.,0.,0.)
        meta = (C.c_int32*4)(0,len(patterns),s["cleaned_nt_length"]//3,0)
        a = (C.c_int8*len(patterns))(*(p[0] for p in patterns))
        b = (C.c_int8*len(patterns))(*(p[1] for p in patterns))
        w = (C.c_double*len(patterns))(*(p[2] for p in patterns))
        gc = (C.c_double*4)(*s["raw_GC_overall_and_positions"])
        ptr, error = C.c_void_p(), C.create_string_buffer(2048)
        code = self.dll.kaks_core_create_prepared_pair(case["id"].encode(),meta,values,a,b,w,len(patterns),gc,C.byref(ptr),error,len(error))
        if code: raise ValueError(error.value.decode())
        try:
            x = [truth["total_pair_path_length_x0"]] + truth["raw_exchangeabilities"] + [truth["process_omega"]]
            x += [0.]*(7-len(x))
            output = (C.c_double*self.WIDTH)()
            code = self.dll.kaks_core_q_snapshot(ptr,MODELS.index(case["generating_model"]),1,(C.c_double*7)(*x),output,self.WIDTH,error,len(error))
            if code: raise ValueError(error.value.decode())
            return list(output)
        finally:
            self.dll.kaks_core_free_prepared_pair(ptr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures",type=Path,default=ROOT/"Codex_test/publication_simulation_smoke_v2")
    parser.add_argument("--native-q",type=Path)
    parser.add_argument("--output",type=Path,default=ROOT/"Codex_test/publication_simulation_smoke_v2/independent_check.json")
    args = parser.parse_args()
    if args.output.exists(): parser.error("Preserve previous check; choose new output")
    fixture = args.fixtures.resolve()
    manifest = json.loads((fixture/"manifest.json").read_text())
    truths = json.loads((fixture/"truth.json").read_text())["cases"]
    issues, comparisons = [], []
    for name, fingerprint in manifest["files_sha256"].items():
        if sha(fixture/name) != fingerprint: issues.append("Hash mismatch: "+name)
    if sha(Path(manifest["generator"])) != manifest["generator_sha256"]:
        issues.append("Generator differs from captured fingerprint")
    records = {}
    for block in (fixture/"benchmark.axt").read_text().strip().split("\n\n"):
        name, a, b = block.splitlines()
        if name in records: issues.append("Duplicate AXT identifier: "+name)
        records[name]=(a,b)
    if set(records) != {c["id"] for c in truths} or len(truths)!=manifest["case_count"]:
        issues.append("AXT/truth/manifest coverage mismatch")
    core = Core(args.native_q) if args.native_q else None
    simulated_seeds = {}
    max_error = 0.
    for case in truths:
        name = case["id"]
        a,b = records[name]
        if hashlib.sha256(f"{name}\n{a}\n{b}\n\n".encode()).hexdigest()!=case["axt_record_sha256"]:
            issues.append(name+": record hash mismatch")
        nt,snp,masked,patterns,gc=inputs(a,b)
        s=case["input_statistics"]
        if (nt,snp,masked,patterns,gc)!=(s["cleaned_nt_length"],s["observed_nucleotide_differences"],s["masked_codons"],s["patterns"],s["raw_GC_overall_and_positions"]):
            issues.append(name+": independent mask/pattern/SNP/GC mismatch")
        if case.get("reverse_of"):
            if records[case["reverse_of"]]!=(b,a): issues.append(name+": reversal mismatch")
            parent=next(c for c in truths if c["id"]==case["reverse_of"])
            if parent["split"]!=case["split"]: issues.append(name+": reversed family crossed splits")
        elif case["truth"]:
            seed=case["generator_seed"]
            if seed in simulated_seeds: issues.append(name+": duplicate independent simulation seed")
            simulated_seeds[seed]=case["split"]
        t=case["truth"]
        if not t: continue
        pi=[float.fromhex(v) for v in t["stationary_pi61_hex"]]
        rates=expand(case["generating_model"],t["raw_exchangeabilities"])
        if rates!=t["expanded_exchangeabilities_TC_AG_TA_CG_TG_CA"]: issues.append(name+": pure-rate truth mismatch")
        q,rs,ra0,mr=independent_q(pi,rates,t["process_omega"])
        ds=t["total_pair_path_length_x0"]*(rs+ra0)/(3*mr)
        dn=ds*t["process_omega"]
        if not math.isclose(ds,t["expected_dS_per_unselected_synonymous_site"],rel_tol=1e-12,abs_tol=1e-14) or not math.isclose(dn,t["expected_dN_per_unselected_nonsynonymous_site"],rel_tol=1e-12,abs_tol=1e-14):
            issues.append(name+": independent site-rate truth mismatch")
        if core:
            out=core.snapshot(case)
            expected=sum(q,[])
            sqrtpi=[math.sqrt(v) for v in pi]
            sym=[[q[i][j]*sqrtpi[i]/sqrtpi[j] if i>j else 0. for j in range(61)] for i in range(61)]
            for i in range(61):
                sym[i][i]=q[i][i]
                for j in range(i): sym[j][i]=sym[i][j]
            expected+=sum(sym,[])
            expected+=[v*t["total_pair_path_length_x0"]/mr for row in sym for v in row]
            expected+=pi+[0.]*3+sqrtpi+rates
            expected+=[rs,ra0,ra0*t["process_omega"],mr,t["process_omega"],t["process_omega"] if not t["raw_exchangeabilities"] else t["raw_exchangeabilities"][0],t["total_pair_path_length_x0"],61.,len(t["raw_exchangeabilities"])+2.,1.,0.]
            local=max(abs(x-y) for x,y in zip(expected,out))
            max_error=max(max_error,local)
            bad=[i for i,(x,y) in enumerate(zip(expected,out)) if not math.isfinite(y) or abs(x-y)>max(ABS_TOL,REL_TOL*abs(x))]
            if len(expected)!=Core.WIDTH or bad: issues.append(name+": independent/shared Q/S/A/scalars mismatch")
            comparisons.append({"case":name,"width":len(expected),"max_absolute_error":local,"mismatching_indices":bad})
    report={"schema":"kakscuda.publication_simulation_audit/1","passed":not issues,"issues":issues,
            "fixtures":str(fixture),"case_count":len(truths),"independent_simulation_families":len(simulated_seeds),
            "manifest_sha256":sha(fixture/"manifest.json"),"check_script_sha256":sha(Path(__file__)),
            "independent_Q_method":"Full directed 61x61 pairs, separate synonymous/unselected nonsynonymous stationary sums, fsum diagonal; not native/source edge-order arithmetic.",
            "native_Q_test":None if core is None else {"dll":str(core.path),"dll_sha256":sha(core.path),"build_sha256":sha(core.build_path),"comparisons":comparisons,"max_absolute_error":max_error,"absolute_tolerance":ABS_TOL,"relative_tolerance":REL_TOL},
            "fitting_performed":False,"GPU_used":False,
            "limitations":"Q/truth/input validation only; no exponential, sampler distribution, optimizer, uncertainty or statistical accuracy evidence. Native test uses specified truth pi, not the estimator's sample-reestimated pi."}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({k:report[k] for k in ("passed","case_count","independent_simulation_families","issues")}))
    return 0 if report["passed"] else 1


if __name__=="__main__":
    raise SystemExit(main())
