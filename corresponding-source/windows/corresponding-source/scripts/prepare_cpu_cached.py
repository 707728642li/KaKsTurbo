"""Prepare a narrow original-source CLI exact-cache variant, never in tools/."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"tools/kakscalculator2-2.0.1/src"
DEST=ROOT/"build/cpu-reference-cached-src"


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(text,old,new):
    if text.count(old)!=1: raise RuntimeError(f"Unexpected source marker count for {old!r}")
    return text.replace(old,new,1)


def main():
    originals={p.name:sha(p) for p in SOURCE.iterdir() if p.is_file()}
    cpp=(SOURCE/"GY94.cpp").read_bytes().decode("latin1")
    hdr=(SOURCE/"GY94.h").read_bytes().decode("latin1")
    nl="\r\n" if "\r\n" in cpp else "\n"
    fields="""
    // Exact per-run objective memoization. No optimizer/precision changes.
    double ExactCacheX[64][7], ExactCacheF[64];
    int ExactCacheValid, ExactCacheNext;
    void ClearExactCache() { ExactCacheValid=ExactCacheNext=0; }
""".replace("\n",nl)
    hdr=replace_once(hdr,"\tint Iround;",fields+"\tint Iround;")
    markers=["GY94::GY94() {","GY94::GY94(string NulModel) {",
             "int GY94::preProcess(const char* seq1, const char* seq2) {",
             "int GY94::PairwiseCodon(double space[]) {",
             "string GY94::Run(const char *seq1, const char *seq2) {"]
    for marker in markers:
        cpp=replace_once(cpp,marker,marker+nl+"    ClearExactCache();")
    original_call="\tEigenQc(0,-1,NULL,NULL,NULL, Root, U, V, pkappa, com.omega, PMat);"
    lookup="""
    // Raw com.kappa/KAPPA/omega assignments above are executed on every call.
    // Only immutable pair/model x[0:np] bit patterns can hit this cache.
    for(int entry=0; entry<ExactCacheValid; entry++) {
        if(memcmp(ExactCacheX[entry],x,np*sizeof(double))!=0) continue;
        if(model=="JC" || model=="K2P" || model=="TNEF" || model=="K3P" ||
           model=="TIMEF" || model=="TVMEF" || model=="SYM")
            initArray(com.pi,64,1.0/(64-getNumNonsense(genetic_code)));
        parseSubRates(model,pkappa);
        return ExactCacheF[entry];
    }
""".replace("\n",nl)
    cpp=replace_once(cpp,original_call,lookup+original_call)
    original_return="\treturn lnL1;"
    save="""
    int entry=ExactCacheNext;
    copyArray(x,ExactCacheX[entry],np);
    ExactCacheF[entry]=lnL1;
    if(ExactCacheValid<64) ExactCacheValid++;
    ExactCacheNext=(entry+1)%64;
""".replace("\n",nl)
    cpp=replace_once(cpp,original_return,save+original_return)
    planned={p.name:p.read_bytes() for p in SOURCE.iterdir() if p.is_file()}
    planned["GY94.cpp"]=cpp.encode("latin1")
    planned["GY94.h"]=hdr.encode("latin1")
    DEST.mkdir(parents=True,exist_ok=True)
    for name,data in planned.items():
        target=DEST/name
        if target.exists() and target.read_bytes()!=data:
            raise RuntimeError(f"Existing cached source differs: {target}; refusing overwrite")
        if not target.exists(): target.write_bytes(data)
    after={p.name:sha(p) for p in SOURCE.iterdir() if p.is_file()}
    if originals!=after: raise RuntimeError("Immutable original source changed")
    changed=[name for name in originals if sha(DEST/name)!=originals[name]]
    if changed!=["GY94.cpp","GY94.h"] and set(changed)!={"GY94.cpp","GY94.h"}:
        raise RuntimeError(f"Unexpected modified source files: {changed}")
    manifest={"original_source":str(SOURCE),"cached_source":str(DEST),"original_source_sha256":originals,
              "cached_source_sha256":{name:sha(DEST/name) for name in originals},"modified_files":changed,
              "original_source_verified_unchanged":True,"cache_entries":64,"cache_eviction":"FIFO",
              "cache_key":"memcmp x[0:np] double bytes","cache_scope":"Cleared at constructors/preProcess/PairwiseCodon/Run",
              "hit_side_effects":"Original raw com.kappa/KAPPA/omega writes, equal-frequency pi reset and parseSubRates once",
              "scientific_changes":[],"output_changes":[],"purpose":"Separates exact-cache algorithm benefit from CUDA hardware acceleration."}
    plan=ROOT/"build/cpu-reference-cached/source_manifest.json"
    plan.parent.mkdir(parents=True,exist_ok=True)
    plan.write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(str(plan))


if __name__=="__main__": main()
