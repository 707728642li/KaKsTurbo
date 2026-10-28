#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <float.h>
#include <ctype.h>
#include <stdlib.h>
#include <cstring>
#include <string>
#include <vector>
#include <set>
#include <stdexcept>
#include <limits.h>

#include "kaks_adapter.h"
#include "adapter_bridge.h"
#include "MSMA.h"

namespace {
const char *MODELS[14] = {"JC", "F81", "K2P", "HKY", "TNEF", "TN", "K3P",
    "K3PUF", "TIMEF", "TIM", "TVMEF", "TVM", "SYM", "GTR"};
const char *SCHEMA = "Sequence\tMethod\tKa\tKs\tKa/Ks\tP-Value(Fisher)\tLength\t"
    "S-Sites\tN-Sites\tFold-Sites(0:2:4)\tSubstitutions\tS-Substitutions\t"
    "N-Substitutions\tFold-S-Substitutions(0:2:4)\tFold-N-Substitutions(0:2:4)\t"
    "Divergence-Time\tSubstitution-Rate-Ratio(rTC:rAG:rTA:rCG:rTG:rCA/rCA)\t"
    "GC(1:2:3)\tML-Score\tAICc\tAkaike-Weight\tModel\n";

CRITICAL_SECTION native_lock;
struct LockLifetime {
    LockLifetime() { InitializeCriticalSection(&native_lock); }
    ~LockLifetime() { DeleteCriticalSection(&native_lock); }
} lock_lifetime;
struct Locked {
    Locked() { EnterCriticalSection(&native_lock); }
    ~Locked() { LeaveCriticalSection(&native_lock); }
private:
    Locked(const Locked &);
    Locked &operator=(const Locked &);
};

struct PairContext {
    std::string id, a, b;
    uint32_t raw_nt;
    double gc[4], snp, pi[64];
    int32_t diagnostics[4];
    std::vector<int8_t> z0, z1;
    std::vector<double> weight;
};
std::set<PairContext *> live_contexts;
PairContext *injected_context = NULL;
const double *injected_results = NULL;
int injected_count = 0;
int native_eigen_warnings = 0;

int fail(int code, const std::string &message, char *buffer, uint32_t capacity) {
    if (buffer && capacity) {
        size_t size = message.size();
        if (size >= capacity) size = capacity - 1;
        std::memcpy(buffer, message.data(), size);
        buffer[size] = 0;
    }
    return code;
}
void clear_error(char *buffer, uint32_t capacity) {
    if (buffer && capacity) buffer[0] = 0;
}
PairContext *get_context(void *raw) {
    PairContext *ctx = static_cast<PairContext *>(raw);
    if (!ctx || live_contexts.find(ctx) == live_contexts.end())
        throw std::invalid_argument("Unknown or already released native pair context");
    return ctx;
}
bool is_finite_native(double value) { return _finite(value) != 0; }
void activate(const PairContext &ctx) {
    genetic_code = 1;
    seq_name = ctx.id;
    length = static_cast<unsigned long>(ctx.a.size());
    for (int i = 0; i < 4; ++i) GC[i] = ctx.gc[i];
}

/* Scientific masking follows KAKS::checkValid. Stop lookup is short-circuited
 * for invalid codons so an N cannot produce a negative translation-table index.
 * This only removes upstream undefined memory access, not an additional codon.
 * GC deliberately follows KAKS::getGCContent BEFORE uppercase/masking. */
void clean_pair(PairContext &ctx, const std::string &a, const std::string &b) {
    Base base;
    const std::string joined = a + b;
    for (int j = 0; j < 4; ++j) ctx.gc[j] = 0.0;
    for (size_t i = 0; i < joined.size(); i += 3)
        for (int j = 0; j < 3; ++j)
            if (joined[i+j] == 'G' || joined[i+j] == 'C') ctx.gc[j+1]++;
    double gc_total = 0.0;
    for (int j = 1; j < 4; ++j) gc_total += ctx.gc[j];
    ctx.gc[0] = gc_total / joined.size();
    for (int j = 1; j < 4; ++j) ctx.gc[j] /= (joined.size()/3);
    ctx.a.reserve(a.size()); ctx.b.reserve(b.size());
    for (size_t i = 0; i < a.size(); i += 3) {
        std::string ca = a.substr(i,3), cb = b.substr(i,3);
        bool drop = false;
        for (int j = 0; j < 3; ++j) {
            ca[j] = static_cast<char>(toupper(static_cast<unsigned char>(ca[j])));
            cb[j] = static_cast<char>(toupper(static_cast<unsigned char>(cb[j])));
            if (ca[j]=='-' || cb[j]=='-' || base.convertChar(ca[j])==-1 || base.convertChar(cb[j])==-1)
                drop = true;
        }
        if (!drop && (base.getAminoAcid(ca)=='!' || base.getAminoAcid(cb)=='!')) drop = true;
        if (!drop) { ctx.a += ca; ctx.b += cb; }
    }
}

/* Original encoding, directed PatternWeight and GetCodonFreqs are reused.
 * The triangular fold and frequency accumulation are the exact loop order of
 * GY94::PairwiseCodon before ming2. No likelihood fit is executed here. */
class ModelPreparation : public GY94 {
public:
    ModelPreparation() : GY94("F81") { com.z[0]=com.z[1]=NULL; GAMMA=0.0f; }
    ~ModelPreparation() { free(com.z[0]); free(com.z[1]); }
    void prepare(PairContext &ctx) {
        str1=ctx.a; str2=ctx.b;
        preProcess(ctx.a.c_str(),ctx.b.c_str());
        if (com.ncode!=61 || com.ls!=static_cast<int>(ctx.a.size()/3))
            throw std::runtime_error("Upstream code-1 encoding dimensions differ");
        double fp[CODON*CODON];
        initArray(fp,CODON*CODON);
        for (int h=0;h<com.npatt;++h) {
            int j=max2(com.z[0][h],com.z[1][h]);
            int k=min2(com.z[0][h],com.z[1][h]);
            if (j<0 || j>=61 || k<0 || k>=61)
                throw std::runtime_error("Invalid encoded sense codon");
            fp[j*com.ncode+k]+=com.fpatt[h];
        }
        for (int j=0;j<com.ncode;++j)
            for (int k=0;k<j+1;++k)
                if (fp[j*com.ncode+k]) {
                    ctx.z0.push_back(static_cast<int8_t>(j));
                    ctx.z1.push_back(static_cast<int8_t>(k));
                    ctx.weight.push_back(fp[j*com.ncode+k]);
                }
        initArray(com.pi,CODON);
        for (size_t h=0;h<ctx.weight.size();++h) {
            com.pi[ctx.z0[h]] += ctx.weight[h]/(2.*com.ls);
            com.pi[ctx.z1[h]] += ctx.weight[h]/(2.*com.ls);
        }
        GetCodonFreqs(com.pi);
        ctx.snp=snp;
        for (int j=0;j<61;++j) ctx.pi[j]=com.pi[j];
        for (int j=61;j<64;++j) ctx.pi[j]=0.0;
    }
};

class CandidateFormatter : public Base {
public:
    std::string format(const PairContext &ctx, const char *candidate, const double *r) {
        name=std::string("GY-")+candidate;
        model=candidate;
        lnL=r[0]; AICc=r[1];
        Ka=r[4]; Ks=r[5]; S=r[6]; N=r[7]; Sd=r[8]; Nd=r[9]; t=r[10];
        snp=ctx.snp;
        for (int j=0;j<6;++j) KAPPA[j]=r[11+j];
        return parseOutput();
    }
};

class NativeMA : public MA {
public:
    NativeMA() : MA() { com.z[0]=com.z[1]=NULL; GAMMA=0.0f; }
    ~NativeMA() { free(com.z[0]); free(com.z[1]); }
};

class NativeEvaluator : public GY94 {
public:
    NativeEvaluator(const char *candidate) : GY94(candidate) { com.z[0]=com.z[1]=NULL; GAMMA=0.0f; }
    ~NativeEvaluator() { free(com.z[0]); free(com.z[1]); }
    void evaluate(const PairContext &ctx,const double *input,double *r,int32_t *status,int model_id) {
        str1=ctx.a; str2=ctx.b; setmark_61_64();
        com.ls=static_cast<int>(ctx.a.size()/3);
        com.npatt=static_cast<int>(ctx.weight.size());
        com.z[0]=static_cast<char *>(malloc(ctx.weight.size()));
        com.z[1]=static_cast<char *>(malloc(ctx.weight.size()));
        if (!com.z[0] || !com.z[1]) throw std::bad_alloc();
        for (int j=0;j<64;++j) com.pi[j]=ctx.pi[j];
        for (int j=0;j<com.npatt;++j) {
            com.z[0][j]=ctx.z0[j]; com.z[1][j]=ctx.z1[j]; com.fpatt[j]=ctx.weight[j];
        }
        snp=ctx.snp;
        double x[7]; for (int j=0;j<7;++j) x[j]=input[j];
        lnL=-lfun2dSdN(x,com.np);
        EigenQc(1,x[0],&S,&Ks,&Ka,NULL,NULL,NULL,com.KAPPA,com.omega,PMat);
        N=com.ls*3-S; Sd=Ks*S; Nd=Ka*N;
        double scale=(Sd+Nd)/snp; Sd/=scale; Nd/=scale;
        AICc=-2*lnL+2.*(com.nkappa+2)*(length/3)/((length/3)-(com.nkappa+2)-1.);
        t=x[0]/3;
        parseSubRates(model,com.KAPPA);
        for (int j=0;j<6;++j) KAPPA[j]=com.KAPPA[j];
        for (int j=0;j<96;++j) r[j]=0.0;
        // Preserve the upstream MS::selectModel multiply after GY division;
        // 3*(x0/3) can differ from x0 by one bit and affects MA averaging.
        r[0]=lnL; r[1]=AICc; r[2]=3.*t; r[3]=com.omega;
        r[4]=Ka; r[5]=Ks; r[6]=S; r[7]=N; r[8]=Sd; r[9]=Nd; r[10]=t;
        for (int j=0;j<6;++j) r[11+j]=KAPPA[j];
        for (int j=0;j<64;++j) r[17+j]=com.pi[j];
        for (int j=0;j<7;++j) r[81+j]=x[j];
        status[0]=model_id; status[1]=-2; status[2]=0; status[3]=native_eigen_warnings;
    }
};

struct InjectionScope {
    InjectionScope(PairContext *ctx,const double *results) {
        if (injected_context) throw std::runtime_error("Nested native model injection");
        injected_context=ctx; injected_results=results; injected_count=0;
    }
    ~InjectionScope() { injected_context=NULL; injected_results=NULL; injected_count=0; }
};
}

/* Called by the macro-hooked MS::selectModel, replacing only its GY94 fit. */
void kaks_native_get_injected_model(const char *model_name,KaksNativeInjectedModel &model) {
    if (!injected_context || !injected_results || injected_count>=14)
        throw std::runtime_error("Missing active GPU result injection; CPU fit fallback is disabled");
    const int index=injected_count;
    if (std::strcmp(model_name,MODELS[index])!=0)
        throw std::runtime_error("Upstream candidate model order differs from the GPU ABI");
    const double *r=injected_results+index*KAKS_ADAPTER_RESULT_WIDTH;
    CandidateFormatter formatter;
    model.row=formatter.format(*injected_context,model_name,r);
    model.aicc=r[1]; model.time_x0=r[2]; model.omega=r[3];
    for (int j=0;j<6;++j) model.rates[j]=r[11+j];
    for (int j=0;j<64;++j) model.freq[j]=r[17+j];
    ++injected_count;
}

void kaks_native_record_eigen_status(int status) {
    if (status) ++native_eigen_warnings;
}

extern "C" {
uint32_t kaks_adapter_abi_version(void) { return KAKS_ADAPTER_ABI_VERSION; }
const char *kaks_adapter_schema(void) { return SCHEMA; }
const char *kaks_adapter_model_name(uint32_t model) { return model<14 ? MODELS[model] : NULL; }

int kaks_prepare_pair(const char *id,const char *a,const char *b,void **out,
                     char *error,uint32_t error_capacity) {
    Locked lock;
    clear_error(error,error_capacity);
    if (out) *out=NULL;
    if (!id || !a || !b || !out)
        return fail(KAKS_ADAPTER_INVALID_ARGUMENT,"Null prepare argument",error,error_capacity);
    PairContext *ctx=NULL;
    try {
        std::string name(id),left(a),right(b);
        if (name.empty() || name.find_first_of("\t\r\n")!=std::string::npos)
            return fail(KAKS_ADAPTER_INVALID_INPUT,"Pair ID must be nonempty and contain no TSV/newline separator",error,error_capacity);
        if (left.empty() || left.size()!=right.size() || left.size()%3 || left.size()>INT_MAX/2)
            return fail(KAKS_ADAPTER_INVALID_INPUT,"AXT sequences must be equal, nonempty codon lengths",error,error_capacity);
        if (left.find_first_of("\r\n\t ")!=std::string::npos || right.find_first_of("\r\n\t ")!=std::string::npos)
            return fail(KAKS_ADAPTER_INVALID_INPUT,"Pass concatenated AXT sequence bodies without whitespace",error,error_capacity);
        genetic_code=1;
        ctx=new PairContext;
        ctx->id=name; ctx->raw_nt=static_cast<uint32_t>(left.size());
        for (int j=0;j<4;++j) ctx->diagnostics[j]=0;
        clean_pair(*ctx,left,right);
        if (ctx->a.empty()) {
            delete ctx;
            return fail(KAKS_ADAPTER_NO_USABLE_CODONS,"Paired masking removed every aligned codon",error,error_capacity);
        }
        activate(*ctx);
        { ModelPreparation prep; prep.prepare(*ctx); }
        for (int j=0;j<61;++j)
            if (!is_finite_native(ctx->pi[j]) || ctx->pi[j]<0.0)
                throw std::runtime_error("Nonfinite upstream prepared codon frequency");
        live_contexts.insert(ctx); *out=ctx;
        return KAKS_ADAPTER_OK;
    } catch (const std::exception &e) {
        delete ctx; return fail(KAKS_ADAPTER_INTERNAL_ERROR,e.what(),error,error_capacity);
    } catch (...) {
        delete ctx; return fail(KAKS_ADAPTER_INTERNAL_ERROR,"Unknown native prepare error",error,error_capacity);
    }
}

int kaks_get_pair_sizes(void *raw,uint32_t *raw_nt,uint32_t *clean_nt,
                        uint32_t *npatt,char *error,uint32_t error_capacity) {
    Locked lock; clear_error(error,error_capacity);
    if (!raw_nt || !clean_nt || !npatt)
        return fail(KAKS_ADAPTER_INVALID_ARGUMENT,"Null size output",error,error_capacity);
    try {
        PairContext *ctx=get_context(raw);
        *raw_nt=ctx->raw_nt; *clean_nt=static_cast<uint32_t>(ctx->a.size());
        *npatt=static_cast<uint32_t>(ctx->weight.size());
        return KAKS_ADAPTER_OK;
    } catch (const std::exception &e) { return fail(KAKS_ADAPTER_INVALID_ARGUMENT,e.what(),error,error_capacity); }
    catch (...) { return fail(KAKS_ADAPTER_INTERNAL_ERROR,"Unknown native size error",error,error_capacity); }
}

int kaks_export_pair(void *raw,int32_t offset,int32_t *meta,double *values,
                     int8_t *z0,int8_t *z1,double *weights,uint32_t capacity,
                     double *gc,char *error,uint32_t error_capacity) {
    Locked lock; clear_error(error,error_capacity);
    if (!meta || !values || !z0 || !z1 || !weights || !gc || offset<0)
        return fail(KAKS_ADAPTER_INVALID_ARGUMENT,"Null export buffer or negative pattern offset",error,error_capacity);
    try {
        PairContext *ctx=get_context(raw);
        const uint32_t size=static_cast<uint32_t>(ctx->weight.size());
        if (capacity<size || static_cast<uint64_t>(offset)+size>INT_MAX)
            return fail(KAKS_ADAPTER_BUFFER_TOO_SMALL,"Pattern capacity/offset is insufficient",error,error_capacity);
        meta[0]=offset; meta[1]=size; meta[2]=ctx->a.size()/3; meta[3]=0;
        values[0]=static_cast<double>(ctx->a.size()); values[1]=ctx->snp;
        for (int j=0;j<64;++j) values[2+j]=ctx->pi[j];
        for (uint32_t j=0;j<size;++j) {
            z0[j]=ctx->z0[j]; z1[j]=ctx->z1[j]; weights[j]=ctx->weight[j];
        }
        for (int j=0;j<4;++j) gc[j]=ctx->gc[j];
        return KAKS_ADAPTER_OK;
    } catch (const std::exception &e) { return fail(KAKS_ADAPTER_INVALID_ARGUMENT,e.what(),error,error_capacity); }
    catch (...) { return fail(KAKS_ADAPTER_INTERNAL_ERROR,"Unknown native export error",error,error_capacity); }
}

int kaks_finalize_pair(void *raw,const double *results,uint32_t model_count,
                       const int32_t *statuses,char *output,uint32_t capacity,
                       uint32_t *required,double *cpu_seconds,char *error,uint32_t error_capacity) {
    Locked lock; clear_error(error,error_capacity);
    if (required) *required=0;
    if (cpu_seconds) *cpu_seconds=0.0;
    if (output && capacity) output[0]=0;
    if (!results || !statuses || !required || model_count!=14)
        return fail(KAKS_ADAPTER_MISSING_MODEL,"Exactly 14 GPU model results/statuses are required",error,error_capacity);
    try {
        PairContext *ctx=get_context(raw);
        const int32_t first=statuses[0];
        if (first<0 || first%14!=0)
            return fail(KAKS_ADAPTER_MISSING_MODEL,"Missing or misordered first GPU model status",error,error_capacity);
        for (int model=0;model<14;++model) {
            const int32_t *s=statuses+model*4;
            if (s[0]!=first+model || s[2]<0 || s[3]<0)
                return fail(KAKS_ADAPTER_MISSING_MODEL,std::string("Missing/misordered GPU model: ")+MODELS[model],error,error_capacity);
            if (s[1]==-1)
                return fail(KAKS_ADAPTER_MISSING_MODEL,std::string("Unfinished GPU fit cannot be used as an MA model: ")+MODELS[model],error,error_capacity);
            if (s[1]==-2)
                return fail(KAKS_ADAPTER_MISSING_MODEL,std::string("Unfitted evaluation cannot be used as an MA model: ")+MODELS[model],error,error_capacity);
            if (s[1]!=0 && s[1]!=1)
                return fail(KAKS_ADAPTER_GPU_MODEL_FAILED,std::string("Unsupported GPU optimizer status in ")+MODELS[model],error,error_capacity);
            // This source version's ming2 returns only 0/1, neither of which
            // proves convergence. Preserve iteration-limit/eigen metadata.
            const double *r=results+model*96;
            // Sd/Nd may be NaN for a zero-SNP degenerate fit in upstream code;
            // preserve the original parseOutput NA rules instead of replacing it.
            const int scalar_indices[]={0,1,2,3,4,5,6,7,10};
            for (unsigned j=0;j<sizeof(scalar_indices)/sizeof(scalar_indices[0]);++j)
                if (!is_finite_native(r[scalar_indices[j]]))
                    return fail(KAKS_ADAPTER_NONFINITE_MODEL,std::string("Nonfinite GPU model scalar in ")+MODELS[model],error,error_capacity);
            if (r[2]<=0.0 || r[3]<=0.0 || r[4]<0.0 || r[5]<0.0 || r[6]<0.0 || r[7]<0.0)
                return fail(KAKS_ADAPTER_GPU_MODEL_FAILED,std::string("Invalid GPU model parameter domain in ")+MODELS[model],error,error_capacity);
            for (int j=0;j<6;++j)
                if (!is_finite_native(r[11+j]) || r[11+j]<=0.0)
                    return fail(KAKS_ADAPTER_NONFINITE_MODEL,std::string("Invalid GPU substitution rate in ")+MODELS[model],error,error_capacity);
            for (int j=0;j<61;++j)
                if (!is_finite_native(r[17+j]) || r[17+j]<0.0)
                    return fail(KAKS_ADAPTER_NONFINITE_MODEL,std::string("Invalid GPU codon frequency in ")+MODELS[model],error,error_capacity);
        }
        LARGE_INTEGER before,after,frequency;
        QueryPerformanceFrequency(&frequency); QueryPerformanceCounter(&before);
        activate(*ctx);
        native_eigen_warnings=0;
        for (int j=0;j<4;++j) ctx->diagnostics[j]=0;
        for (int model=0;model<14;++model) {
            const int32_t *s=statuses+4*model;
            // ABI slot retained for compatibility; validated returns are 0/1.
            ctx->diagnostics[0]+=(s[1]<0);
            ctx->diagnostics[1]+=(s[2]>=2000);
            ctx->diagnostics[2]+=s[3];
        }
        std::string row;
        {
            InjectionScope injection(ctx,results);
            // MLResult is protected: the wrapper below owns the call chain.
            class RunInjected : public MS {
            public:
                std::string run(const PairContext &context) {
                    GAMMA=0.0f;
                    std::vector<MLResult> models;
                    std::string details;
                    MS::Run(context.a.c_str(),context.b.c_str(),models,details);
                    if (injected_count!=14)
                        throw std::runtime_error("Upstream MS did not consume exactly 14 GPU models");
                    NativeMA ma;
                    return ma.Run(context.a.c_str(),context.b.c_str(),models);
                }
            } injected;
            row=injected.run(*ctx);
        }
        QueryPerformanceCounter(&after);
        ctx->diagnostics[3]=native_eigen_warnings;
        if (cpu_seconds) *cpu_seconds=double(after.QuadPart-before.QuadPart)/frequency.QuadPart;
        unsigned columns=1;
        for (size_t j=0;j<row.size();++j) if (row[j]=='\t') ++columns;
        if (columns!=22 || row.empty() || row[row.size()-1]!='\n')
            throw std::runtime_error("Original MA postprocessing produced malformed 22-column output");
        if (row.size()+1>UINT_MAX)
            throw std::runtime_error("Native output size overflow");
        *required=static_cast<uint32_t>(row.size()+1);
        if (!output || capacity<*required)
            return fail(KAKS_ADAPTER_BUFFER_TOO_SMALL,"MA output buffer too small; required_capacity is set",error,error_capacity);
        std::memcpy(output,row.c_str(),row.size()+1);
        return KAKS_ADAPTER_OK;
    } catch (const std::invalid_argument &e) { return fail(KAKS_ADAPTER_INVALID_ARGUMENT,e.what(),error,error_capacity); }
    catch (const std::exception &e) { return fail(KAKS_ADAPTER_INTERNAL_ERROR,e.what(),error,error_capacity); }
    catch (...) { return fail(KAKS_ADAPTER_INTERNAL_ERROR,"Unknown native finalize error",error,error_capacity); }
}

int kaks_evaluate_pair(void *raw,uint32_t model,const double *x,double *results,
                       int32_t *status,char *error,uint32_t error_capacity) {
    Locked lock; clear_error(error,error_capacity);
    if (!x || !results || !status || model>=14)
        return fail(KAKS_ADAPTER_INVALID_ARGUMENT,"Invalid single-evaluation arguments",error,error_capacity);
    try {
        PairContext *ctx=get_context(raw); activate(*ctx);
        const int nkappa[]={0,0,1,1,2,2,2,2,3,3,4,4,5,5};
        const int np=nkappa[model]+2;
        for (int j=0;j<np;++j)
            if (!is_finite_native(x[j]) || x[j]<=0)
                return fail(KAKS_ADAPTER_INVALID_ARGUMENT,"Evaluation x parameters must be finite and positive",error,error_capacity);
        native_eigen_warnings=0;
        NativeEvaluator evaluator(MODELS[model]);
        evaluator.evaluate(*ctx,x,results,status,model);
        return KAKS_ADAPTER_OK;
    } catch (const std::invalid_argument &e) { return fail(KAKS_ADAPTER_INVALID_ARGUMENT,e.what(),error,error_capacity); }
    catch (const std::exception &e) { return fail(KAKS_ADAPTER_INTERNAL_ERROR,e.what(),error,error_capacity); }
    catch (...) { return fail(KAKS_ADAPTER_INTERNAL_ERROR,"Unknown native evaluation error",error,error_capacity); }
}

int kaks_get_postprocess_diagnostics(void *raw,int32_t *diagnostics,char *error,uint32_t error_capacity) {
    Locked lock; clear_error(error,error_capacity);
    if (!diagnostics) return fail(KAKS_ADAPTER_INVALID_ARGUMENT,"Null diagnostics output",error,error_capacity);
    try {
        PairContext *ctx=get_context(raw);
        for (int j=0;j<4;++j) diagnostics[j]=ctx->diagnostics[j];
        return KAKS_ADAPTER_OK;
    } catch (const std::exception &e) { return fail(KAKS_ADAPTER_INVALID_ARGUMENT,e.what(),error,error_capacity); }
    catch (...) { return fail(KAKS_ADAPTER_INTERNAL_ERROR,"Unknown native diagnostics error",error,error_capacity); }
}

void kaks_free_pair(void *raw) {
    Locked lock;
    PairContext *ctx=static_cast<PairContext *>(raw);
    std::set<PairContext *>::iterator found=live_contexts.find(ctx);
    if (found!=live_contexts.end()) { live_contexts.erase(found); delete ctx; }
}
}
