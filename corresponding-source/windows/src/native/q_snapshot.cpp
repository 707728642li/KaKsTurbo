#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <stdint.h>
#include <cstring>
#include <cstdlib>
#include <string>
#include <stdexcept>
#include "GY94.h"
#include "q_snapshot_hook.h"

/* Flat FP64 ABI, no structure alignment:
 * Q=0[3721], S=3721[3721], U=7442[3721], V=11163[3721], pi=14884[61],
 * sqrt_pi=14945[61], Root=15006[61], parsed_rates=15067[6], mr=15073,
 * omega=15074, native_lnL=15075, original_expt=15076[61]. Total=15137.
 * S stores the compressed active block in a padded 61-column row stride.
 * meta[4]={ncode,active_count,pattern_count,cleaned_codon_count}.
 * Inputs must already be paired masked, uppercase code-1 sense codons.
 * No CPU optimizer is called. Q/S callbacks copy source data and return.
 */
namespace {
const char *models[14] = {"JC","F81","K2P","HKY","TNEF","TN","K3P",
    "K3PUF","TIMEF","TIM","TVMEF","TVM","SYM","GTR"};
enum { QSVALUES=15137, QOFF=0, SOFF=3721, UOFF=7442, VOFF=11163,
       PIOFF=14884, SQOFF=14945, ROFF=15006, RATEOFF=15067,
       MROFF=15073, WOFF=15074, LL=15075, EXPOFF=15076 };
CRITICAL_SECTION guard;
struct Lifetime {
    Lifetime() { InitializeCriticalSection(&guard); }
    ~Lifetime() { DeleteCriticalSection(&guard); }
} lifetime;
struct Locked {
    Locked() { EnterCriticalSection(&guard); }
    ~Locked() { LeaveCriticalSection(&guard); }
};
double *target=NULL;
int q_count=0, s_count=0, active_count=0;
struct CaptureScope {
    CaptureScope(double *out) { target=out; q_count=s_count=active_count=0; }
    ~CaptureScope() { target=NULL; }
};

class Snapshot : public GY94 {
public:
    Snapshot(const char *model) : GY94(model) { com.z[0]=com.z[1]=NULL; GAMMA=0; }
    ~Snapshot() { free(com.z[0]); free(com.z[1]); }
    void run(const char *a,const char *b,const double *input,double *out,
             int32_t *meta,int8_t *z0,int8_t *z1,double *weights) {
        str1=a; str2=b;
        preProcess(a,b);
        if (com.ncode!=61) throw std::runtime_error("Expected code-1 ncode=61");
        double fp[CODON*CODON]; initArray(fp,CODON*CODON);
        for (int h=0;h<com.npatt;++h) {
            int j=max2(com.z[0][h],com.z[1][h]);
            int k=min2(com.z[0][h],com.z[1][h]);
            fp[j*com.ncode+k]+=com.fpatt[h];
        }
        com.npatt=0;
        for (int j=0;j<com.ncode;++j) for (int k=0;k<=j;++k)
            if (fp[j*com.ncode+k]) {
                com.z[0][com.npatt]=static_cast<char>(j);
                com.z[1][com.npatt]=static_cast<char>(k);
                com.fpatt[com.npatt++]=fp[j*com.ncode+k];
            }
        initArray(com.pi,CODON);
        for (int h=0;h<com.npatt;++h) {
            com.pi[static_cast<int>(com.z[0][h])]+=com.fpatt[h]/(2.*com.ls);
            com.pi[static_cast<int>(com.z[1][h])]+=com.fpatt[h]/(2.*com.ls);
            z0[h]=com.z[0][h]; z1[h]=com.z[1][h]; weights[h]=com.fpatt[h];
        }
        GetCodonFreqs(com.pi);
        double x[7]; for (int i=0;i<7;++i) x[i]=input[i];
        CaptureScope capture(out);
        out[LL]=-lfun2dSdN(x,com.np);
        if (q_count!=1 || s_count!=1) throw std::runtime_error("Snapshot callback count mismatch");
        meta[0]=com.ncode; meta[1]=active_count; meta[2]=com.npatt; meta[3]=com.ls;
        for (int i=0;i<61;++i) {
            out[ROFF+i]=Root[i];
            out[EXPOFF+i]=exp(x[0]*Root[i]);
        }
        for (int i=0;i<61*61;++i) { out[UOFF+i]=U[i]; out[VOFF+i]=V[i]; }
    }
};

void validate_clean(const char *a,const char *b) {
    if (!a || !b) throw std::invalid_argument("Missing cleaned sequence");
    size_t n=std::strlen(a);
    if (!n || n!=std::strlen(b) || n%3) throw std::invalid_argument("Cleaned lengths must be equal, nonempty and divisible by3");
    Base base;
    for (size_t i=0;i<n;i+=3) {
        for (int k=0;k<3;++k)
            if (!std::strchr("TCAG",a[i+k]) || !std::strchr("TCAG",b[i+k]))
                throw std::invalid_argument("Input is not uppercase masked ACGT");
        if (base.getAminoAcid(std::string(a+i,3))=='!' || base.getAminoAcid(std::string(b+i,3))=='!')
            throw std::invalid_argument("Cleaned input still contains a stop codon");
    }
}
}

void kaks_q_snapshot_record_q(const double *q,const double *pi,double mr,
                             const double *rates,double omega,int n) {
    if (!target) return;
    if (n!=61) throw std::runtime_error("Unexpected captured Q dimension");
    ++q_count;
    for (int i=0;i<n*n;++i) target[QOFF+i]=q[i];
    for (int i=0;i<n;++i) target[PIOFF+i]=pi[i];
    for (int i=0;i<6;++i) target[RATEOFF+i]=rates[i];
    target[MROFF]=mr; target[WOFF]=omega;
}

void kaks_q_snapshot_record_s(const double *s,int n,const double *sqrt_pi) {
    if (!target) return;
    if (n<1 || n>61) throw std::runtime_error("Unexpected captured S dimension");
    ++s_count; active_count=n;
    for (int i=0;i<n;++i) {
        target[SQOFF+i]=sqrt_pi[i];
        for (int j=0;j<n;++j) target[SOFF+i*61+j]=s[i*n+j];
    }
}

extern "C" __declspec(dllexport) int kaks_q_snapshot_capture(
    const char *clean_a,const char *clean_b,uint32_t code,uint32_t model,
    const double *x,double *out,uint32_t out_count,int32_t *meta,
    int8_t *z0,int8_t *z1,double *weights,uint32_t pattern_capacity,
    char *error,uint32_t error_capacity) {
    Locked lock;
    if (error && error_capacity) error[0]=0;
    try {
        if (code!=1 || model>=14) throw std::invalid_argument("Only code1 and14 known models are supported");
        if (!x || !out || out_count<QSVALUES || !meta || !z0 || !z1 || !weights || pattern_capacity<61*61)
            throw std::invalid_argument("Missing or undersized snapshot output buffer");
        genetic_code=1;
        validate_clean(clean_a,clean_b);
        length=static_cast<unsigned long>(std::strlen(clean_a));
        seq_name="native-q-snapshot";
        for (int i=0;i<QSVALUES;++i) out[i]=0;
        Snapshot snapshot(models[model]);
        snapshot.run(clean_a,clean_b,x,out,meta,z0,z1,weights);
        return 0;
    } catch (const std::exception &e) {
        if (error && error_capacity) {
            size_t n=std::strlen(e.what()); if (n>=error_capacity) n=error_capacity-1;
            std::memcpy(error,e.what(),n); error[n]=0;
        }
        return 1;
    } catch (...) {
        if (error && error_capacity) {
            const char *message="Unknown native snapshot exception";
            size_t n=std::strlen(message); if (n>=error_capacity) n=error_capacity-1;
            std::memcpy(error,message,n); error[n]=0;
        }
        return 2;
    }
}
