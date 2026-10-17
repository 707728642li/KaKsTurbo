/* FP64 code-1 GY94 matrix construction extracted from immutable
 * KaKs_Calculator 2.0.1 GY94::EigenQc/eigenQREV/parseSubRates.
 * Original archive SHA256: d3f321647c262aaab5560fdc7fe886af11d3570fc27340d81ce5a757dcc2f0af.
 * Original source remains under tools/; arithmetic order is retained here.
 * This is not an optimizer or a fitted-result reproduction claim. */
#include "shared_core.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>

namespace kakscuda {
namespace {
constexpr char code1[]="FFLLSSSSYY!!CC!WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG";
constexpr int ncode=61, edge_count=263;
struct Edge { int i,j,rate; bool synonymous; };
struct EdgeTable { std::array<Edge,edge_count> edges{}; int count=0; };
constexpr std::array<int,ncode> make_sense() {
    std::array<int,ncode> sense{};int count=0;
    for(int i=0;i<64;i++)if(code1[i]!='!')sense[count++]=i;
    return sense;
}
constexpr auto sense=make_sense();
constexpr EdgeTable make_edges() {
    EdgeTable table{};
    for(int i=0;i<ncode;i++)for(int j=0;j<i;j++) {
        const int a=sense[i],b=sense[j];
        const int from[3]={a/16,(a/4)%4,a%4},to[3]={b/16,(b/4)%4,b%4};
        int ndiff=0,pos=0;
        for(int k=0;k<3;k++)if(from[k]!=to[k]){ndiff++;pos=k;}
        if(ndiff!=1)continue;
        const int lo=from[pos]<to[pos]?from[pos]:to[pos];
        const int hi=from[pos]>to[pos]?from[pos]:to[pos];
        int rate=5;
        if(lo==0&&hi==1)rate=0;else if(lo==2&&hi==3)rate=1;
        else if(lo==0&&hi==2)rate=2;else if(lo==1&&hi==3)rate=3;
        else if(lo==0&&hi==3)rate=4;
        if(table.count<edge_count)table.edges[table.count]={i,j,rate,code1[a]==code1[b]};
        table.count++;
    }
    return table;
}
constexpr EdgeTable table=make_edges();
static_assert(table.count==edge_count,"Code1 edge count must match the audited 263 entries");
constexpr int nkappa[14]={0,0,1,1,2,2,2,2,3,3,4,4,5,5};
bool integral(double x){return std::isfinite(x)&&x==std::floor(x);}
}

PairData::PairData(const char *id,const int32_t *meta,const double *values,
    const int8_t *z0,const int8_t *z1,const double *weight,uint32_t count,const double *gc) {
    if(!id||!*id||!meta||!values||!z0||!z1||!weight||!gc)
        throw std::invalid_argument("Missing prepared pair input");
    if(std::strchr(id,'\t')||std::strchr(id,'\n')||std::strchr(id,'\r'))
        throw std::invalid_argument("Pair identifier must be TSV compatible");
    if(!count||count>1891||meta[0]<0||meta[1]!=static_cast<int32_t>(count)||meta[2]<1||meta[3]!=0)
        throw std::invalid_argument("Invalid prepared pattern metadata");
    if(!integral(values[0])||values[0]!=3.*meta[2]||!integral(values[1])||values[1]<0.||values[1]>values[0])
        throw std::invalid_argument("Invalid cleaned length or observed substitutions");
    for(int i=0;i<61;i++)if(!std::isfinite(values[2+i])||values[2+i]<0.||values[2+i]>1.)
        throw std::invalid_argument("Invalid original codon frequency");
    double pi_sum=0.;for(int i=0;i<61;i++)pi_sum+=values[2+i];
    if(std::abs(pi_sum-1.)>1e-12)
        throw std::invalid_argument("Codon frequencies must already sum to one within input roundoff; no normalization is performed");
    for(int i=63;i<66;i++)if(values[i]!=0.)throw std::invalid_argument("Nonzero prepared ABI tail");
    for(int i=0;i<4;i++)if(!std::isfinite(gc[i])||gc[i]<0.||gc[i]>1.)
        throw std::invalid_argument("Invalid original GC");
    double total=0.,observed=0.;int previous=-1;
    patterns_.reserve(count);
    for(uint32_t h=0;h<count;h++) {
        const int i=z0[h],j=z1[h],index=i*61+j;
        if(i<0||i>=61||j<0||j>i||index<=previous||!integral(weight[h])||weight[h]<=0.)
            throw std::invalid_argument("Patterns must be positive counts in original triangular order");
        if(!(values[2+i]>0.)||!(values[2+j]>0.))
            throw std::invalid_argument("Observed pattern endpoints must have positive codon frequencies");
        const int a=sense[i],b=sense[j];
        const int diff=(a/16!=b/16)+((a/4)%4!=(b/4)%4)+(a%4!=b%4);
        observed+=diff*weight[h];
        total+=weight[h];previous=index;patterns_.push_back({z0[h],z1[h],weight[h]});
    }
    if(total!=meta[2])throw std::invalid_argument("Pattern count sum differs from cleaned codons");
    if(observed!=values[1])throw std::invalid_argument("Observed substitutions differ from code1 pattern Hamming counts");
    id_=id;std::copy_n(meta,4,meta_.begin());std::copy_n(values,66,values_.begin());std::copy_n(gc,4,gc_.begin());
}
ModelConfig model_config(uint32_t model,SupportPolicy support) {
    if(model>=14)throw std::invalid_argument("Only the original fourteen code1 models are supported");
    if(support!=SupportPolicy::LegacyCutoff&&support!=SupportPolicy::StrictPositive)
        throw std::invalid_argument("Unknown explicit frequency support policy");
    return {model,nkappa[model],model%2==0,support,support==SupportPolicy::LegacyCutoff?1e-6:0.};
}
std::array<double,6> expanded_rates(uint32_t model,const double *x) {
    if(model>=14||!x)throw std::invalid_argument("Invalid raw rate mapping input");
    std::array<double,6> output{};output.fill(1.);double *k=output.data();
    for(int i=0;i<nkappa[model];i++)k[i]=x[i+1];
    k[5]=1.;
    switch(model/2) {
    case 0:k[0]=k[1]=k[2]=k[3]=k[4]=k[5];break;
    case 1:k[1]=k[0];k[2]=k[3]=k[4]=k[5];break;
    case 2:k[2]=k[3]=k[4]=k[5];break;
    case 3:k[4]=k[5];k[2]=k[3]=k[1];k[1]=k[0];break;
    case 4:k[4]=k[5];k[3]=k[2];break;
    case 5:k[4]=k[3];k[3]=k[2];k[2]=k[1];k[1]=k[0];break;
    case 6:break;
    }
    return output;
}
ModelState model_state(const PairData &pair,uint32_t model,SupportPolicy support,const double *x) {
    if(!x)throw std::invalid_argument("Missing raw model parameters");
    ModelState state{};state.config=model_config(model,support);
    for(int i=0;i<7;i++) {
        if(!std::isfinite(x[i]))throw std::invalid_argument("Raw parameters must all be finite");
        state.x[i]=x[i];
    }
    if(x[0]<0.)throw std::invalid_argument("Negative codon branch length");
    state.kappa=x[1];
    for(int i=0;i<state.config.nkappa;i++) {
        if(x[i+1]<=0.)throw std::invalid_argument("Nonpositive active raw substitution rate");
    }
    state.omega=x[state.config.nkappa+1];
    if(state.omega<=0.)throw std::invalid_argument("Nonpositive omega");
    if(state.config.equal_frequency) {
        state.pi.fill(state.config.support_policy==SupportPolicy::LegacyCutoff?1.0/(64-3):0.);
        for(int i=0;i<61;i++)state.pi[i]=1.0/(64-3);
    }
    else {for(int i=0;i<61;i++)state.pi[i]=pair.values()[i+2];for(int i=61;i<64;i++)state.pi[i]=0.;}
    state.rates=expanded_rates(model,x);
    return state;
}
void build_q_snapshot(const ModelState &state,double *out) {
    if(!out)throw std::invalid_argument("Missing matrix output");
    std::fill_n(out,KAKS_CORE_SNAPSHOT_WIDTH,0.);
    double *q=out+KAKS_CORE_Q_OFFSET,*s=out+KAKS_CORE_S_OFFSET,*a=out+KAKS_CORE_A_OFFSET;
    const double *pi=state.pi.data(),*rates=state.rates.data();
    double rs=0.,ra0=0.,ra=0.;
    for(const Edge &edge:table.edges) {
        const int i=edge.i,j=edge.j;
        q[i*ncode+j]=rates[edge.rate];q[j*ncode+i]=q[i*ncode+j];
        q[i*ncode+j]*=pi[j];q[j*ncode+i]*=pi[i];
        const double pij=2*pi[i]*q[i*ncode+j];
        if(edge.synonymous)rs+=pij;
        else {ra0+=pij;q[i*ncode+j]*=state.omega;q[j*ncode+i]*=state.omega;ra+=pij*state.omega;}
    }
    const double mr=rs+ra;
    if(!(mr>0.)||!std::isfinite(mr))throw std::invalid_argument("Nonpositive or nonfinite original full-Q scale");
    for(int i=0;i<ncode;i++) {
        double sum=0.;for(int j=0;j<ncode;j++)sum+=q[i*ncode+j];q[i*ncode+i]=-sum;
    }
    int active=0;double *sqrtpi=out+KAKS_CORE_SQRT_PI_OFFSET;
    const double cutoff=state.config.support_cutoff;
    for(int i=0;i<ncode;i++){sqrtpi[i]=pi[i]>cutoff?std::sqrt(pi[i]):1.;if(pi[i]>cutoff)active++;}
    if(!active)throw std::invalid_argument("No active original frequencies");
    for(int i=0;i<ncode;i++)if(pi[i]>cutoff) {
        s[i*ncode+i]=q[i*ncode+i];
        for(int j=0;j<i;j++)if(pi[j]>cutoff)
            s[i*ncode+j]=s[j*ncode+i]=q[i*ncode+j]*sqrtpi[i]/sqrtpi[j];
    }
    for(int i=0;i<ncode*ncode;i++)a[i]=(s[i]*state.x[0])/mr;
    std::copy(state.pi.begin(),state.pi.end(),out+KAKS_CORE_PI_OFFSET);
    std::copy_n(rates,6,out+KAKS_CORE_RATES_OFFSET);
    double *v=out+KAKS_CORE_SCALARS_OFFSET;
    v[0]=rs;v[1]=ra0;v[2]=ra;v[3]=mr;v[4]=state.omega;v[5]=state.kappa;
    v[6]=state.x[0];v[7]=active;v[8]=state.config.nkappa+2;
    v[9]=static_cast<uint32_t>(state.config.support_policy);v[10]=cutoff;
}
}

namespace {
int report_error(const char *message,char *error,uint32_t cap) {
    if(error&&cap){size_t n=std::strlen(message);if(n>=cap)n=cap-1;std::memcpy(error,message,n);error[n]=0;}
    return 1;
}
void clear_error(char *error,uint32_t cap){if(error&&cap)error[0]=0;}
}
extern "C" uint32_t kaks_core_abi_version(void){return KAKS_CORE_ABI_VERSION;}
extern "C" uint32_t kaks_core_edge_count(void){return 263;}
extern "C" int kaks_core_create_prepared_pair(const char *id,const int32_t *meta,const double *values,
    const int8_t *z0,const int8_t *z1,const double *weights,uint32_t count,const double *gc,
    void **pair,char *error,uint32_t cap) {
    if(pair)*pair=nullptr;
    clear_error(error,cap);
    try {if(!pair)throw std::invalid_argument("Missing prepared pair handle output");
        auto owned=std::make_unique<kakscuda::PairData>(id,meta,values,z0,z1,weights,count,gc);
        *pair=owned.release();return 0;
    }catch(const std::exception &e){return report_error(e.what(),error,cap);}
    catch(...){return report_error("Unknown prepared pair creation error",error,cap);}
}
extern "C" int kaks_core_export_prepared_pair(const void *raw,int32_t *meta,double *values,
    int8_t *z0,int8_t *z1,double *weights,uint32_t capacity,double *gc,char *error,uint32_t cap) {
    clear_error(error,cap);
    try {if(!raw||!meta||!values||!z0||!z1||!weights||!gc)throw std::invalid_argument("Missing prepared pair export buffer");
        const auto &pair=*static_cast<const kakscuda::PairData *>(raw);
        if(capacity<pair.patterns().size())throw std::invalid_argument("Prepared export capacity is too small");
        std::copy(pair.meta().begin(),pair.meta().end(),meta);std::copy(pair.values().begin(),pair.values().end(),values);
        std::copy(pair.gc().begin(),pair.gc().end(),gc);
        for(size_t i=0;i<pair.patterns().size();i++){z0[i]=pair.patterns()[i].z0;z1[i]=pair.patterns()[i].z1;weights[i]=pair.patterns()[i].weight;}
        return 0;
    }catch(const std::exception &e){return report_error(e.what(),error,cap);}
    catch(...){return report_error("Unknown prepared pair export error",error,cap);}
}
extern "C" void kaks_core_free_prepared_pair(void *raw){delete static_cast<kakscuda::PairData *>(raw);}
extern "C" int kaks_core_q_snapshot(const void *raw,uint32_t model,uint32_t support,const double *x,
    double *out,uint32_t capacity,char *error,uint32_t cap) {
    clear_error(error,cap);
    try {if(!raw||!out||capacity<KAKS_CORE_SNAPSHOT_WIDTH)throw std::invalid_argument("Missing or undersized Q output buffer");
        const auto &pair=*static_cast<const kakscuda::PairData *>(raw);
        auto state=kakscuda::model_state(pair,model,static_cast<kakscuda::SupportPolicy>(support),x);
        kakscuda::build_q_snapshot(state,out);return 0;
    }catch(const std::exception &e){return report_error(e.what(),error,cap);}
    catch(...){return report_error("Unknown matrix construction error",error,cap);}
}
