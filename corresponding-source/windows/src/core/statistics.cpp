#include "statistics.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace kakscuda {
namespace {
constexpr char code[]="FFLLSSSSYY!!CC!WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG";
constexpr std::array<int,61> sense_codes() {
    std::array<int,61> out{};int n=0;
    for(int i=0;i<64;i++)if(code[i]!='!')out[n++]=i;
    return out;
}
constexpr auto sense=sense_codes();
struct Edge {int i,j,rate;bool synonymous;};
struct Table {std::array<Edge,263> values{};int count=0;};
constexpr Table make_edges() {
    Table table{};
    for(int i=0;i<61;i++)for(int j=0;j<i;j++) {
        int a=sense[i],b=sense[j],n=0,pos=0;
        const int av[3]={a/16,(a/4)%4,a%4},bv[3]={b/16,(b/4)%4,b%4};
        for(int k=0;k<3;k++)if(av[k]!=bv[k]){n++;pos=k;}
        if(n!=1)continue;
        int lo=av[pos]<bv[pos]?av[pos]:bv[pos],hi=av[pos]>bv[pos]?av[pos]:bv[pos],r=5;
        if(lo==0&&hi==1)r=0;else if(lo==2&&hi==3)r=1;
        else if(lo==0&&hi==2)r=2;else if(lo==1&&hi==3)r=3;else if(lo==0&&hi==3)r=4;
        if(table.count<263)table.values[table.count]={i,j,r,code[a]==code[b]};
        table.count++;
    }
    return table;
}
constexpr auto edges=make_edges();
static_assert(edges.count==263,"Code1 flow edge count");
static_assert(std::is_standard_layout<KaksModelStatistics>::value,"C statistics layout");
constexpr uint64_t bit(int i){return uint64_t(1)<<i;}
double nan(){return std::numeric_limits<double>::quiet_NaN();}
void unavailable(KaksModelStatistics &s) {
    s={};s.status=KAKS_STAT_UNAVAILABLE;
    std::fill_n(s.values,KAKS_STAT_VALUE_COUNT,nan());
    std::fill_n(s.rates,6,nan());std::fill_n(s.pi,64,nan());std::fill_n(s.x,7,nan());
}
void unavailable(KaksMaStatistics &s) {
    s={};s.status=KAKS_STAT_UNAVAILABLE;s.best_model=-1;
    std::fill_n(s.weights,14,nan());std::fill_n(s.rate_weight_masses,5,nan());unavailable(s.averaged);
}
void put(KaksModelStatistics &s,int i,double v) {
    if(!std::isfinite(v))throw std::invalid_argument("Nonfinite mathematical statistic");
    s.values[i]=v;s.math_mask|=bit(i);
}
/* Explicit byte fingerprint of copied data, avoiding struct padding/global
 * pointer lifetime. This is identity checking, not a cryptographic signature. */
uint64_t fingerprint(const PairData &p) {
    uint64_t h=14695981039346656037ull;
    auto add=[&](const void *ptr,size_t n){const auto *b=static_cast<const unsigned char *>(ptr);
        for(size_t i=0;i<n;i++){h^=b[i];h*=1099511628211ull;}};
    add(p.id().data(),p.id().size());const unsigned char separator=0;add(&separator,1);
    add(p.meta().data(),sizeof(int32_t)*4);add(p.values().data(),sizeof(double)*66);add(p.gc().data(),sizeof(double)*4);
    for(const auto &v:p.patterns()){add(&v.z0,1);add(&v.z1,1);add(&v.weight,sizeof(double));}
    return h;
}
struct Sum {double total=0.,correction=0.;void add(double v){double t=total+v;
    correction+=std::abs(total)>=std::abs(v)?(total-t)+v:(v-t)+total;total=t;}
    double value()const{return total+correction;}};
bool free_rate(uint32_t model,int rate) {
    const int family=model/2;
    return family>=1&&(rate<2||((family>=3)&&rate<4)||((family>=5)&&rate==4));
}
bool inside_source_box(const PairData &p,const ModelState &s) {
    double lower=1e-6;
    if(p.values()[1]/p.values()[0]>1e-6)lower=3*p.values()[1]/p.values()[0];
    if(s.x[0]<lower||s.x[0]>3.||s.omega<.001||s.omega>50.)return false;
    for(int i=0;i<s.config.nkappa;i++)if(s.x[i+1]<.01||s.x[i+1]>30.)return false;
    return true;
}
bool has(const KaksModelStatistics &s,int field){return (s.math_mask&bit(field))!=0;}
}

KaksModelStatistics statistics_from_state(const PairData &p,const ModelState &s,
    bool score_available,double lnL,bool is_ma) {
    KaksModelStatistics out;unavailable(out);out.model=s.config.id;
    out.support_policy=static_cast<uint32_t>(s.config.support_policy);
    out.pair_fingerprint=fingerprint(p);out.flags=KAKS_STAT_FIXED_PARAMETER_ONLY;
    std::copy(s.x.begin(),s.x.end(),out.x);std::copy(s.rates.begin(),s.rates.end(),out.rates);
    std::copy(s.pi.begin(),s.pi.end(),out.pi);
    const double L=p.meta()[2],snp=p.values()[1];double rs=0.,ra0=0.,ra=0.;
    /* Same source i/j order and association, on the exact immutable rates/pi
     * used by build_q_snapshot. No repeated expanded_rates call here. */
    for(const auto &e:edges.values) {
        double qij=s.rates[e.rate]*s.pi[e.j],flow=2*s.pi[e.i]*qij;
        if(e.synonymous)rs+=flow;else{ra0+=flow;ra+=flow*s.omega;}
    }
    const double mr=rs+ra,unselected=rs+ra0;
    if(!std::isfinite(mr)||!std::isfinite(unselected)||rs<0.||ra0<0.||ra<0.)
        throw std::invalid_argument("Nonfinite or negative stationary flow");
    put(out,2,s.x[0]);put(out,3,s.x[0]/3);put(out,4,s.omega);
    put(out,5,rs);put(out,6,ra0);put(out,7,ra);put(out,8,mr);
    put(out,18,L);put(out,19,snp);if(!is_ma)put(out,20,s.config.nkappa+2);
    if(snp==0.)out.flags|=KAKS_STAT_ZERO_OBSERVED;
    if(!inside_source_box(p,s))out.flags|=KAKS_STAT_OUTSIDE_SOURCE_BOX;
    if(s.x[0]==0.)out.flags|=KAKS_STAT_ZERO_TIME;
    if(mr==0.)out.flags|=KAKS_STAT_ZERO_SCALE;
    if(rs==0.)out.flags|=KAKS_STAT_NO_SYNONYMOUS_FLOW;
    if(ra0==0.)out.flags|=KAKS_STAT_NO_NONSYNONYMOUS_FLOW;
    if(unselected>0.) {
        const double s0=rs/unselected,n0=ra0/unselected;
        const double sites=3*L*s0;put(out,9,sites);put(out,10,3*L-sites);
        if(mr>0.) {
            put(out,14,L*s.x[0]*(rs/mr));put(out,15,L*s.x[0]*(ra/mr));
            if(rs>0.)put(out,11,s.x[0]*(rs/mr)/(3*s0));
            if(ra0>0.)put(out,12,s.x[0]*(ra/mr)/(3*n0));
            if(has(out,11)&&has(out,12)&&out.values[11]>0.)put(out,13,out.values[12]/out.values[11]);
            if(snp==0.){put(out,16,0.);put(out,17,0.);}
            else if(s.x[0]>0.){put(out,16,snp*(rs/mr));put(out,17,snp*(ra/mr));}
        }
    }
    if(score_available&&mr>0.) {
        if(!std::isfinite(lnL)||lnL>0.)throw std::invalid_argument("Supplied lnL must be finite and nonpositive, not positive F");
        put(out,0,lnL);
    }else out.flags|=KAKS_STAT_SCORE_UNAVAILABLE;
    if(is_ma)out.flags|=KAKS_STAT_MA_AICC_UNDEFINED;
    else {
        const double k=s.config.nkappa+2,denom=L-k-1;
        if(denom<=0.)out.flags|=KAKS_STAT_SHORT_AICC;
        else if(has(out,0))put(out,1,-2*lnL+2*k*L/denom);
    }
    out.report_mask=out.math_mask;
    if(snp==0.) {
        for(int i:{3,11,12,13,14,15,16,17})out.report_mask&=~bit(i);
    }
    const bool full_rates=has(out,11)&&has(out,12)&&has(out,13);
    out.status=(!full_rates||mr==0.)?KAKS_STAT_UNAVAILABLE:
        snp==0.?KAKS_STAT_UNIDENTIFIABLE:KAKS_STAT_AVAILABLE;
    return out;
}

KaksMaStatistics average_statistics(const PairData &p,const KaksModelStatistics *models,uint32_t count) {
    KaksMaStatistics out;unavailable(out);out.flags=KAKS_MA_FIXED_PARAMETER_ONLY|KAKS_MA_SCORE_PENDING;
    if(!models&&count)throw std::invalid_argument("Missing fixed model inputs");
    if(count>14)throw std::invalid_argument("Too many model inputs");
    std::array<KaksModelStatistics,14> canonical{};uint32_t support=0;
    for(uint32_t i=0;i<count;i++) {
        const auto &in=models[i];if(in.model>=14)throw std::invalid_argument("Unknown model input");
        const uint32_t mask=1u<<in.model;
        if(out.present_mask&mask)throw std::invalid_argument("Duplicate model input");
        out.present_mask|=mask;
        if(in.pair_fingerprint!=fingerprint(p))throw std::invalid_argument("Model statistics belong to a different pair");
        if(i&&in.support_policy!=support)throw std::invalid_argument("Mixed model support policies");
        support=in.support_policy;
        auto state=model_state(p,in.model,static_cast<SupportPolicy>(support),in.x);
        for(int j=0;j<6;j++)if(in.rates[j]!=state.rates[j])throw std::invalid_argument("Model rates do not match one raw expansion");
        for(int j=0;j<64;j++)if(in.pi[j]!=state.pi[j])throw std::invalid_argument("Model frequencies differ from declared prepared policy");
        auto stats=statistics_from_state(p,state,has(in,0),in.values[0]);canonical[in.model]=stats;
        if(!has(stats,0)||!has(stats,1))out.unavailable_mask|=mask;
    }
    if(out.present_mask!=0x3fff)out.flags|=KAKS_MA_INCOMPLETE;
    if(out.unavailable_mask)out.flags|=KAKS_MA_CANDIDATE_UNAVAILABLE;
    if(out.flags&(KAKS_MA_INCOMPLETE|KAKS_MA_CANDIDATE_UNAVAILABLE))return out;
    int best=0;for(int i=1;i<14;i++)if(canonical[i].values[1]<canonical[best].values[1])best=i;
    out.best_model=best;Sum total;
    for(int i=0;i<14;i++){out.weights[i]=std::exp(-.5*(canonical[i].values[1]-canonical[best].values[1]));total.add(out.weights[i]);}
    for(double &w:out.weights)w/=total.value();
    ModelState avg{};avg.config=model_config(13,static_cast<SupportPolicy>(support));
    Sum time,omega;
    for(int i=0;i<14;i++){time.add(out.weights[i]*canonical[i].x[0]);omega.add(out.weights[i]*canonical[i].values[4]);}
    avg.x[0]=time.value();avg.x[6]=avg.omega=omega.value();
    for(int r=0;r<5;r++) {
        Sum value,mass;
        for(int i=0;i<14;i++)if(free_rate(i,r)){value.add(out.weights[i]*canonical[i].rates[r]);mass.add(out.weights[i]);}
        out.rate_weight_masses[r]=mass.value();
        avg.x[r+1]=mass.value()<1e-50?1.:value.value()/mass.value();
        if(mass.value()<1e-50){out.rate_default_mask|=1u<<r;out.flags|=KAKS_MA_RATE_MASS_DEFAULTED;}
    }
    /* GTR mapping is identity for the first five rates, done once. */
    avg.rates=expanded_rates(13,avg.x.data());avg.kappa=avg.x[1];
    for(int c=0;c<64;c++){Sum freq;for(int i=0;i<14;i++)freq.add(out.weights[i]*canonical[i].pi[c]);avg.pi[c]=freq.value();}
    out.averaged=statistics_from_state(p,avg,false,nan(),true);
    if(out.averaged.flags&KAKS_STAT_ZERO_SCALE){out.flags|=KAKS_MA_PROCESS_UNAVAILABLE;return out;}
    out.status=out.averaged.status;
    return out;
}

ModelState averaged_model_state(const PairData &p,const KaksMaStatistics &ma) {
    if(ma.present_mask!=0x3fff||ma.unavailable_mask||ma.best_model<0||ma.best_model>=14||
       (ma.flags&(KAKS_MA_INCOMPLETE|KAKS_MA_CANDIDATE_UNAVAILABLE|KAKS_MA_PROCESS_UNAVAILABLE)))
        throw std::invalid_argument("MA parameters unavailable: every model is required");
    const auto &v=ma.averaged;
    if(v.pair_fingerprint!=fingerprint(p)||v.model!=13)throw std::invalid_argument("Wrong pair or averaged model metadata");
    ModelState s{};s.config=model_config(13,static_cast<SupportPolicy>(v.support_policy));
    for(int i=0;i<7;i++){if(!std::isfinite(v.x[i]))throw std::invalid_argument("Nonfinite averaged raw parameter");s.x[i]=v.x[i];}
    if(s.x[0]<0.||s.x[6]<=0.)throw std::invalid_argument("Invalid averaged time/omega");
    s.kappa=s.x[1];s.omega=s.x[6];
    for(int i=0;i<6;i++) {
        if(!std::isfinite(v.rates[i])||v.rates[i]<=0.||v.rates[i]!=(i<5?s.x[i+1]:1.))
            throw std::invalid_argument("Invalid averaged expanded rate vector");
        s.rates[i]=v.rates[i];
    }
    Sum total;
    for(int i=0;i<64;i++) {
        if(!std::isfinite(v.pi[i])||v.pi[i]<0.||v.pi[i]>1.)throw std::invalid_argument("Invalid averaged codon frequencies");
        s.pi[i]=v.pi[i];if(i<61)total.add(v.pi[i]);
        else if(s.config.support_policy==SupportPolicy::StrictPositive&&v.pi[i]!=0.)throw std::invalid_argument("Nonzero strict-positive padding");
    }
    if(std::abs(total.value()-1.)>1e-12)throw std::invalid_argument("Averaged codon frequencies do not sum to one");
    return s;
}
}

namespace {
int fail(const char *message,char *error,uint32_t cap) {
    if(error&&cap){size_t n=std::strlen(message);if(n>=cap)n=cap-1;std::memcpy(error,message,n);error[n]=0;}return 1;
}
void clear(char *error,uint32_t cap){if(error&&cap)error[0]=0;}
}
extern "C" uint32_t kaks_statistics_abi_version(void){return 1;}
extern "C" uint32_t kaks_statistics_model_size(void){return sizeof(KaksModelStatistics);}
extern "C" uint32_t kaks_statistics_ma_size(void){return sizeof(KaksMaStatistics);}
extern "C" int kaks_statistics_fixed(const void *raw,uint32_t model,uint32_t support,const double *x,
    uint32_t supplied,double lnL,KaksModelStatistics *out,char *error,uint32_t cap) {
    clear(error,cap);try {
        if(!raw||!out||!x||supplied>1)throw std::invalid_argument("Missing or invalid fixed-statistics input");
        const auto &p=*static_cast<const kakscuda::PairData *>(raw);
        auto state=kakscuda::model_state(p,model,static_cast<kakscuda::SupportPolicy>(support),x);
        *out=kakscuda::statistics_from_state(p,state,supplied!=0,lnL);return 0;
    }catch(const std::exception &e){return fail(e.what(),error,cap);}catch(...){return fail("Unknown fixed-statistics error",error,cap);}
}
extern "C" int kaks_statistics_average(const void *raw,const KaksModelStatistics *models,uint32_t count,
    KaksMaStatistics *out,char *error,uint32_t cap) {
    clear(error,cap);try {
        if(!raw||!out)throw std::invalid_argument("Missing MA input");
        *out=kakscuda::average_statistics(*static_cast<const kakscuda::PairData *>(raw),models,count);return 0;
    }catch(const std::exception &e){return fail(e.what(),error,cap);}catch(...){return fail("Unknown MA error",error,cap);}
}
extern "C" int kaks_statistics_ma_q(const void *raw,const KaksMaStatistics *ma,double *out,uint32_t count,
    char *error,uint32_t cap) {
    clear(error,cap);try {
        if(!raw||!ma||!out||count<KAKS_CORE_SNAPSHOT_WIDTH)throw std::invalid_argument("Missing or undersized MA-Q buffer");
        auto state=kakscuda::averaged_model_state(*static_cast<const kakscuda::PairData *>(raw),*ma);
        kakscuda::build_q_snapshot(state,out);return 0;
    }catch(const std::exception &e){return fail(e.what(),error,cap);}catch(...){return fail("Unknown MA-Q error",error,cap);}
}
extern "C" int kaks_statistics_ma_score(const void *raw,KaksMaStatistics *ma,double lnL,char *error,uint32_t cap) {
    clear(error,cap);try {
        if(!raw||!ma)throw std::invalid_argument("Missing MA score input");
        const auto &p=*static_cast<const kakscuda::PairData *>(raw);
        auto state=kakscuda::averaged_model_state(p,*ma);
        auto stats=kakscuda::statistics_from_state(p,state,true,lnL,true);
        if(!(stats.math_mask&1ull))throw std::invalid_argument("MA normalized process has no score");
        ma->averaged=stats;ma->flags&=~KAKS_MA_SCORE_PENDING;ma->status=stats.status;return 0;
    }catch(const std::exception &e){return fail(e.what(),error,cap);}catch(...){return fail("Unknown MA score error",error,cap);}
}
