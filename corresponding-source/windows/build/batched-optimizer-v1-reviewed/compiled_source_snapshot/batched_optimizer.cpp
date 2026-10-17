/* New optimizer implementation, independent of original ming2 source.
 * BFGS and Armijo are standard methods. No scientific global-optimum claim. */
#include "batched_optimizer.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <deque>
#include <limits>
#include <memory>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>
namespace {
using V=std::array<double,7>;
using H=std::array<double,49>;
enum class Phase { Initial, Gradient, Line, Quality, Done };
struct Request { uint64_t ticket; uint32_t job,key; int64_t slot; V y,x; };
struct Job {
    int64_t slot; uint32_t n,iteration=0,evaluations=0,resets=0,quality_rounds=0;
    int status=0; Phase phase=Phase::Initial;
    V x,y,lo,hi,rawlo,rawhi,g{},oldg{},oldy{},p{},bestx{},besty{},probe_lo{},probe_hi{};
    H h{};
    double f=0.,initial_f=0.,bestf=std::numeric_limits<double>::infinity();
    double scale,projected=std::numeric_limits<double>::infinity(),quality=0.;
    double alpha=1.,slope=0.,oldf=0.; uint32_t remaining=0,line_trials=0;
    bool have_previous=false;
    std::vector<double> responses; std::vector<V> points;
};
struct Context {
    std::thread::id owner=std::this_thread::get_id(); uint32_t max_jobs,max_iterations;
    double gradient_tolerance,quality_tolerance;uint64_t next_ticket=1;
    std::vector<Job> jobs;std::deque<Request> queued;
    std::unordered_map<uint64_t,Request> issued;
    bool started=false;
    Context(uint32_t m,uint32_t i,double g,double q):max_jobs(m),max_iterations(i),gradient_tolerance(g),quality_tolerance(q){jobs.reserve(m);}
};
Context &context(void *p){if(!p)throw std::invalid_argument("Null optimizer context");auto &c=*static_cast<Context*>(p);
    if(c.owner!=std::this_thread::get_id())throw std::invalid_argument("Optimizer context requires creating thread");
    return c;}
int error(const char *s,char *p,uint32_t n){if(p&&n){size_t k=std::min(std::strlen(s),static_cast<size_t>(n-1));std::memcpy(p,s,k);p[k]=0;}return 1;}
void clear(char *p,uint32_t n){if(p&&n)p[0]=0;}
V raw(const Job &j,const V &y){V x=j.x;for(uint32_t i=0;i<j.n;i++)x[i]=std::clamp(std::exp(y[i]),j.rawlo[i],j.rawhi[i]);return x;}
void reset(Job &j){j.h.fill(0.);for(uint32_t i=0;i<j.n;i++)j.h[i*7+i]=1.;j.resets++;}
void enqueue(Context &c,uint32_t index,uint32_t key,const V &y){
    if(c.next_ticket==0)throw std::overflow_error("Optimizer ticket exhausted");
    auto &j=c.jobs[index];c.queued.push_back({c.next_ticket++,index,key,j.slot,y,raw(j,y)});j.remaining++;
}
void gradient(Context &c,uint32_t index){auto &j=c.jobs[index];j.phase=Phase::Gradient;
    j.responses.assign(j.n*2,0.);j.points.clear();j.remaining=0;
    for(uint32_t k=0;k<j.n;k++){
        const double step=2e-5*(1.+std::abs(j.y[k]));
        V minus=j.y,plus=j.y;minus[k]=std::max(j.lo[k],j.y[k]-step);plus[k]=std::min(j.hi[k],j.y[k]+step);
        j.probe_lo[k]=minus[k];j.probe_hi[k]=plus[k];
        enqueue(c,index,k*2,minus);enqueue(c,index,k*2+1,plus);
    }
}
void quality(Context &c,uint32_t index){auto &j=c.jobs[index];j.phase=Phase::Quality;
    if(j.bestf<j.f){j.y=j.besty;j.x=j.bestx;j.f=j.bestf;j.have_previous=false;}
    j.responses.clear();j.points.clear();j.remaining=0;j.quality_rounds++;
    for(double step:{1e-3,1e-2,1e-1})for(uint32_t k=0;k<j.n;k++)for(int sign:{-1,1}){
        V probe=j.y;probe[k]=std::clamp(j.y[k]+sign*step,j.lo[k],j.hi[k]);
        if(probe[k]==j.y[k])continue;
        const uint32_t key=static_cast<uint32_t>(j.points.size());j.points.push_back(probe);j.responses.push_back(0.);enqueue(c,index,key,probe);
    }
    if(!j.remaining){j.quality=0.;j.status=1;j.phase=Phase::Done;}
}
void line(Context &c,uint32_t index){auto &j=c.jobs[index];j.phase=Phase::Line;j.remaining=0;j.responses.assign(1,0.);
    V trial=j.y;for(uint32_t i=0;i<j.n;i++)trial[i]=std::clamp(j.y[i]+j.alpha*j.p[i],j.lo[i],j.hi[i]);
    enqueue(c,index,0,trial);
}
double projected(Job &j){double value=0.;for(uint32_t i=0;i<j.n;i++){
    double g=j.g[i];if((j.y[i]<=j.lo[i]+1e-12&&g>0.)||(j.y[i]>=j.hi[i]-1e-12&&g<0.))g=0.;value=std::max(value,std::abs(g));
}return value;}
void direction(Context &c,uint32_t index){auto &j=c.jobs[index];j.projected=projected(j);
    if(j.projected<=c.gradient_tolerance||j.iteration>=c.max_iterations){quality(c,index);return;}
    for(uint32_t i=0;i<j.n;i++){
        double p=0.;for(uint32_t k=0;k<j.n;k++)p-=j.h[i*7+k]*j.g[k];
        if((j.y[i]<=j.lo[i]+1e-12&&p<0.)||(j.y[i]>=j.hi[i]-1e-12&&p>0.))p=0.;
        j.p[i]=p;
    }
    auto compute_slope=[&](){double s=0.;for(uint32_t i=0;i<j.n;i++)s+=j.g[i]*j.p[i];return s;};
    j.slope=compute_slope();
    if(!(j.slope<0.)||!std::isfinite(j.slope)){reset(j);for(uint32_t i=0;i<j.n;i++){
        j.p[i]=-j.g[i];if((j.y[i]<=j.lo[i]+1e-12&&j.p[i]<0.)||(j.y[i]>=j.hi[i]-1e-12&&j.p[i]>0.))j.p[i]=0.;
    }j.slope=compute_slope();}
    double size=0.;for(uint32_t i=0;i<j.n;i++)size=std::max(size,std::abs(j.p[i]));
    if(!(size>0.)||!(j.slope<0.)){quality(c,index);return;}
    if(size>1.)for(uint32_t i=0;i<j.n;i++)j.p[i]/=size;
    j.slope=compute_slope();
    if(!std::isfinite(j.slope)){j.status=3;j.phase=Phase::Done;return;}
    j.alpha=1.;j.line_trials=0;line(c,index);
}
void advance(Context &c,uint32_t index){auto &j=c.jobs[index];if(j.remaining)return;
    switch(j.phase){
    case Phase::Initial:j.f=j.responses[0];j.initial_f=j.f;gradient(c,index);break;
    case Phase::Gradient:{
        for(uint32_t i=0;i<j.n;i++){
            const long double value=(static_cast<long double>(j.responses[i*2+1])-j.responses[i*2])/(j.probe_hi[i]-j.probe_lo[i])/j.scale;
            j.g[i]=static_cast<double>(value);
            if(!std::isfinite(j.g[i])){j.status=3;j.phase=Phase::Done;return;}
        }
        if(j.have_previous){
            V s{},v{},hv{};double sv=0.,vHv=0.;
            for(uint32_t i=0;i<j.n;i++){s[i]=j.y[i]-j.oldy[i];v[i]=j.g[i]-j.oldg[i];sv+=s[i]*v[i];}
            if(sv>1e-14){
                for(uint32_t i=0;i<j.n;i++){for(uint32_t k=0;k<j.n;k++)hv[i]+=j.h[i*7+k]*v[k];vHv+=v[i]*hv[i];}
                for(uint32_t i=0;i<j.n;i++)for(uint32_t k=0;k<j.n;k++)
                    j.h[i*7+k]+=(1.+vHv/sv)*(s[i]*s[k])/sv-(s[i]*hv[k]+hv[i]*s[k])/sv;
                if(!std::all_of(j.h.begin(),j.h.end(),[](double x){return std::isfinite(x);}))reset(j);
            }else reset(j);
            j.have_previous=false;
        }
        direction(c,index);break;}
    case Phase::Line:{
        const double f=j.responses[0];double actual_slope=0.;V trial=j.y;
        for(uint32_t i=0;i<j.n;i++){trial[i]=std::clamp(j.y[i]+j.alpha*j.p[i],j.lo[i],j.hi[i]);actual_slope+=j.g[i]*(trial[i]-j.y[i]);}
        if(f<j.f&&f<=j.f+1e-4*actual_slope*j.scale){
            j.oldy=j.y;j.oldg=j.g;j.oldf=j.f;j.y=trial;j.x=raw(j,trial);j.f=f;j.iteration++;j.have_previous=true;gradient(c,index);
        }else if(++j.line_trials<24){j.alpha*=.5;line(c,index);}
        else {quality(c,index);}break;}
    case Phase::Quality:{
        uint32_t best=0;double bestf=j.f;bool improve=false;
        for(uint32_t i=0;i<j.responses.size();i++)if(j.responses[i]<bestf){bestf=j.responses[i];best=i;improve=true;}
        j.quality=j.f-bestf;
        if(improve&&j.quality>c.quality_tolerance&&j.iteration<c.max_iterations&&j.quality_rounds<64){
            j.y=j.points[best];j.x=raw(j,j.y);j.f=bestf;j.have_previous=false;j.iteration++;reset(j);gradient(c,index);
        }else {j.status=j.quality<=c.quality_tolerance?1:2;j.phase=Phase::Done;}
        break;}
    case Phase::Done:break;
    }
}
}
extern "C" uint32_t kaks_opt_abi_version(void){return 1;}
extern "C" int kaks_opt_create(uint32_t max_jobs,uint32_t max_iterations,double g,double q,void **out,char *e,uint32_t n){
    clear(e,n);if(out)*out=nullptr;try{
        if(!out||!max_jobs||max_jobs>100000||!max_iterations||max_iterations>10000||!std::isfinite(g)||g<=0.||!std::isfinite(q)||q<=0.)throw std::invalid_argument("Invalid optimizer configuration");
        *out=new Context(max_jobs,max_iterations,g,q);return 0;
    }catch(const std::exception &v){return error(v.what(),e,n);}catch(...){return error("Optimizer create failed",e,n);}}
extern "C" int kaks_opt_add(void *p,int64_t slot,uint32_t dim,const double *x,const double *lo,const double *hi,double scale,char *e,uint32_t n){
    clear(e,n);try{auto &c=context(p);if(c.started||c.jobs.size()>=c.max_jobs||slot<0||dim<1||dim>7||!x||!lo||!hi||!std::isfinite(scale)||scale<=0.)throw std::invalid_argument("Invalid optimizer job");
        for(const auto &j:c.jobs)if(j.slot==slot)throw std::invalid_argument("Duplicate optimizer job slot");
        Job j{};j.slot=slot;j.n=dim;j.scale=scale;
        for(uint32_t i=0;i<7;i++){
            if(!std::isfinite(x[i]))throw std::invalid_argument("Nonfinite optimizer initial parameter");
            j.x[i]=x[i];j.bestx[i]=x[i];
            if(i<dim){if(!std::isfinite(lo[i])||!std::isfinite(hi[i])||lo[i]<=0.||!(hi[i]>lo[i])||x[i]<lo[i]||x[i]>hi[i])throw std::invalid_argument("Invalid positive box or infeasible initial point");j.rawlo[i]=lo[i];j.rawhi[i]=hi[i];j.lo[i]=std::log(lo[i]);j.hi[i]=std::log(hi[i]);
                if(!(j.hi[i]>j.lo[i]))throw std::invalid_argument("Positive raw box collapses at log-coordinate precision");
                j.y[i]=std::clamp(std::log(x[i]),j.lo[i],j.hi[i]);}
        }
        reset(j);j.responses.assign(1,0.);c.jobs.push_back(j);enqueue(c,static_cast<uint32_t>(c.jobs.size()-1),0,j.y);return 0;
    }catch(const std::exception &v){return error(v.what(),e,n);}catch(...){return error("Optimizer add failed",e,n);}}
extern "C" int kaks_opt_poll(void *p,uint32_t cap,int64_t *slots,uint64_t *tickets,double *x,uint32_t *count,uint32_t *done,char *e,uint32_t n){
    clear(e,n);try{auto &c=context(p);if(!cap||!slots||!tickets||!x||!count||!done)throw std::invalid_argument("Invalid optimizer poll buffers");c.started=true;*count=0;*done=0;
        for(const auto &j:c.jobs)if(j.phase==Phase::Done)(*done)++;
        while(*count<cap&&!c.queued.empty()){
            Request r=c.queued.front();c.queued.pop_front();const uint32_t k=(*count)++;slots[k]=r.slot;tickets[k]=r.ticket;std::copy(r.x.begin(),r.x.end(),x+k*7);c.issued.emplace(r.ticket,r);
        }return 0;
    }catch(const std::exception &v){return error(v.what(),e,n);}catch(...){return error("Optimizer poll failed",e,n);}}
extern "C" int kaks_opt_submit(void *p,uint32_t count,const int64_t *slots,const uint64_t *tickets,const double *f,char *e,uint32_t n){
    clear(e,n);try{auto &c=context(p);if(!count||!slots||!tickets||!f)throw std::invalid_argument("Invalid optimizer submit buffers");std::unordered_set<uint64_t> seen;
        for(uint32_t k=0;k<count;k++){
            const auto it=c.issued.find(tickets[k]);if(it==c.issued.end()||it->second.slot!=slots[k]||!seen.insert(tickets[k]).second)throw std::invalid_argument("Unknown, duplicate, or misbound optimizer ticket");
            if(!std::isfinite(f[k]))throw std::invalid_argument("Nonfinite objective rejected before mutation");
        }
        for(uint32_t k=0;k<count;k++){
            const Request r=c.issued.at(tickets[k]);auto &j=c.jobs[r.job];c.issued.erase(tickets[k]);j.responses.at(r.key)=f[k];j.evaluations++;j.remaining--;
            if(j.phase==Phase::Initial){j.x=r.x;j.y=r.y;}
            if(f[k]<j.bestf){j.bestf=f[k];j.bestx=r.x;j.besty=r.y;}
            advance(c,r.job);
        }return 0;
    }catch(const std::exception &v){return error(v.what(),e,n);}catch(...){return error("Optimizer submit failed",e,n);}}
extern "C" int kaks_opt_results(void *p,uint32_t cap,double *out,char *e,uint32_t n){
    clear(e,n);try{auto &c=context(p);if(!out||cap<c.jobs.size())throw std::invalid_argument("Optimizer result capacity too small");
        for(size_t k=0;k<c.jobs.size();k++){const auto &j=c.jobs[k];double *r=out+k*18;std::copy(j.x.begin(),j.x.end(),r);r[7]=j.f;r[8]=j.initial_f;r[9]=j.projected;r[10]=j.quality;r[11]=j.iteration;r[12]=j.evaluations;r[13]=j.quality_rounds;r[14]=j.status;r[15]=j.resets;r[16]=j.bestf;r[17]=j.f-j.bestf;}
        return 0;
    }catch(const std::exception &v){return error(v.what(),e,n);}catch(...){return error("Optimizer results failed",e,n);}}
extern "C" void kaks_opt_free(void *p){delete static_cast<Context*>(p);}
