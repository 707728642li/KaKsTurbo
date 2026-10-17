#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <float.h>
#include <ctype.h>
#include <cstring>
#include <vector>
#include <set>
#include <stdexcept>
#include <limits.h>
#include "fiber_optimizer.h"
#include "fiber_optimizer_bridge.h"
#include "GY94.h"

namespace {
const char *MODELS[14]={"JC","F81","K2P","HKY","TNEF","TN","K3P",
    "K3PUF","TIMEF","TIM","TVMEF","TVM","SYM","GTR"};
struct Pair { std::string id,a,b; double gc[4]; };
struct Globals {
    int code; unsigned long nt; std::string id; double gc[4];
    Globals() : code(genetic_code),nt(length),id(seq_name) { for(int i=0;i<4;i++) gc[i]=GC[i]; }
    ~Globals() { genetic_code=code; length=nt; seq_name=id; for(int i=0;i<4;i++) GC[i]=gc[i]; }
};
void activate(const Pair &p) {
    genetic_code=1; seq_name=p.id; length=static_cast<unsigned long>(p.a.size());
    for(int i=0;i<4;i++) GC[i]=p.gc[i];
}
void clean(Pair &p,const char *a,const char *b) {
    Base base; std::string left(a),right(b),joined=left+right;
    for(int i=0;i<4;i++) p.gc[i]=0;
    for(size_t i=0;i<joined.size();i+=3) for(int j=0;j<3;j++)
        if(joined[i+j]=='G'||joined[i+j]=='C') p.gc[j+1]++;
    double total=0; for(int j=1;j<4;j++) total+=p.gc[j];
    p.gc[0]=total/joined.size();
    for(int j=1;j<4;j++) p.gc[j]/=(joined.size()/3);
    for(size_t i=0;i<left.size();i+=3) {
        std::string ca=left.substr(i,3),cb=right.substr(i,3); bool drop=false;
        for(int j=0;j<3;j++) {
            ca[j]=static_cast<char>(toupper(static_cast<unsigned char>(ca[j])));
            cb[j]=static_cast<char>(toupper(static_cast<unsigned char>(cb[j])));
            if(base.convertChar(ca[j])==-1||base.convertChar(cb[j])==-1) drop=true;
        }
        if(!drop&&(base.getAminoAcid(ca)=='!'||base.getAminoAcid(cb)=='!')) drop=true;
        if(!drop) { p.a+=ca; p.b+=cb; }
    }
    if(p.a.empty()) throw std::invalid_argument("Paired masking removed every codon");
}
struct Task;
class Model : public GY94 {
    char *owned[2]; double workspace[256];
public:
    explicit Model(const char *name) : GY94(name) {
        owned[0]=owned[1]=com.z[0]=com.z[1]=0; com.space=0; GAMMA=0;
        // Only otherwise-unused ABI tails are initialized. Source arithmetic
        // overwrites all 61 active frequencies and six parsed rates itself.
        for(int i=0;i<64;i++) com.pi[i]=0;
        for(int i=0;i<8;i++) com.KAPPA[i]=1;
    }
    ~Model() { free(owned[0]); free(owned[1]); }
    void fit(const Pair &p) {
        str1=p.a; str2=p.b; preProcess(p.a.c_str(),p.b.c_str());
        owned[0]=com.z[0]; owned[1]=com.z[1];
        if(com.np<2||com.np>7) throw std::runtime_error("Unexpected source optimizer dimension");
        // Source ming2 needs at most 168 doubles including its integer tail.
        // The original Run's 800000-double allocation is merely spare capacity.
        com.space=workspace; com.sspace=256;
        PairwiseCodon(workspace);
    }
    void side_effects(const double *x) {
        com.kappa=x[1];
        for(int i=0;i<com.nkappa;i++) com.KAPPA[i]=x[i+1];
        com.omega=x[com.nkappa+1];
        if(model=="JC"||model=="K2P"||model=="TNEF"||model=="K3P"||
           model=="TIMEF"||model=="TVMEF"||model=="SYM")
            initArray(com.pi,64,1.0/(64-getNumNonsense(genetic_code)));
        parseSubRates(model,com.KAPPA);
    }
    double original(double *x,int np) { return lfun2dSdN(x,np); }
    int round() const { return Iround; }
    double observed_snp() const { return snp; }
    void snapshot(double *out,const double *raw_x) {
        for(int i=0;i<96;i++) out[i]=0;
        out[0]=lnL; out[1]=AICc; out[2]=3.*t; out[3]=com.omega;
        out[4]=Ka; out[5]=Ks; out[6]=S; out[7]=N; out[8]=Sd; out[9]=Nd; out[10]=t;
        for(int i=0;i<6;i++) out[11+i]=KAPPA[i];
        for(int i=0;i<64;i++) out[17+i]=com.pi[i];
        for(int i=0;i<7;i++) out[81+i]=raw_x[i];
    }
};
enum State { READY, WAITING, DONE, FAILED };
struct Session;
struct Task {
    Session *session; Pair *pair; uint32_t slot,model_id,cache_capacity;
    Model *model; void *fiber; State state; bool canceled,oracle;
    double key[64][7],cached[64]; int cached_warning[64]; uint32_t valid,next;
    double request_x[7],response,final_x[7],result[96]; int response_warning,warnings,opt,fit_round;
    uint64_t ticket,requests,hits; std::string failure;
    bool pinned_valid[3]; double pinned_key[3][7],pinned_value[3];
    int pinned_warning[3],last_warning; uint64_t pinned_hits[3],pinned_updates[3];
    Task(Session *s,Pair *p,uint32_t index,uint32_t m,uint32_t capacity)
        : session(s),pair(p),slot(index),model_id(m),cache_capacity(capacity),model(0),fiber(0),
          state(READY),canceled(false),oracle(false),valid(0),next(0),response(0),
          response_warning(0),warnings(0),opt(-1),fit_round(0),ticket(0),requests(0),hits(0),last_warning(0) {
        for(int i=0;i<3;i++){pinned_valid[i]=false;pinned_hits[i]=pinned_updates[i]=0;pinned_warning[i]=0;}
        const double initial[7]={.3,1,.5,.5,.5,.5,.3};
        for(int i=0;i<7;i++) request_x[i]=final_x[i]=initial[i];
        activate(*pair); model=new Model(MODELS[m]);
    }
    ~Task() { if(fiber) DeleteFiber(fiber); delete model; }
};
struct Session {
    DWORD owner; void *host; bool converted,started; uint32_t max_pairs,cache_capacity,reserve,commit;
    std::vector<Pair *> pairs; std::vector<Task *> tasks;
    Session() : owner(GetCurrentThreadId()),host(0),converted(false),started(false) {}
    ~Session() { for(size_t i=0;i<tasks.size();i++) delete tasks[i];
        for(size_t i=0;i<pairs.size();i++) delete pairs[i]; }
};
Session *live=0; Task *current=0; bool bypass=false;
// Session lifetime is shorter than a Python RequestBatch lifetime. Keep ticket
// identity unique across close/reopen, including reused opaque addresses.
uint64_t request_ticket_sequence=0;
CRITICAL_SECTION api_lock;
struct LockInit { LockInit(){InitializeCriticalSection(&api_lock);} ~LockInit(){DeleteCriticalSection(&api_lock);} } lock_init;
struct Lock { Lock(){EnterCriticalSection(&api_lock);} ~Lock(){LeaveCriticalSection(&api_lock);} };
struct Cancel : public std::exception { const char *what() const throw(){return "Fiber session canceled";} };
struct CurrentScope {
    Task *previous; bool old_bypass;
    CurrentScope(Task *t,bool b=false) : previous(current),old_bypass(bypass) { current=t; bypass=b; }
    ~CurrentScope(){ current=previous; bypass=old_bypass; }
};
int error_out(const std::string &message,char *error,uint32_t cap) {
    if(error&&cap) { size_t n=message.size(); if(n>=cap)n=cap-1;
        std::memcpy(error,message.data(),n); error[n]=0; }
    return 1;
}
Session *checked(void *p) {
    if(!p||p!=live) throw std::invalid_argument("Unknown or closed fiber session");
    if(live->owner!=GetCurrentThreadId()) throw std::invalid_argument("Fiber session requires its creating OS thread");
    if(GetCurrentFiber()!=live->host) throw std::invalid_argument("Call API from the owning host fiber");
    return live;
}
Task *pending(Session *s,int32_t slot,uint64_t ticket) {
    if(slot<0||static_cast<size_t>(slot)>=s->tasks.size()) throw std::invalid_argument("Invalid request slot");
    Task *t=s->tasks[slot];
    if(t->state!=WAITING||t->ticket!=ticket) throw std::invalid_argument("Stale or nonpending request");
    return t;
}
void advance(Task *t) {
    activate(*t->pair); CurrentScope scope(t); SwitchToFiber(t->fiber);
    if(t->state==FAILED&&!t->canceled) throw std::runtime_error(t->failure);
}
VOID WINAPI task_main(void *value) {
    Task *t=static_cast<Task *>(value);
    try { t->model->fit(*t->pair); t->model->snapshot(t->result,t->final_x); t->state=DONE; }
    catch(const std::exception &e) { t->failure=e.what(); t->state=FAILED; }
    catch(...) { t->failure="Unknown optimizer fiber exception"; t->state=FAILED; }
    SwitchToFiber(t->session->host);
    // Completed fibers are deleted; accidentally switching one again is fatal.
    abort();
}
void validate_rows(Session *s,uint32_t count,const int32_t *slots,const uint64_t *tickets) {
    if(count&&(!slots||!tickets)) throw std::invalid_argument("Null request buffer");
    std::set<int32_t> unique;
    for(uint32_t i=0;i<count;i++) { pending(s,slots[i],tickets[i]);
        if(!unique.insert(slots[i]).second) throw std::invalid_argument("Duplicate request slot"); }
}
}

bool kaks_fiber_objective(void *raw,double *x,int np,double *objective) {
    if(bypass||!current) return false;
    Task *t=current;
    if(t->model!=raw||np!=t->model->com.np||np<2||np>7) throw std::runtime_error("Optimizer hook identity/dimension mismatch");
    t->model->side_effects(x);
    for(int entry=0;entry<3;entry++) if(t->pinned_valid[entry]&&std::memcmp(t->pinned_key[entry],x,np*sizeof(double))==0) {
        t->last_warning=t->pinned_warning[entry];t->warnings+=t->last_warning;
        t->hits++;t->pinned_hits[entry]++;*objective=t->pinned_value[entry];return true;
    }
    for(uint32_t entry=0;entry<t->valid;entry++) if(std::memcmp(t->key[entry],x,np*sizeof(double))==0) {
        t->last_warning=t->cached_warning[entry]; t->warnings+=t->last_warning; t->hits++; *objective=t->cached[entry]; return true;
    }
    double value; int warning;
    if(t->oracle) {
        int before=t->warnings; CurrentScope scope(t,true);
        value=t->model->original(x,np); warning=t->warnings-before;
    } else {
        const double initial[7]={.3,1,.5,.5,.5,.5,.3};
        for(int i=0;i<7;i++) t->request_x[i]=i<np?x[i]:initial[i];
        t->ticket=++request_ticket_sequence; t->requests++; t->state=WAITING;
        SwitchToFiber(t->session->host);
        if(t->canceled) throw Cancel();
        value=t->response; warning=t->response_warning; t->warnings+=warning;
    }
    if(t->cache_capacity) {
        uint32_t entry=t->next;
        std::memcpy(t->key[entry],x,np*sizeof(double)); t->cached[entry]=value;
        t->cached_warning[entry]=warning;
        if(t->valid<t->cache_capacity)t->valid++;
        t->next=(entry+1)%t->cache_capacity;
    }
    t->last_warning=warning; *objective=value; return true;
}
void kaks_fiber_record_fit(void *raw,int result,const double *x,int np,int round) {
    if(!current||current->model!=raw||np!=current->model->com.np) throw std::runtime_error("Unexpected fit hook");
    current->opt=result; current->fit_round=round;
    for(int i=0;i<7;i++) current->final_x[i]=x[i];
}
void kaks_fiber_record_eigen(int status) { if(status&&current) current->warnings++; }
void kaks_fiber_pin_objective(void *raw,const double *x,int np,double value,int entry) {
    if(!current||current->model!=raw||np!=current->model->com.np||np<2||np>7||entry<0||entry>2||!_finite(value))
        throw std::runtime_error("Invalid protected objective entry");
    Task *t=current;
    std::memcpy(t->pinned_key[entry],x,np*sizeof(double));
    t->pinned_value[entry]=value;t->pinned_warning[entry]=t->last_warning;
    t->pinned_valid[entry]=true;t->pinned_updates[entry]++;
}

#define FIBER_BEGIN Lock lock; if(error&&error_capacity) error[0]=0; try
#define FIBER_END catch(const std::exception &e){return error_out(e.what(),error,error_capacity);} catch(...){return error_out("Unknown fiber API exception",error,error_capacity);} return 0
extern "C" {
uint32_t kaks_fiber_abi_version(void){return KAKS_FIBER_ABI_VERSION;}
int kaks_fiber_create(uint32_t cache_entries,uint32_t reserve,uint32_t commit,uint32_t max_pairs,
                     void **out,char *error,uint32_t error_capacity) {
    FIBER_BEGIN {
        if(out)*out=0;
        if(live||!out||cache_entries>64||reserve<131072||commit<4096||commit>reserve||!max_pairs||max_pairs>1000)
            throw std::invalid_argument("Invalid session options or an existing active session");
        Session *s=new Session;
        s->max_pairs=max_pairs; s->cache_capacity=cache_entries; s->reserve=reserve; s->commit=commit;
        if(IsThreadAFiber()) s->host=GetCurrentFiber();
        else { s->host=ConvertThreadToFiberEx(0,FIBER_FLAG_FLOAT_SWITCH); s->converted=true; }
        if(!s->host) { delete s; throw std::runtime_error("ConvertThreadToFiberEx failed"); }
        live=s; *out=s;
    } FIBER_END;
}
int kaks_fiber_add_pair(void *raw,const char *id,const char *a,const char *b,uint32_t *pair_index,
                        char *error,uint32_t error_capacity) {
    FIBER_BEGIN {
        Session *s=checked(raw); Globals restore;
        if(s->started||s->pairs.size()>=s->max_pairs||!id||!a||!b||!pair_index)
            throw std::invalid_argument("Add pairs before polling; check pair capacity and buffers");
        size_t n=std::strlen(a);
        if(!*id||std::strpbrk(id,"\t\r\n")||!n||n!=std::strlen(b)||n%3||n>INT_MAX/2||
           std::strpbrk(a,"\r\n\t ")||std::strpbrk(b,"\r\n\t ")) throw std::invalid_argument("Invalid AXT pair");
        genetic_code=1; Pair *p=new Pair; p->id=id;
        const size_t old_count=s->tasks.size();
        try {
            clean(*p,a,b);
            for(uint32_t m=0;m<14;m++) {
                Task *t=new Task(s,p,static_cast<uint32_t>(old_count+m),m,s->cache_capacity);
                t->fiber=CreateFiberEx(s->commit,s->reserve,FIBER_FLAG_FLOAT_SWITCH,task_main,t);
                if(!t->fiber) { delete t; throw std::runtime_error("CreateFiberEx failed"); }
                s->tasks.push_back(t);
            }
            *pair_index=static_cast<uint32_t>(s->pairs.size()); s->pairs.push_back(p);
        } catch(...) {
            while(s->tasks.size()>old_count) { delete s->tasks.back(); s->tasks.pop_back(); }
            delete p; throw;
        }
    } FIBER_END;
}
int kaks_fiber_poll(void *raw,uint32_t capacity,int32_t *meta,double *x,uint64_t *tickets,
                    uint32_t *count,uint32_t *completed,char *error,uint32_t error_capacity) {
    FIBER_BEGIN {
        Session *s=checked(raw); Globals restore;
        if(!capacity||!meta||!x||!tickets||!count||!completed) throw std::invalid_argument("Null/empty poll output");
        s->started=true; *count=0; *completed=0;
        // Advance every ready task once to expose a broad batch. A single task
        // can traverse exact-cache hits before its next uncached request.
        for(size_t i=0;i<s->tasks.size();i++) if(s->tasks[i]->state==READY) advance(s->tasks[i]);
        for(size_t i=0;i<s->tasks.size();i++) {
            Task *t=s->tasks[i]; if(t->state==DONE){(*completed)++;continue;}
            if(t->state==FAILED) throw std::runtime_error(t->failure);
            if(t->state!=WAITING||*count==capacity)continue;
            uint32_t row=(*count)++; int32_t *m=meta+row*6;
            m[0]=t->slot; m[1]=t->slot/14; m[2]=t->model_id; m[3]=t->model->com.np; m[4]=t->model->round(); m[5]=1;
            for(int j=0;j<7;j++)x[row*7+j]=t->request_x[j]; tickets[row]=t->ticket;
        }
    } FIBER_END;
}
int kaks_fiber_submit(void *raw,uint32_t count,const int32_t *slots,const uint64_t *tickets,
                      const double *objectives,const int32_t *warnings,char *error,uint32_t error_capacity) {
    FIBER_BEGIN {
        Session *s=checked(raw); validate_rows(s,count,slots,tickets);
        if(count&&!objectives) throw std::invalid_argument("Null objective input");
        for(uint32_t i=0;i<count;i++) if(!_finite(objectives[i])||(warnings&&warnings[i]<0))
            throw std::invalid_argument("Nonfinite objective or negative warning delta");
        for(uint32_t i=0;i<count;i++) { Task *t=pending(s,slots[i],tickets[i]);
            t->response=objectives[i]; t->response_warning=warnings?warnings[i]:0; t->state=READY; }
    } FIBER_END;
}
int kaks_fiber_mock_evaluate(void *raw,uint32_t count,const int32_t *slots,const uint64_t *tickets,
                             double *objectives,int32_t *warnings,char *error,uint32_t error_capacity) {
    FIBER_BEGIN {
        Session *s=checked(raw); Globals restore; validate_rows(s,count,slots,tickets);
        if(count&&(!objectives||!warnings))throw std::invalid_argument("Null mock output");
        for(uint32_t i=0;i<count;i++) {
            Task *t=pending(s,slots[i],tickets[i]); activate(*t->pair); CurrentScope scope(t,true);
            int before=t->warnings;
            objectives[i]=t->model->original(t->request_x,t->model->com.np);
            warnings[i]=t->warnings-before; t->warnings=before;
        }
    } FIBER_END;
}
int kaks_fiber_results(void *raw,uint32_t capacity,double *out,int32_t *statuses,char *error,uint32_t error_capacity) {
    FIBER_BEGIN {
        Session *s=checked(raw);
        if(capacity<s->tasks.size()||!out||!statuses)throw std::invalid_argument("Result capacity too small");
        for(size_t i=0;i<s->tasks.size();i++) if(s->tasks[i]->state!=DONE)throw std::invalid_argument("All models must finish before exporting results");
        for(size_t i=0;i<s->tasks.size();i++) { Task *t=s->tasks[i];
            std::memcpy(out+i*96,t->result,96*sizeof(double));
            statuses[i*4]=t->slot; statuses[i*4+1]=t->opt; statuses[i*4+2]=t->fit_round; statuses[i*4+3]=t->warnings; }
    } FIBER_END;
}
int kaks_fiber_cpu_oracle(void *raw,uint32_t pair,uint32_t model,double *out,int32_t *status,char *error,uint32_t error_capacity) {
    FIBER_BEGIN {
        Session *s=checked(raw); Globals restore;
        if(pair>=s->pairs.size()||model>=14||!out||!status)throw std::invalid_argument("Invalid oracle pair/model/output");
        Task t(s,s->pairs[pair],pair*14+model,model,0); t.oracle=true; CurrentScope scope(&t);
        t.model->fit(*t.pair); t.model->snapshot(out,t.final_x);
        status[0]=t.slot; status[1]=t.opt; status[2]=t.fit_round; status[3]=t.warnings;
    } FIBER_END;
}
int kaks_fiber_metrics(void *raw,uint32_t capacity,uint64_t *out,char *error,uint32_t error_capacity) {
    FIBER_BEGIN {
        Session *s=checked(raw); if(capacity<s->tasks.size()||!out)throw std::invalid_argument("Metrics capacity too small");
        for(size_t i=0;i<s->tasks.size();i++) { Task *t=s->tasks[i];
            out[i*6]=t->requests; out[i*6+1]=t->hits; out[i*6+2]=t->valid;
            out[i*6+3]=t->state==DONE; out[i*6+4]=t->model->round(); out[i*6+5]=t->warnings; }
    } FIBER_END;
}
__declspec(dllexport) int kaks_fiber_protected_cache_metrics(void *raw,uint32_t capacity,uint64_t *out,char *error,uint32_t error_capacity) {
    FIBER_BEGIN {
        Session *s=checked(raw);if(capacity<s->tasks.size()||!out)throw std::invalid_argument("Protected metric capacity too small");
        for(size_t i=0;i<s->tasks.size();i++)for(int j=0;j<2;j++) {
            out[i*4+j]=s->tasks[i]->pinned_hits[j];out[i*4+2+j]=s->tasks[i]->pinned_updates[j];
        }
    } FIBER_END;
}
int kaks_fiber_close(void *raw,char *error,uint32_t error_capacity) {
    FIBER_BEGIN {
        Session *s=checked(raw); Globals restore;
        for(size_t i=0;i<s->tasks.size();i++) { Task *t=s->tasks[i];
            if(t->state==WAITING||(t->state==READY&&t->requests)) { t->canceled=true; advance(t); }
        }
        bool converted=s->converted; delete s; live=0;
        if(converted&&!ConvertFiberToThread())throw std::runtime_error("ConvertFiberToThread failed after cleanup");
    } FIBER_END;
}
}
