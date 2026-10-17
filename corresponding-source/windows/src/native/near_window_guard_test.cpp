// Executes the actual production near_impl with a tiny deterministic provider.
// Does not fit a biological model or initialize CUDA. Export-only scaffolding
// permits including the unchanged implementation as a standalone unit.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <float.h>
#include <stdint.h>
#include <cmath>
#include <cstring>
#include <map>
#include <vector>
#include <stdexcept>
#include <cstdio>
#include "fiber_optimizer_near_bridge.h"
struct Task;
Task *current=0;
double provider_value=10.;int provider_warning=17,provider_calls=0,exit_calls=0;
struct Model {
    struct {int np;} com;
    Model(){com.np=2;}
    double original(double *,int);
};
struct Task {
    Model *model;bool oracle,pinned_valid[3];double pinned_key[3][7],pinned_value[3];
    int last_warning,pinned_warning[3],state;
    Task(Model *m):model(m),oracle(false),last_warning(0),state(2) {
        for(int i=0;i<3;i++)pinned_valid[i]=false;
    }
};
double Model::original(double *,int) {provider_calls++;current->last_warning=provider_warning;return provider_value;}
bool stable_inside(const double *x,const double xb[][2],int n) {
    for(int i=0;i<n;i++)if(!_finite(x[i])||x[i]<xb[i][0]||x[i]>xb[i][1])return false;
    return true;
}
void kaks_fiber_pin_objective(void *raw,const double *x,int n,double f,int entry) {
    if(current->model!=raw||entry!=2)throw std::runtime_error("Wrong test pin");
    std::memcpy(current->pinned_key[entry],x,n*sizeof(double));
    current->pinned_value[entry]=f;current->pinned_warning[entry]=current->last_warning;
    current->pinned_valid[entry]=true;
}
bool kaks_fiber_record_stagnation(void *,int,int streak) {
    if(streak!=64)throw std::runtime_error("Wrong test threshold");exit_calls++;return true;
}
enum { DONE=2 };
struct Session {std::vector<Task *> tasks;};
Session *checked(void *){static Session s;return &s;}
#define FIBER_BEGIN try
#define FIBER_END catch(...){return 1;} return 0
#include "fiber_optimizer_near_impl.inc"

int assertions=0;
void require(bool condition,const char *what) {
    assertions++;if(!condition)throw std::runtime_error(what);
}
struct Input {
    double a[2],x[2],g[2],f,incoming_f,size,box[2][2];int marks[2],ix[2];
    Input():f(10.),incoming_f(10.),size(2.) {
        a[0]=x[0]=.5;a[1]=x[1]=1.;g[0]=g[1]=.1;
        marks[0]=marks[1]=0;ix[0]=0;ix[1]=1;
        box[0][0]=1e-6;box[0][1]=3.;box[1][0]=.001;box[1][1]=50.;
    }
    bool step(Model &m,KaksNearWindow &w,int round,bool identity=true) {
        return kaks_fiber_near_step(&m,w,round,2,identity,a,incoming_f,g,size,
            marks,ix,2,x,&f,g,size,marks,ix,2,box,1e-6,1e-6);
    }
};
int main() {
    try {
        Model m;Task t(&m);current=&t;KaksNearWindow w;Input p;
        kaks_fiber_near_begin(&m,w);
        for(int i=1;i<64;i++)require(!p.step(m,w,i),"Premature early signal");
        require(p.step(m,w,64),"Eligible window failed to signal at 64");
        require(w.count==64&&t.pinned_valid[2]&&t.pinned_warning[2]==17,"Exact warning pin missing");
        require(p.f==10.&&std::memcmp(p.x,t.pinned_key[2],2*sizeof(double))==0,"Best restoration failed");
        const int old_exits=exit_calls;
        kaks_fiber_near_begin(&m,w);p=Input();
        for(int i=1;i<30;i++)require(!p.step(m,w,i),"Premature reset setup");
        require(!p.step(m,w,30,false)&&w.count==0&&!t.pinned_valid[2],"Nonidentity did not clear");
        for(int i=31;i<64;i++)require(!p.step(m,w,i),"Old window survived invalid branch");
        kaks_fiber_near_clear(&m,w);require(w.count==0&&!t.pinned_valid[2],"Nonreset branch did not clear");
        p=Input();p.step(m,w,1);p.marks[0]=-1;
        require(!p.step(m,w,2)&&w.count==0,"Working-set change did not clear");
        p=Input();p.step(m,w,1);p.x[0]+=.8e-6;provider_value=10.;
        require(!p.step(m,w,2)&&w.count==2,"Within-span point rejected");
        p.a[0]=p.x[0];p.x[0]=.5-.8e-6;
        require(!p.step(m,w,3)&&w.count==0,"Interior span was ignored");
        p=Input();kaks_fiber_near_begin(&m,w);
        for(int i=1;i<=64;i++) {
            p.incoming_f=p.f;p.f+=(i%2?-1.:1.)*4e-8;provider_value=p.f;
            require(!p.step(m,w,i),"Net-flat accumulated-gain window signaled");
        }
        require(exit_calls==old_exits,"Cumulative positive gains used net change");
        p=Input();kaks_fiber_near_begin(&m,w);provider_value=10.+1e-9;
        for(int i=1;i<=64;i++)require(!p.step(m,w,i),"Re-score-worse candidate signaled");
        require(w.count==0&&!t.pinned_valid[2],"No feasible matching-score best should clear");
        p=Input();provider_value=10.;p.step(m,w,1);p.x[0]=p.box[0][0]-1e-10;
        require(!p.step(m,w,2)&&w.count==0,"Unbounded current point did not clear");
        p=Input();p.step(m,w,1);p.g[0]=HUGE_VAL;
        require(!p.step(m,w,2)&&w.count==0,"Nonfinite gradient did not clear");
        p=Input();p.step(m,w,1);p.incoming_f+=1e-9;
        require(!p.step(m,w,2)&&w.count==0,"F continuity break did not clear");
        t.oracle=true;p=Input();const int old_calls=provider_calls;
        for(int i=1;i<=70;i++)require(!p.step(m,w,i),"Oracle early signal");
        require(provider_calls==old_calls,"Oracle provider altered by near guard");
        std::printf("{\"passed\":true,\"assertions\":%d,\"source_guard_executed\":true,\"CUDA_used\":false}\n",assertions);
        return 0;
    }catch(const std::exception &e){std::fprintf(stderr,"Guard test failed: %s\n",e.what());return 1;}
}
