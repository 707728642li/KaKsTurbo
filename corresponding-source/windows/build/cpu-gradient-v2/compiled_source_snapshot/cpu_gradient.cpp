/* Native Windows FP64 fused F/log-gradient for the pure StrictPositive model.
 * The immutable shared_core translation unit builds the actual forward A.
 * Five-GEMM T18 coefficients/theta match reviewed cpu_backend.cpp/PyTorch2.5.1.
 * Manual reverse products differentiate the entire selected polynomial and
 * all squarings. Integer scaling decisions are held fixed within each point.
 * Complete original-order Q/mr Jacobians retain the mean-rate quotient rule.
 * No optimizer, GPU, approximate cache, clipping or legacy rate mutation. */
#include "cpu_gradient.h"
#include "shared_core.hpp"
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace kakscuda { namespace gradient {
constexpr size_t n=61, nn=n*n, diag_width=KAKS_CPU_GRADIENT_DIAGNOSTIC_WIDTH;
constexpr double theta=1.090863719290036;
constexpr double coefficients[5][5]={
 {0.,-1.00365581030144618291e-1,-8.02924648241156932449e-3,-8.92138498045729985177e-4,0.},
 {0.,3.97849749499645077844e-1,1.36783778460411720168,4.98289622525382669416e-1,-6.37898194594723280150e-4},
 {-1.09676396052962061844e1,1.68015813878906206114,5.71779846478865511061e-2,-6.98210122488052056106e-3,3.34975017086070470649e-5},
 {-9.04316832390810593223e-2,-6.76404519071381882256e-2,6.75961301770459654925e-2,2.95552570429315521194e-2,-1.39180257516060693404e-5},
 {0.,0.,-9.23364619367118555360e-2,-1.69364939002081722752e-2,-1.40086798182036094347e-5}};
constexpr int rate_map[7][6]={{5,5,5,5,5,5},{0,0,5,5,5,5},
 {0,1,5,5,5,5},{0,0,1,1,5,5},{0,1,2,2,5,5},
 {0,0,1,2,3,5},{0,1,2,3,4,5}};
constexpr char code1[]="FFLLSSSSYY!!CC!WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG";
struct Edge {int i,j,rate;bool synonymous;};
struct EdgeTable {std::array<Edge,263> edges{};int count=0;};
constexpr EdgeTable make_edges() {
 std::array<int,61> sense{};int count=0;
 for(int i=0;i<64;i++)if(code1[i]!='!')sense[count++]=i;
 EdgeTable result{};
 for(int i=0;i<61;i++)for(int j=0;j<i;j++) {
  const int a=sense[i],b=sense[j],from[3]={a/16,(a/4)%4,a%4},to[3]={b/16,(b/4)%4,b%4};
  int changes=0,pos=0;for(int k=0;k<3;k++)if(from[k]!=to[k]){changes++;pos=k;}
  if(changes!=1)continue;
  const int lo=from[pos]<to[pos]?from[pos]:to[pos],hi=from[pos]>to[pos]?from[pos]:to[pos];
  int rate=5;if(lo==0&&hi==1)rate=0;else if(lo==2&&hi==3)rate=1;
  else if(lo==0&&hi==2)rate=2;else if(lo==1&&hi==3)rate=3;else if(lo==0&&hi==3)rate=4;
  if(result.count<263)result.edges[result.count]={i,j,rate,code1[a]==code1[b]};
  result.count++;
 }
 return result;
}
constexpr auto edges=make_edges();
static_assert(edges.count==263,"Independent Code1 derivative edge table must have263edges");
uint64_t workspace_width(uint32_t cap) {return KAKS_CORE_SNAPSHOT_WIDTH+(24ULL+cap)*nn;}
/* Explicit CBLAS_64 dimensions remain int64 regardless of default MKL layer. */
using Dgemm=void (__cdecl *)(int,int,int,int64_t,int64_t,int64_t,double,const double *,int64_t,const double *,int64_t,double,double *,int64_t);
using LocalThreads=int (__cdecl *)(int);
using GetThreads=int (__cdecl *)(void);
using VersionString=void (__cdecl *)(char *,int);
template<class T>T symbol(HMODULE module,const char *name) {
 const FARPROC address=GetProcAddress(module,name);T function=nullptr;
 static_assert(sizeof(function)==sizeof(address),"Windows dynamic function pointer ABI");
 std::memcpy(&function,&address,sizeof(function));return function;
}
struct MklLibrary final {
 HMODULE handle=nullptr;Dgemm gemm=nullptr;LocalThreads local=nullptr;GetThreads get_threads=nullptr;
 std::wstring loaded_path;std::string version;
 explicit MklLibrary(const wchar_t *path) {
  if(!path||!*path||!(path[0]==L'\\'||(path[0]&&path[1]==L':'&&path[2]==L'\\')))
   throw std::invalid_argument("MKL path must be an absolute native Windows path");
  handle=LoadLibraryExW(path,nullptr,LOAD_WITH_ALTERED_SEARCH_PATH);
  if(!handle)throw std::runtime_error("Cannot load supplied MKL DLL (Windows error "+std::to_string(GetLastError())+")");
  try {
   gemm=symbol<Dgemm>(handle,"cblas_dgemm_64");local=symbol<LocalThreads>(handle,"MKL_Set_Num_Threads_Local");
   get_threads=symbol<GetThreads>(handle,"MKL_Get_Max_Threads");auto get_version=symbol<VersionString>(handle,"MKL_Get_Version_String");
   if(!gemm||!local||!get_threads||!get_version)throw std::runtime_error("MKL runtime lacks explicit64CBLAS/thread/version exports");
   wchar_t actual[32768]{};const DWORD len=GetModuleFileNameW(handle,actual,32768);
   if(!len||len>=32768)throw std::runtime_error("Cannot resolve actually loaded MKL path");
   loaded_path.assign(actual,len);char text[1024]{};get_version(text,1024);version=text;
  }catch(...){FreeLibrary(handle);handle=nullptr;throw;}
 }
 ~MklLibrary(){if(handle)FreeLibrary(handle);}
 MklLibrary(const MklLibrary&)=delete;MklLibrary&operator=(const MklLibrary&)=delete;
 void product(const double *a,const double *b,double *out,bool transpose_a=false,bool transpose_b=false,double beta=0.)const {
  gemm(101,transpose_a?112:111,transpose_b?112:111,n,n,n,1.,a,n,b,n,beta,out,n);
 }
};
struct OneMklThread final {
 const MklLibrary &mkl;int previous,observed;
 explicit OneMklThread(const MklLibrary &lib):mkl(lib),previous(lib.local(1)),observed(lib.get_threads()) {
  if(observed!=1){lib.local(previous);throw std::runtime_error("MKL did not honor one thread per worker");}
 }
 ~OneMklThread(){mkl.local(previous);}
};
/* Exactly one caller-owned/worker-owned arena, allocated once. Every GEMM has
 * distinct inputs/output; adjoints use beta=1 only for explicit accumulation. */
struct Arena {
 double *snapshot,*z,*a2,*a3,*a6,*b[5],*c,*d,*tape;
 double *g,*previous,*gb0,*gb2,*gb3,*gb4,*gz,*ga2,*ga3,*ga6,*dq,*ds;
 Arena(double *storage,uint32_t cap):snapshot(storage) {
  double *cursor=storage+KAKS_CORE_SNAPSHOT_WIDTH;
  auto take=[&cursor](){double *out=cursor;cursor+=nn;return out;};
  z=take();a2=take();a3=take();a6=take();for(auto &item:b)item=take();c=take();d=take();
  tape=cursor;cursor+=(cap+1ULL)*nn;
  g=take();previous=take();gb0=take();gb2=take();gb3=take();gb4=take();
  gz=take();ga2=take();ga3=take();ga6=take();dq=take();ds=take();
 }
};
void init_diagnostics(double *diag) {
 if(!diag)throw std::invalid_argument("Missing gradient diagnostics");
 std::fill_n(diag,diag_width,std::numeric_limits<double>::quiet_NaN());
}
void validate_state(const PairData &pair,const ModelState &state) {
 if(state.config.support_policy!=SupportPolicy::StrictPositive||state.config.support_cutoff!=0.)
  throw std::invalid_argument("Gradient backend accepts StrictPositive only");
 for(const Pattern &pattern:pair.patterns())if(!(state.pi[pattern.z0]>0.)||!(state.pi[pattern.z1]>0.))
  throw std::invalid_argument("Observed endpoint absent from stationary support");
}
int forward(const MklLibrary &mkl,uint32_t primary,uint32_t cap,bool fallback,Arena &w,double *diag) {
 const double *matrix=w.snapshot+KAKS_CORE_A_OFFSET;double norm=0.;
 for(size_t j=0;j<n;j++) {
  double sum=0.;for(size_t i=0;i<n;i++) {
   const double value=matrix[i*n+j];if(!std::isfinite(value))throw std::invalid_argument("Nonfinite frozen exponential matrix");
   sum+=std::abs(value);
  }
  if(!std::isfinite(sum))throw std::invalid_argument("Nonfinite matrix1-norm");
  norm=std::max(norm,sum);
 }
 int squarings=0;
 if(norm>theta) {
  const double required=std::ceil(std::log2(norm)-std::log2(theta));
  if(!std::isfinite(required)||required>1023.)throw std::runtime_error("Required scaling exceeds representable FP64 range");
  squarings=static_cast<int>(required);if(std::ldexp(norm,-squarings)>theta)squarings++;
 }
 const bool adaptive=static_cast<uint32_t>(squarings)>primary;
 diag[0]=norm;diag[1]=squarings;diag[2]=adaptive?1.:0.;diag[3]=diag[8]=diag[9]=0.;
 if(adaptive&&!fallback)throw std::runtime_error("Actual scaling exceeds primary budget; adaptive fallback disabled");
 if(static_cast<uint32_t>(squarings)>cap)throw std::runtime_error("Actual scaling exceeds allocated gradient tape budget; use validated F-only/FD provider");
 for(size_t k=0;k<nn;k++)w.z[k]=std::ldexp(matrix[k],-squarings);
 mkl.product(w.z,w.z,w.a2);mkl.product(w.z,w.a2,w.a3);mkl.product(w.a3,w.a3,w.a6);
 for(int row=0;row<5;row++)for(size_t k=0;k<nn;k++) {
  const double powers[5]={static_cast<double>(k/n==k%n),w.z[k],w.a2[k],w.a3[k],w.a6[k]};
  double value=0.;for(int p=0;p<5;p++)value+=powers[p]*coefficients[row][p];w.b[row][k]=value;
 }
 mkl.product(w.b[0],w.b[4],w.c);
 for(size_t k=0;k<nn;k++){w.c[k]=w.b[3][k]+w.c[k];w.d[k]=w.b[2][k]+w.c[k];}
 mkl.product(w.d,w.c,w.tape);
 for(size_t k=0;k<nn;k++)w.tape[k]=w.b[1][k]+w.tape[k];
 for(int s=0;s<squarings;s++)mkl.product(w.tape+s*nn,w.tape+s*nn,w.tape+(s+1)*nn);
 const double *result=w.tape+squarings*nn;
 for(size_t k=0;k<nn;k++)if(!std::isfinite(result[k]))throw std::runtime_error("Nonfinite T18 exponential result");
 diag[3]=diag[8]=5.+squarings;return squarings;
}
/* Reverse square: Gprev=G*T^T+T^T*G, with no symmetry assumption on G. */
void reverse(const MklLibrary &mkl,Arena &w,int squarings,double *diag) {
 double *g=w.g,*previous=w.previous;
 for(int s=squarings;s>0;s--) {
  const double *t=w.tape+(s-1)*nn;
  mkl.product(g,t,previous,false,true);mkl.product(t,g,previous,true,false,1.);std::swap(g,previous);
 }
 /* R=B1+D*C, D=B2+C, C=B3+B0*B4. gb1 aliases the unchanged g. */
 mkl.product(g,w.c,w.gb2,false,true);mkl.product(w.d,g,w.gb3,true,false);
 for(size_t k=0;k<nn;k++)w.gb3[k]+=w.gb2[k];
 mkl.product(w.gb3,w.b[4],w.gb0,false,true);mkl.product(w.b[0],w.gb3,w.gb4,true,false);
 const double *gb[5]={w.gb0,g,w.gb2,w.gb3,w.gb4};
 double *gp[4]={w.gz,w.ga2,w.ga3,w.ga6};
 for(int p=0;p<4;p++)for(size_t k=0;k<nn;k++) {
  double value=0.;for(int row=0;row<5;row++)value+=gb[row][k]*coefficients[row][p+1];gp[p][k]=value;
 }
 /* A6=A3*A3, A3=Z*A2, A2=Z*Z, Z=ldexp(A,-s). */
 mkl.product(w.ga6,w.a3,w.ga3,false,true,1.);mkl.product(w.a3,w.ga6,w.ga3,true,false,1.);
 mkl.product(w.ga3,w.a2,w.gz,false,true,1.);mkl.product(w.z,w.ga3,w.ga2,true,false,1.);
 mkl.product(w.ga2,w.z,w.gz,false,true,1.);mkl.product(w.z,w.ga2,w.gz,true,false,1.);
 for(size_t k=0;k<nn;k++) {
  w.gz[k]=std::ldexp(w.gz[k],-squarings);
  if(!std::isfinite(w.gz[k]))throw std::runtime_error("Nonfinite matrix-exponential adjoint");
 }
 diag[9]=10.+2.*squarings;diag[3]=diag[8]+diag[9];
}
/* Matrix derivative is conditional on fixed prepared pi. The full unmasked Q
 * sets mr; lower active entries alone build symmetric S, exactly as forward.
 * The quotient term -A*(dmr/mr) includes synonymous and nonsynonymous rates. */
void contract(const ModelState &state,Arena &w,double *raw,double *log) {
 const int nk=state.config.nkappa,np=nk+2;const double *s=w.snapshot+KAKS_CORE_S_OFFSET;
 const double *a=w.snapshot+KAKS_CORE_A_OFFSET,*sqrtpi=w.snapshot+KAKS_CORE_SQRT_PI_OFFSET;
 const double mr=w.snapshot[KAKS_CORE_SCALARS_OFFSET+3],time=state.x[0];
 if(!(mr>0.)||!std::isfinite(mr))throw std::runtime_error("Nonpositive or nonfinite mean rate; derivative unavailable");
 std::fill_n(raw,7,0.);std::fill_n(log,7,0.);
 for(size_t k=0;k<nn;k++)raw[0]+=w.gz[k]*(s[k]/mr);
 for(int parameter=1;parameter<np;parameter++) {
  std::fill_n(w.dq,nn,0.);std::fill_n(w.ds,nn,0.);double drs=0.,dra=0.;
  const bool is_omega=parameter==nk+1;
  for(const Edge &edge:edges.edges) {
   const int i=edge.i,j=edge.j,r=edge.rate;const double base=state.rates[r]*state.pi[j];
   if(is_omega) {
    if(edge.synonymous)continue;
    w.dq[i*n+j]=base;w.dq[j*n+i]=state.rates[r]*state.pi[i];dra+=2.*state.pi[i]*base;
   }else if(rate_map[state.config.id/2][r]==parameter-1) {
    w.dq[i*n+j]=state.pi[j];w.dq[j*n+i]=state.pi[i];const double dpij=2.*state.pi[i]*state.pi[j];
    if(edge.synonymous)drs+=dpij;
    else {w.dq[i*n+j]*=state.omega;w.dq[j*n+i]*=state.omega;dra+=dpij*state.omega;}
   }
  }
  for(size_t i=0;i<n;i++) {double sum=0.;for(size_t j=0;j<n;j++)sum+=w.dq[i*n+j];w.dq[i*n+i]=-sum;}
  for(size_t i=0;i<n;i++)if(state.pi[i]>0.) {
   w.ds[i*n+i]=w.dq[i*n+i];
   for(size_t j=0;j<i;j++)if(state.pi[j]>0.)w.ds[i*n+j]=w.ds[j*n+i]=w.dq[i*n+j]*sqrtpi[i]/sqrtpi[j];
  }
  const double dm_over_m=(drs+dra)/mr;double derivative=0.;
  for(size_t k=0;k<nn;k++)derivative+=w.gz[k]*((w.ds[k]*time)/mr-a[k]*dm_over_m);
  raw[parameter]=derivative;
 }
 for(int j=0;j<np;j++) {
  log[j]=state.x[j]*raw[j];
  if(!std::isfinite(raw[j])||!std::isfinite(log[j]))throw std::runtime_error("Nonfinite raw-model gradient");
 }
}
struct Batch final {
 const void *const *pairs=nullptr;const int64_t *slots=nullptr;const double *parameters=nullptr;const uint32_t *kinds=nullptr;
 uint64_t count=0;double *objectives=nullptr,*gradient_log=nullptr,*diagnostics=nullptr;uint32_t *codes=nullptr;
};
class Session;
void evaluate(Session &,const PairData &,const ModelState &,uint32_t,double *,double &,double *,double *,double *,double *);
class Session final {
public:
 MklLibrary mkl;const uint32_t workers,primary_limit,tape_cap;const bool fallback;std::mutex call_mutex;
 Session(const wchar_t *path,uint32_t count,uint32_t primary,uint32_t cap,bool adaptive):
  mkl(path),workers(count),primary_limit(primary),tape_cap(cap),fallback(adaptive) {
  if(!workers||workers>32||primary>16||cap<primary||cap>64)throw std::invalid_argument("Use workers1..32, primary0..16, tape primary..64");
  arenas.resize(workers);for(auto &arena:arenas)arena.resize(workspace_width(cap));threads.reserve(workers);
  try {for(uint32_t i=0;i<workers;i++)threads.emplace_back([this,i]{worker(i);});}catch(...){stop();throw;}
 }
 ~Session(){stop();}
 int batch(const Batch &input) {
  if(!input.count)return 0;
  {std::lock_guard<std::mutex> guard(work_mutex);current=input;next.store(0);finished=0;epoch++;}ready.notify_all();
  std::unique_lock<std::mutex> lock(work_mutex);completed.wait(lock,[this]{return finished==workers;});current=Batch{};lock.unlock();
  return std::any_of(input.codes,input.codes+input.count,[](uint32_t code){return code!=0;})?2:0;
 }
private:
 std::vector<std::vector<double>> arenas;std::vector<std::thread> threads;
 std::mutex work_mutex;std::condition_variable ready,completed;std::atomic<uint64_t> next{0};
 Batch current{};uint64_t epoch=0;uint32_t finished=0;bool stopping=false;
 void stop()noexcept {
  {std::lock_guard<std::mutex> guard(work_mutex);stopping=true;}ready.notify_all();for(auto &thread:threads)if(thread.joinable())thread.join();threads.clear();
 }
 void worker(uint32_t index)noexcept {
  uint64_t seen=0;
  for(;;) {
   Batch input;
   {std::unique_lock<std::mutex> lock(work_mutex);ready.wait(lock,[this,seen]{return stopping||epoch!=seen;});if(stopping)return;seen=epoch;input=current;}
   for(;;) {
    const uint64_t row=next.fetch_add(1);if(row>=input.count)break;double *diag=input.diagnostics+row*diag_width;
    std::fill_n(diag,diag_width,std::numeric_limits<double>::quiet_NaN());
    try {
     const int64_t slot=input.slots[row];const auto &pair=*static_cast<const PairData *>(input.pairs[slot/14]);
     const auto state=model_state(pair,slot%14,SupportPolicy::StrictPositive,input.parameters+row*7);
     evaluate(*this,pair,state,input.kinds[row],arenas[index].data(),input.objectives[row],input.gradient_log+row*7,nullptr,diag,nullptr);input.codes[row]=0;
    }catch(...) {
     input.objectives[row]=std::numeric_limits<double>::quiet_NaN();std::fill_n(input.gradient_log+row*7,7,std::numeric_limits<double>::quiet_NaN());input.codes[row]=1;
    }
   }
   {std::lock_guard<std::mutex> lock(work_mutex);finished++;if(finished==workers)completed.notify_one();}
  }
 }
};
void evaluate(Session &session,const PairData &pair,const ModelState &state,uint32_t kind,
 double *workspace,double &objective,double *gradient_log,double *gradient_raw,double *diag,double *transition) {
 init_diagnostics(diag);validate_state(pair,state);OneMklThread single(session.mkl);Arena w(workspace,session.tape_cap);
 build_q_snapshot(state,w.snapshot);diag[10]=state.config.nkappa+2;diag[11]=kind;diag[7]=single.observed;
 const int squarings=forward(session.mkl,session.primary_limit,session.tape_cap,session.fallback,w,diag);
 const double *exp=w.tape+squarings*nn,*sqrtpi=w.snapshot+KAKS_CORE_SQRT_PI_OFFSET;
 double minimum_entry=std::numeric_limits<double>::infinity(),row_error=0.;
 for(size_t i=0;i<n;i++) {
  double sum=0.;for(size_t j=0;j<n;j++) {
   const double value=state.pi[i]>0.&&state.pi[j]>0.?(exp[i*n+j]/sqrtpi[i])*sqrtpi[j]:static_cast<double>(i==j);
   if(transition)transition[i*n+j]=value;
   sum+=value;minimum_entry=std::min(minimum_entry,value);
  }
  if(state.pi[i]>0.)row_error=std::max(row_error,std::abs(sum-1.));
 }
 double lnL=0.,minimum=std::numeric_limits<double>::infinity();std::fill_n(w.g,nn,0.);
 for(const Pattern &pattern:pair.patterns()) {
  const size_t i=pattern.z0,j=pattern.z1,k=i*n+j;
  const double joint=((exp[k]/sqrtpi[i])*sqrtpi[j])*state.pi[i];minimum=std::min(minimum,joint);
  if(!(joint>0.)||!std::isfinite(joint))throw std::runtime_error("Nonpositive/nonfinite observed probability; clipping forbidden");
  lnL+=std::log(joint)*pattern.weight;
  if(kind==1)w.g[k]+=(((-pattern.weight/joint)*state.pi[i])*sqrtpi[j])/sqrtpi[i];
 }
 if(!std::isfinite(lnL))throw std::runtime_error("Nonfinite objective");
 objective=-lnL;
 diag[4]=minimum;diag[5]=row_error;diag[6]=minimum_entry;
 double raw[7]{};std::fill_n(gradient_log,7,0.);
 if(kind==1){reverse(session.mkl,w,squarings,diag);contract(state,w,raw,gradient_log);}
 if(gradient_raw)std::copy_n(raw,7,gradient_raw);
}
}}
namespace {
void clear_error(char *error,uint32_t cap){if(error&&cap)error[0]=0;}
int report_error(const char *message,char *error,uint32_t cap,int code=1) {
 if(error&&cap){size_t len=std::strlen(message);if(len>=cap)len=cap-1;std::memcpy(error,message,len);error[len]=0;}return code;
}
kakscuda::gradient::Session &session_of(void *raw){if(!raw)throw std::invalid_argument("Missing gradient session");return *static_cast<kakscuda::gradient::Session *>(raw);}
const kakscuda::PairData &pair_of(const void *raw){if(!raw)throw std::invalid_argument("Missing live prepared pair");return *static_cast<const kakscuda::PairData *>(raw);}
#define KAKS_GRADIENT_TRY clear_error(error,cap);try
#define KAKS_GRADIENT_CATCH catch(const std::exception &e){return report_error(e.what(),error,cap);}catch(...){return report_error("Unknown CPU gradient error",error,cap);}
}
extern "C" uint32_t kaks_cpu_gradient_abi_version(void){return KAKS_CPU_GRADIENT_ABI_VERSION;}
extern "C" uint64_t kaks_cpu_gradient_workspace_doubles(uint32_t cap){return cap<=64?kakscuda::gradient::workspace_width(cap):0;}
extern "C" int kaks_cpu_gradient_create(const wchar_t *path,uint32_t workers,uint32_t primary,uint32_t tape,uint32_t fallback,void **output,char *error,uint32_t cap) {
 if(output)*output=nullptr;
 KAKS_GRADIENT_TRY {
  if(!output||fallback>1)throw std::invalid_argument("Missing session output or invalid fallback policy");
  auto owned=std::make_unique<kakscuda::gradient::Session>(path,workers,primary,tape,fallback!=0);*output=owned.release();return 0;
 } KAKS_GRADIENT_CATCH
}
extern "C" void kaks_cpu_gradient_free(void *raw){delete static_cast<kakscuda::gradient::Session *>(raw);}
extern "C" int kaks_cpu_gradient_runtime_info(const void *raw,wchar_t *path,uint32_t path_cap,char *version,uint32_t version_cap,uint32_t *workers,uint32_t *tape,char *error,uint32_t cap) {
 KAKS_GRADIENT_TRY {
  if(!raw||!path||!version||!workers||!tape)throw std::invalid_argument("Missing gradient runtime output");
  const auto &s=*static_cast<const kakscuda::gradient::Session *>(raw);
  if(path_cap<=s.mkl.loaded_path.size()||version_cap<=s.mkl.version.size())throw std::invalid_argument("Undersized runtime output");
  std::copy(s.mkl.loaded_path.begin(),s.mkl.loaded_path.end(),path);path[s.mkl.loaded_path.size()]=0;
  std::copy(s.mkl.version.begin(),s.mkl.version.end(),version);version[s.mkl.version.size()]=0;*workers=s.workers;*tape=s.tape_cap;return 0;
 } KAKS_GRADIENT_CATCH
}
extern "C" int kaks_cpu_gradient_evaluate_batch(void *raw,const void *const *pairs,uint32_t pair_count,const int64_t *slots,const double *parameters,
 const uint32_t *kinds,uint64_t count,double *objectives,double *gradient_log,double *diagnostics,uint32_t *codes,uint64_t capacity,char *error,uint32_t cap) {
 KAKS_GRADIENT_TRY {
  auto &s=session_of(raw);std::unique_lock<std::mutex> call(s.call_mutex,std::try_to_lock);
  if(!call.owns_lock())throw std::runtime_error("Concurrent calls on one gradient session unsupported");
  if(capacity<count||count>static_cast<uint64_t>(std::numeric_limits<int64_t>::max()/12))throw std::invalid_argument("Invalid batch capacity/count");
  if(!count)return 0;
  if(!pairs||!pair_count||!slots||!parameters||!kinds||!objectives||!gradient_log||!diagnostics||!codes)throw std::invalid_argument("Missing mixed-gradient batch buffer");
  for(uint64_t row=0;row<count;row++) {
   const int64_t slot=slots[row];if(slot<0||static_cast<uint64_t>(slot)>=static_cast<uint64_t>(pair_count)*14||kinds[row]>1)throw std::invalid_argument("Invalid batch slot/kind");
   auto state=kakscuda::model_state(pair_of(pairs[slot/14]),slot%14,kakscuda::SupportPolicy::StrictPositive,parameters+row*7);
   kakscuda::gradient::validate_state(pair_of(pairs[slot/14]),state);
  }
  const int code=s.batch({pairs,slots,parameters,kinds,count,objectives,gradient_log,diagnostics,codes});
  if(code)return report_error("One or more CPU F/gradient rows failed numerically; inspect retained row codes/diagnostics",error,cap,code);
  return 0;
 } KAKS_GRADIENT_CATCH
}
extern "C" int kaks_cpu_gradient_evaluate_one(void *raw,const void *pair,uint32_t model,const double *x,uint32_t kind,double *workspace,uint64_t size,
 double *objective,double *gradient_log,double *gradient_raw,double *diag,double *transition,char *error,uint32_t cap) {
 KAKS_GRADIENT_TRY {
  auto &s=session_of(raw);std::unique_lock<std::mutex> call(s.call_mutex,std::try_to_lock);
  if(!call.owns_lock())throw std::runtime_error("Concurrent calls on one gradient session unsupported");
  if(kind>1||!workspace||size<kakscuda::gradient::workspace_width(s.tape_cap)||!objective||!gradient_log||!diag)throw std::invalid_argument("Invalid single-gradient kind/workspace/output");
  const auto &p=pair_of(pair);const auto state=kakscuda::model_state(p,model,kakscuda::SupportPolicy::StrictPositive,x);
  kakscuda::gradient::evaluate(s,p,state,kind,workspace,*objective,gradient_log,gradient_raw,diag,transition);return 0;
 } KAKS_GRADIENT_CATCH
}
