/* Independent native Windows FP64 objective backend for the publication model.
 * T18 coefficients/theta: PyTorch v2.5.1 ATen/native/LinearAlgebra.cpp,
 * https://raw.githubusercontent.com/pytorch/pytorch/v2.5.1/aten/src/ATen/native/LinearAlgebra.cpp
 * No optimizer, GPU runtime, legacy rate mutation, or approximate cache. */
#include "cpu_backend.hpp"
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

namespace kakscuda {
namespace {
constexpr size_t n=61, nn=n*n, workspace_width=KAKS_CORE_SNAPSHOT_WIDTH+12*nn;
constexpr double theta=1.090863719290036;
constexpr double coefficients[5][5]={
 {0.,-1.00365581030144618291e-1,-8.02924648241156932449e-3,-8.92138498045729985177e-4,0.},
 {0.,3.97849749499645077844e-1,1.36783778460411720168,4.98289622525382669416e-1,-6.37898194594723280150e-4},
 {-1.09676396052962061844e1,1.68015813878906206114,5.71779846478865511061e-2,-6.98210122488052056106e-3,3.34975017086070470649e-5},
 {-9.04316832390810593223e-2,-6.76404519071381882256e-2,6.75961301770459654925e-2,2.95552570429315521194e-2,-1.39180257516060693404e-5},
 {0.,0.,-9.23364619367118555360e-2,-1.69364939002081722752e-2,-1.40086798182036094347e-5}};
/* Explicit _64 CBLAS dimensions are int64_t regardless of the runtime's
 * configurable default LP64/ILP64 layer; enums remain 32-bit CBLAS enums.
 * No MKL_Set_Interface_Layer call mutates a process-wide interface. */
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
 HMODULE handle=nullptr; Dgemm gemm=nullptr; LocalThreads local=nullptr;
 GetThreads get_threads=nullptr; std::wstring loaded_path; std::string version;
 explicit MklLibrary(const wchar_t *path) {
  if(!kaksforge_absolute_blas_path(path))
   throw std::invalid_argument("BLAS path must be an absolute native Linux path");
  handle=LoadLibraryExW(path,nullptr,LOAD_WITH_ALTERED_SEARCH_PATH);
  if(!handle)throw std::runtime_error("Cannot load supplied MKL DLL (Windows error "+std::to_string(GetLastError())+")");
  try {
   gemm=symbol<Dgemm>(handle,"cblas_dgemm_64");
   local=symbol<LocalThreads>(handle,"MKL_Set_Num_Threads_Local");
   get_threads=symbol<GetThreads>(handle,"MKL_Get_Max_Threads");
   auto get_version=symbol<VersionString>(handle,"MKL_Get_Version_String");
   if(!gemm||!local||!get_threads||!get_version)throw std::runtime_error("MKL runtime lacks the explicit 64-bit CBLAS/thread/version exports");
   wchar_t actual[32768]{};const DWORD len=GetModuleFileNameW(handle,actual,32768);
   if(!len||len>=32768)throw std::runtime_error("Cannot resolve the actually loaded MKL module path");
   loaded_path.assign(actual,len);char text[1024]{};get_version(text,1024);version=text;
  }catch(...){FreeLibrary(handle);handle=nullptr;throw;}
 }
 ~MklLibrary(){if(handle)FreeLibrary(handle);}
 MklLibrary(const MklLibrary&)=delete; MklLibrary&operator=(const MklLibrary&)=delete;
 void product(const double *a,const double *b,double *out)const {
  gemm(101,111,111,n,n,n,1.,a,n,b,n,0.,out,n);
 }
};
struct OneMklThread final {
 const MklLibrary &mkl;int previous,observed;
 explicit OneMklThread(const MklLibrary &lib):mkl(lib),previous(lib.local(1)),observed(lib.get_threads()) {
  if(observed!=1){lib.local(previous);throw std::runtime_error("MKL did not honor thread-local one-thread execution");}
 }
 ~OneMklThread(){mkl.local(previous);}
};
void init_diagnostics(double *diag) {if(!diag)throw std::invalid_argument("Missing objective diagnostics");std::fill_n(diag,8,std::numeric_limits<double>::quiet_NaN());}
/* Five GEMMs evaluate the complete degree-18 polynomial. Required scaling is
 * computed from the actual matrix 1-norm. Fallback adds the actual extra
 * squarings; no coefficient/time/support is changed and no cap is accepted. */
void exponential(const MklLibrary &mkl,uint32_t primary_limit,bool fallback,
 const double *matrix,double *out,double *workspace,double *diag) {
 init_diagnostics(diag);double norm=0.;
 for(size_t j=0;j<n;j++) {double sum=0.;for(size_t i=0;i<n;i++) {
   const double entry=matrix[i*n+j];if(!std::isfinite(entry))throw std::invalid_argument("Nonfinite exponential input");
   sum+=std::abs(entry);
  }if(!std::isfinite(sum))throw std::invalid_argument("Nonfinite exponential matrix 1-norm");norm=std::max(norm,sum);
 }
 int squarings=0;
 if(norm>theta) {
  const double required=std::ceil(std::log2(norm)-std::log2(theta));
  if(!std::isfinite(required)||required>1023.)throw std::runtime_error("Required exponential scaling exceeds the representable FP64 range");
  squarings=static_cast<int>(required);
  /* Correct a possible one-ULP log2 boundary underestimate explicitly. */
  if(std::ldexp(norm,-squarings)>theta)squarings++;
  if(squarings>1023)throw std::runtime_error("Required exponential scaling exceeds the representable FP64 range");
 }
 const bool used_fallback=static_cast<uint32_t>(squarings)>primary_limit;
 diag[0]=norm;diag[1]=squarings;diag[2]=used_fallback?1.:0.;diag[3]=0.;
 if(used_fallback&&!fallback)throw std::runtime_error("Actual matrix norm exceeds the configured primary T18 scaling budget; fallback disabled");
 double *scaled=workspace+KAKS_CORE_SNAPSHOT_WIDTH;
 double *a2=scaled+nn,*a3=a2+nn,*a6=a3+nn,*b[5];b[0]=a6+nn;
 for(int i=1;i<5;i++)b[i]=b[i-1]+nn;
 double *b3=b[4]+nn,*result=b3+nn,*temp=result+nn;
 for(size_t k=0;k<nn;k++)scaled[k]=std::ldexp(matrix[k],-squarings);
 mkl.product(scaled,scaled,a2);mkl.product(scaled,a2,a3);mkl.product(a3,a3,a6);
 for(int row=0;row<5;row++)for(size_t k=0;k<nn;k++) {
  const double powers[5]={static_cast<double>(k/n==k%n),scaled[k],a2[k],a3[k],a6[k]};
  double value=0.;for(int p=0;p<5;p++)value+=powers[p]*coefficients[row][p];b[row][k]=value;
 }
 mkl.product(b[0],b[4],b3);
 for(size_t k=0;k<nn;k++){b3[k]=b[3][k]+b3[k];temp[k]=b[2][k]+b3[k];}
 mkl.product(temp,b3,result);
 for(size_t k=0;k<nn;k++)result[k]=b[1][k]+result[k];
 for(int i=0;i<squarings;i++){mkl.product(result,result,temp);std::swap(result,temp);}
 for(size_t k=0;k<nn;k++)if(!std::isfinite(result[k]))throw std::runtime_error("Nonfinite T18 exponential result");
 std::copy_n(result,nn,out);diag[3]=5.+squarings;
}
struct Batch final {
 const void *const *pairs=nullptr;const int64_t *slots=nullptr;const double *parameters=nullptr;
 uint64_t count=0;double *objectives=nullptr,*diagnostics=nullptr;uint32_t *codes=nullptr;
};
void validate_state(const PairData &pair,const ModelState &state) {
 if(state.config.support_policy!=SupportPolicy::StrictPositive||state.config.support_cutoff!=0.)
  throw std::invalid_argument("CPU objective accepts StrictPositive coherent states only");
 double total=0.;for(int i=0;i<61;i++){
  if(!std::isfinite(state.pi[i])||state.pi[i]<0.||state.pi[i]>1.)throw std::invalid_argument("Invalid supplied state frequencies");
  total+=state.pi[i];
 }
 if(std::abs(total-1.)>1e-12||state.pi[61]!=0.||state.pi[62]!=0.||state.pi[63]!=0.)
  throw std::invalid_argument("Supplied state pi64 must be normalized packed sense61 with zero padding");
 for(const auto &pattern:pair.patterns())if(!(state.pi[pattern.z0]>0.)||!(state.pi[pattern.z1]>0.))
  throw std::invalid_argument("Supplied state must retain positive frequencies at every observed endpoint");
 for(double rate:state.rates)if(!std::isfinite(rate)||!(rate>0.))throw std::invalid_argument("Expanded rates must be positive finite doubles");
 if(!std::isfinite(state.omega)||!(state.omega>0.)||!std::isfinite(state.x[0])||state.x[0]<0.)
  throw std::invalid_argument("Invalid supplied omega or codon branch length");
}
}

class CpuSession final {
public:
 MklLibrary mkl;const uint32_t workers,primary_limit;const bool fallback;
 std::mutex call_mutex;
 CpuSession(const wchar_t *path,uint32_t count,uint32_t limit,bool adaptive):mkl(path),workers(count),primary_limit(limit),fallback(adaptive) {
  if(!workers||workers>128||limit>16)throw std::invalid_argument("CPU workers must be 1..128 and primary scaling budget 0..16");
  arenas.resize(workers);for(auto &arena:arenas)arena.resize(workspace_width);
  threads.reserve(workers);
  try {for(uint32_t i=0;i<workers;i++)threads.emplace_back([this,i]{worker(i);});}
  catch(...){stop();throw;}
 }
 ~CpuSession(){stop();}
 int batch(const Batch &input) {
  if(!input.count)return 0;
  {std::lock_guard<std::mutex> guard(work_mutex);current=input;next.store(0);finished=0;epoch++;}
  ready.notify_all();
  std::unique_lock<std::mutex> lock(work_mutex);completed.wait(lock,[this]{return finished==workers;});
  current=Batch{};lock.unlock();
  return std::any_of(input.codes,input.codes+input.count,[](uint32_t code){return code!=0;})?2:0;
 }
private:
 std::vector<std::vector<double>> arenas;std::vector<std::thread> threads;
 std::mutex work_mutex;std::condition_variable ready,completed;std::atomic<uint64_t> next{0};
 Batch current{};uint64_t epoch=0;uint32_t finished=0;bool stopping=false;
 void stop()noexcept {
  {std::lock_guard<std::mutex> guard(work_mutex);stopping=true;}ready.notify_all();
  for(auto &thread:threads)if(thread.joinable())thread.join();
  threads.clear();
 }
 void worker(uint32_t index)noexcept {
  uint64_t seen=0;
  for(;;) {
   Batch batch;
   {std::unique_lock<std::mutex> lock(work_mutex);ready.wait(lock,[this,seen]{return stopping||epoch!=seen;});
    if(stopping)return;
    seen=epoch;batch=current;}
   for(;;) {
    const uint64_t row=next.fetch_add(1);if(row>=batch.count)break;
    double *diag=batch.diagnostics+row*8;
    std::fill_n(diag,8,std::numeric_limits<double>::quiet_NaN());
    try {
     const int64_t slot=batch.slots[row];const auto &pair=*static_cast<const PairData *>(batch.pairs[slot/14]);
     auto state=model_state(pair,slot%14,SupportPolicy::StrictPositive,batch.parameters+row*7);
     cpu_evaluate_state(*this,pair,state,arenas[index].data(),batch.objectives[row],diag,nullptr);batch.codes[row]=0;
    }catch(...) {batch.objectives[row]=std::numeric_limits<double>::quiet_NaN();batch.codes[row]=1;}
   }
   {std::lock_guard<std::mutex> lock(work_mutex);finished++;if(finished==workers)completed.notify_one();}
  }
 }
};

void cpu_evaluate_state(CpuSession &session,const PairData &pair,const ModelState &state,
 double *workspace,double &objective,double *diag,double *transition) {
 init_diagnostics(diag);validate_state(pair,state);OneMklThread single(session.mkl);
 build_q_snapshot(state,workspace);
 /* Q and S are no longer read after construction, so Q is the exp destination.
  * All polynomial work is outside the immutable snapshot region. */
 double *exp=workspace+KAKS_CORE_Q_OFFSET;
 exponential(session.mkl,session.primary_limit,session.fallback,workspace+KAKS_CORE_A_OFFSET,exp,workspace,diag);
 const double *sqrtpi=workspace+KAKS_CORE_SQRT_PI_OFFSET,*pi=state.pi.data();
 double min_entry=std::numeric_limits<double>::infinity(),row_error=0.;
 for(size_t i=0;i<n;i++) {
  double sum=0.;for(size_t j=0;j<n;j++) {
   const double value=pi[i]>0.&&pi[j]>0.?(exp[i*n+j]/sqrtpi[i])*sqrtpi[j]:static_cast<double>(i==j);
   exp[i*n+j]=value;sum+=value;min_entry=std::min(min_entry,value);
  }if(pi[i]>0.)row_error=std::max(row_error,std::abs(sum-1.));
 }
 double lnL=0.,minimum=std::numeric_limits<double>::infinity();
 for(const Pattern &pattern:pair.patterns()) {
  const double fh=exp[pattern.z0*n+pattern.z1]*pi[pattern.z0];minimum=std::min(minimum,fh);
  if(!(fh>0.)||!std::isfinite(fh))throw std::runtime_error("Nonpositive or nonfinite observed joint probability; no clipping is allowed");
  lnL+=std::log(fh)*pattern.weight;
 }
 if(!std::isfinite(lnL))throw std::runtime_error("Nonfinite objective");
 objective=-lnL;diag[4]=minimum;diag[5]=row_error;diag[6]=min_entry;diag[7]=single.observed;
 if(transition)std::copy_n(exp,nn,transition);
}
}

namespace {
void clear_error(char *error,uint32_t capacity){if(error&&capacity)error[0]=0;}
int report_error(const char *message,char *error,uint32_t cap,int code=1){if(error&&cap){size_t n=std::strlen(message);if(n>=cap)n=cap-1;std::memcpy(error,message,n);error[n]=0;}return code;}
kakscuda::CpuSession &session_of(void *raw){if(!raw)throw std::invalid_argument("Missing CPU session");return *static_cast<kakscuda::CpuSession *>(raw);}
const kakscuda::PairData &pair_of(const void *raw){if(!raw)throw std::invalid_argument("Missing live prepared pair");return *static_cast<const kakscuda::PairData *>(raw);}
void require_workspace(double *workspace,uint64_t width){if(!workspace||width<kakscuda::workspace_width)throw std::invalid_argument("Missing or undersized reusable CPU workspace");}
#define KAKS_CPU_TRY clear_error(error,cap);try
#define KAKS_CPU_CATCH catch(const std::exception &e){return report_error(e.what(),error,cap);}catch(...){return report_error("Unknown CPU backend error",error,cap);}
}
extern "C" uint32_t kaks_cpu_abi_version(void){return KAKS_CPU_ABI_VERSION;}
extern "C" uint64_t kaks_cpu_workspace_doubles(void){return kakscuda::workspace_width;}
extern "C" int kaks_cpu_create(const wchar_t *path,uint32_t workers,uint32_t limit,uint32_t fallback,void **output,char *error,uint32_t cap) {
 if(output)*output=nullptr;
 KAKS_CPU_TRY {if(!output||fallback>1)throw std::invalid_argument("Missing session output or invalid fallback policy");
  auto owned=std::make_unique<kakscuda::CpuSession>(path,workers,limit,fallback!=0);*output=owned.release();return 0;} KAKS_CPU_CATCH
}
extern "C" void kaks_cpu_free(void *raw){delete static_cast<kakscuda::CpuSession *>(raw);}
extern "C" int kaks_cpu_runtime_info(const void *raw,wchar_t *path,uint32_t path_cap,char *version,uint32_t version_cap,uint32_t *workers,char *error,uint32_t cap) {
 KAKS_CPU_TRY {if(!raw||!path||!version||!workers)throw std::invalid_argument("Missing runtime provenance output");
  const auto &s=*static_cast<const kakscuda::CpuSession *>(raw);
  if(path_cap<=s.mkl.loaded_path.size()||version_cap<=s.mkl.version.size())throw std::invalid_argument("Undersized runtime provenance buffer");
  std::copy(s.mkl.loaded_path.begin(),s.mkl.loaded_path.end(),path);path[s.mkl.loaded_path.size()]=0;
  std::copy(s.mkl.version.begin(),s.mkl.version.end(),version);version[s.mkl.version.size()]=0;*workers=s.workers;return 0;} KAKS_CPU_CATCH
}
extern "C" int kaks_cpu_evaluate_batch(void *raw,const void *const *pairs,uint32_t pair_count,const int64_t *slots,const double *parameters,uint64_t count,
 double *objectives,double *diagnostics,uint32_t *codes,uint64_t capacity,char *error,uint32_t cap) {
 KAKS_CPU_TRY {auto &s=session_of(raw);std::unique_lock<std::mutex> call(s.call_mutex,std::try_to_lock);
  if(!call.owns_lock())throw std::runtime_error("Concurrent calls on one CPU session are not supported");
  if(capacity<count||count>static_cast<uint64_t>(std::numeric_limits<int64_t>::max()/8))throw std::invalid_argument("Invalid batch output capacity or request count");
  if(!count)return 0;
  if(!pairs||!pair_count||!slots||!parameters||!objectives||!diagnostics||!codes)throw std::invalid_argument("Missing batch buffer");
  for(uint64_t row=0;row<count;row++) {
   const int64_t slot=slots[row];if(slot<0||static_cast<uint64_t>(slot)>=static_cast<uint64_t>(pair_count)*14)throw std::invalid_argument("Batch slot outside supplied pair table");
   auto state=kakscuda::model_state(pair_of(pairs[slot/14]),slot%14,kakscuda::SupportPolicy::StrictPositive,parameters+row*7);
   kakscuda::validate_state(pair_of(pairs[slot/14]),state);
  }
  const int code=s.batch({pairs,slots,parameters,count,objectives,diagnostics,codes});
  if(code)return report_error("One or more CPU objectives failed numerically; inspect retained row codes and diagnostics",error,cap,code);
  return 0;
 } KAKS_CPU_CATCH
}
extern "C" int kaks_cpu_evaluate_one(void *raw,const void *pair,uint32_t model,const double *x,double *workspace,uint64_t size,
 double *objective,double *diag,double *transition,char *error,uint32_t cap) {
 KAKS_CPU_TRY {auto &s=session_of(raw);std::unique_lock<std::mutex> call(s.call_mutex,std::try_to_lock);
  if(!call.owns_lock())throw std::runtime_error("Concurrent calls on one CPU session are not supported");
  require_workspace(workspace,size);
  if(!objective)throw std::invalid_argument("Missing objective output");
  const auto &p=pair_of(pair);
  auto state=kakscuda::model_state(p,model,kakscuda::SupportPolicy::StrictPositive,x);
  kakscuda::cpu_evaluate_state(s,p,state,workspace,*objective,diag,transition);return 0;} KAKS_CPU_CATCH
}
extern "C" int kaks_cpu_evaluate_expanded(void *raw,const void *pair,const double *pi,const double *rates,double omega,double time,
 double *workspace,uint64_t size,double *objective,double *diag,double *transition,char *error,uint32_t cap) {
 KAKS_CPU_TRY {auto &s=session_of(raw);std::unique_lock<std::mutex> call(s.call_mutex,std::try_to_lock);
  if(!call.owns_lock())throw std::runtime_error("Concurrent calls on one CPU session are not supported");
  require_workspace(workspace,size);
  if(!pi||!rates||!objective)throw std::invalid_argument("Missing expanded-state input or output");
  kakscuda::ModelState state{};state.config=kakscuda::model_config(13,kakscuda::SupportPolicy::StrictPositive);
  std::copy_n(pi,64,state.pi.begin());std::copy_n(rates,6,state.rates.begin());state.x[0]=time;state.kappa=rates[0];state.omega=omega;
  kakscuda::cpu_evaluate_state(s,pair_of(pair),state,workspace,*objective,diag,transition);return 0;} KAKS_CPU_CATCH
}
extern "C" int kaks_cpu_matrix_exp(void *raw,const double *matrix,double *out,double *workspace,uint64_t size,double *diag,char *error,uint32_t cap) {
 KAKS_CPU_TRY {auto &s=session_of(raw);std::unique_lock<std::mutex> call(s.call_mutex,std::try_to_lock);
  if(!call.owns_lock())throw std::runtime_error("Concurrent calls on one CPU session are not supported");
  require_workspace(workspace,size);
  if(!matrix||!out)throw std::invalid_argument("Missing matrix exponential input/output");
  kakscuda::OneMklThread single(s.mkl);
  kakscuda::exponential(s.mkl,s.primary_limit,s.fallback,matrix,out,workspace,diag);diag[7]=single.observed;return 0;} KAKS_CPU_CATCH
}
