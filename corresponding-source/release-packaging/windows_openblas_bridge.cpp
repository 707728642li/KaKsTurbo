// ABI compatibility adapter, not a numerical implementation. GPL-3.0-only.
// The immutable numerical DLLs request one BLAS thread per worker. This private
// OpenBLAS instance is fixed to one thread once, before concurrent evaluations.
// Historical MKL export names are only an ABI; no Intel library is loaded.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
using Gemm=void (*)(int,int,int,int64_t,int64_t,int64_t,double,const double*,int64_t,const double*,int64_t,double,double*,int64_t);
using Set=void (*)(int); using Get=int (*)(); using Config=char* (*)();
static std::once_flag once;
static HMODULE handle=nullptr;
static Gemm gemm=nullptr; static Get get=nullptr; static Config config=nullptr;
static std::string error;
static thread_local int local_request=1;
static void initialize(){
 std::call_once(once,[]{
  HMODULE self=nullptr;
  if(!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS|GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
      reinterpret_cast<LPCWSTR>(&initialize),&self)){error="cannot identify bridge module";return;}
  wchar_t path[32768]{};DWORD n=GetModuleFileNameW(self,path,32768);
  if(!n||n>=32768){error="cannot resolve bridge directory";return;}
  std::wstring directory(path,n);directory=directory.substr(0,directory.find_last_of(L"\\/"));
  auto library=directory+L"\\libscipy_openblas64_-63c857e738469261263c764a36be9436.dll";
  handle=LoadLibraryExW(library.c_str(),nullptr,LOAD_WITH_ALTERED_SEARCH_PATH);
  if(!handle){error="OpenBLAS load failed: "+std::to_string(GetLastError());return;}
  gemm=reinterpret_cast<Gemm>(GetProcAddress(handle,"scipy_cblas_dgemm64_"));
  auto set=reinterpret_cast<Set>(GetProcAddress(handle,"scipy_openblas_set_num_threads64_"));
  get=reinterpret_cast<Get>(GetProcAddress(handle,"scipy_openblas_get_num_threads64_"));
  config=reinterpret_cast<Config>(GetProcAddress(handle,"scipy_openblas_get_config64_"));
  if(!gemm||!set||!get||!config){error="missing ILP64 CBLAS/thread/config exports";return;}
  set(1);if(get()!=1)error="OpenBLAS did not honor single-thread policy";
 });
}
extern "C" {
__declspec(dllexport) void cblas_dgemm_64(int layout,int ta,int tb,int64_t m,int64_t n,int64_t k,double alpha,const double*a,int64_t lda,const double*b,int64_t ldb,double beta,double*c,int64_t ldc){
 initialize();if(!error.empty())std::abort();gemm(layout,ta,tb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc);
}
__declspec(dllexport) int MKL_Set_Num_Threads_Local(int count){
 initialize();int previous=local_request;local_request=count;
 return previous;
}
__declspec(dllexport) int MKL_Get_Max_Threads(){
 initialize();return error.empty()&&(local_request==0||local_request==1)?get():0;
}
__declspec(dllexport) void MKL_Get_Version_String(char*out,int capacity){
 initialize();if(out&&capacity>0)std::snprintf(out,size_t(capacity),"KaKsTurbo Windows OpenBLAS64 adapter; fixed single BLAS thread; %s",error.empty()?config():error.c_str());
}
}
