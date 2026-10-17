#ifndef KAKSFORGE_LINUX_WINDOWS_COMPAT_V1_H
#define KAKSFORGE_LINUX_WINDOWS_COMPAT_V1_H
/* Platform-only ABI overlay. Numerical translation units are retained.
 * wchar_t is the native Linux wchar_t used by ctypes.c_wchar_p. Paths are
 * explicitly encoded as UTF-8; no process locale mutation is needed. */
#include <stdint.h>
#include <cstring>
#include <string>
#include <cstdlib>
#include <cwchar>
#include <cmath>
#include <cerrno>
#include <dlfcn.h>
#include <link.h>
#include <pthread.h>
#include <time.h>
#ifndef __cdecl
#define __cdecl
#endif
#ifndef __declspec
#define __declspec(x)
#endif
typedef void *HMODULE;
typedef void (*FARPROC)();
typedef uint32_t DWORD;
typedef pthread_mutex_t CRITICAL_SECTION;
struct LARGE_INTEGER { int64_t QuadPart; };
#define LOAD_WITH_ALTERED_SEARCH_PATH 0
inline std::string kaksforge_utf8_path(const wchar_t *path) {
    std::string out;
    if (!path) return out;
    for (; *path; ++path) {
        const uint32_t c=static_cast<uint32_t>(*path);
        if(c>=0xd800 && c<=0xdfff)return std::string();
        if(c<=0x7f)out.push_back(static_cast<char>(c));
        else if(c<=0x7ff){out.push_back(static_cast<char>(0xc0|(c>>6)));out.push_back(static_cast<char>(0x80|(c&63)));}
        else if(c<=0xffff && !(c>=0xd800 && c<=0xdfff)){out.push_back(static_cast<char>(0xe0|(c>>12)));out.push_back(static_cast<char>(0x80|((c>>6)&63)));out.push_back(static_cast<char>(0x80|(c&63)));}
        else if(c<=0x10ffff){out.push_back(static_cast<char>(0xf0|(c>>18)));out.push_back(static_cast<char>(0x80|((c>>12)&63)));out.push_back(static_cast<char>(0x80|((c>>6)&63)));out.push_back(static_cast<char>(0x80|(c&63)));}
        else return std::string();
    }
    return out;
}
inline bool kaksforge_absolute_blas_path(const wchar_t *path) {return path && *path==L'/';}
inline HMODULE LoadLibraryExW(const wchar_t *path,void *,int) {
    const std::string text=kaksforge_utf8_path(path);
    return text.empty()?NULL:dlopen(text.c_str(),RTLD_NOW|RTLD_LOCAL);
}
inline FARPROC GetProcAddress(HMODULE module,const char *name) {
    void *address=dlsym(module,name);FARPROC function=NULL;
    std::memcpy(&function,&address,sizeof(function));return function;
}
inline int FreeLibrary(HMODULE module) {return dlclose(module)==0;}
inline DWORD GetLastError() {return static_cast<DWORD>(errno);}
inline DWORD GetModuleFileNameW(HMODULE module,wchar_t *out,DWORD capacity) {
    struct link_map *map=NULL;
    if(!out||!capacity||dlinfo(module,RTLD_DI_LINKMAP,&map)!=0||!map||!map->l_name)return 0;
    char *resolved=realpath(map->l_name,NULL);if(!resolved)return 0;
    const unsigned char *text=reinterpret_cast<const unsigned char *>(resolved);
    DWORD used=0;bool valid=true;
    while(*text && used+1<capacity) {
        uint32_t c=*text++;int extra=0;
        if(c<128){}
        else if((c&0xe0)==0xc0){c&=31;extra=1;}
        else if((c&0xf0)==0xe0){c&=15;extra=2;}
        else if((c&0xf8)==0xf0){c&=7;extra=3;}
        else {valid=false;break;}
        for(int i=0;i<extra;i++){if((*text&0xc0)!=0x80){valid=false;break;}c=(c<<6)|(*text++&63);}
        if(!valid)break;out[used++]=static_cast<wchar_t>(c);
    }
    if(*text)valid=false;out[used]=0;std::free(resolved);return valid?used:0;
}
inline void InitializeCriticalSection(CRITICAL_SECTION *mutex) {
    pthread_mutexattr_t attr;pthread_mutexattr_init(&attr);
    pthread_mutexattr_settype(&attr,PTHREAD_MUTEX_RECURSIVE);
    pthread_mutex_init(mutex,&attr);pthread_mutexattr_destroy(&attr);
}
inline void DeleteCriticalSection(CRITICAL_SECTION *mutex){pthread_mutex_destroy(mutex);}
inline void EnterCriticalSection(CRITICAL_SECTION *mutex){pthread_mutex_lock(mutex);}
inline void LeaveCriticalSection(CRITICAL_SECTION *mutex){pthread_mutex_unlock(mutex);}
inline int QueryPerformanceFrequency(LARGE_INTEGER *value){value->QuadPart=1000000000LL;return 1;}
inline int QueryPerformanceCounter(LARGE_INTEGER *value){struct timespec t;if(clock_gettime(CLOCK_MONOTONIC,&t))return 0;value->QuadPart=static_cast<int64_t>(t.tv_sec)*1000000000LL+t.tv_nsec;return 1;}
inline int _finite(double value){return std::isfinite(value)?1:0;}
#ifdef KAKSFORGE_HOST_FINITE_RENAME
/* glibc supplies ::finite(double). Rename the source's anonymous helper
 * after the system headers are parsed; its body and all calls are retained. */
#define finite kaksforge_local_host_finite
#endif
#endif
