/* KaKsTurbo portable launcher. GPL-3.0-only. Numerical algorithms are separate. */
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <wchar.h>
#include <stdio.h>
#include <stdlib.h>
#include <winioctl.h>
/* Only an installation alias is temporary. Scientific inputs/results never
   move. Mount-point reparse points need no administrator/symlink privilege. */
static int alias_root(const wchar_t *target,wchar_t *folder,wchar_t *alias){
 wchar_t tmp[MAX_PATH],nt[32768];
 if(!GetTempPathW(MAX_PATH,tmp)||!GetTempFileNameW(tmp,L"kkt",0,folder))return 0;
 if(!DeleteFileW(folder)||!CreateDirectoryW(folder,NULL))return 0;
 swprintf(alias,32768,L"%ls\\root",folder);
 if(!CreateDirectoryW(alias,NULL)){RemoveDirectoryW(folder);return 0;}
 if(wcsncmp(target,L"\\\\?\\",4)==0)swprintf(nt,32768,L"\\??\\%ls",target+4);
 else if(wcsncmp(target,L"\\\\",2)==0)swprintf(nt,32768,L"\\??\\UNC\\%ls",target+2);
 else swprintf(nt,32768,L"\\??\\%ls",target);
 size_t sn=wcslen(nt)*sizeof(wchar_t),pn=wcslen(target)*sizeof(wchar_t);
 size_t bytes=16+sn+2+pn+2;
 if(bytes>MAXIMUM_REPARSE_DATA_BUFFER_SIZE){RemoveDirectoryW(alias);RemoveDirectoryW(folder);return 0;}
 BYTE *data=calloc(1,bytes);if(!data){RemoveDirectoryW(alias);RemoveDirectoryW(folder);return 0;}
 *(DWORD*)data=IO_REPARSE_TAG_MOUNT_POINT;*(WORD*)(data+4)=(WORD)(bytes-8);
 *(WORD*)(data+10)=(WORD)sn;*(WORD*)(data+12)=(WORD)(sn+2);*(WORD*)(data+14)=(WORD)pn;
 memcpy(data+16,nt,sn);memcpy(data+16+sn+2,target,pn);
 HANDLE h=CreateFileW(alias,GENERIC_WRITE,0,NULL,OPEN_EXISTING,FILE_FLAG_OPEN_REPARSE_POINT|FILE_FLAG_BACKUP_SEMANTICS,NULL);
 DWORD used=0;int ok=h!=INVALID_HANDLE_VALUE&&DeviceIoControl(h,FSCTL_SET_REPARSE_POINT,data,(DWORD)bytes,NULL,0,&used,NULL);
 if(h!=INVALID_HANDLE_VALUE)CloseHandle(h);
 free(data);
 if(!ok){RemoveDirectoryW(alias);RemoveDirectoryW(folder);}return ok;
}
int wmain(int argc,wchar_t**argv){
 wchar_t path[32768];DWORD n=GetModuleFileNameW(NULL,path,32768);
 if(!n||n>=32768)return 126;
 wchar_t*slash=wcsrchr(path,L'\\');if(!slash)return 126;*slash=0;
 wchar_t folder[MAX_PATH]={0},alias[32768]={0};const wchar_t *launchroot=path;
 // The interpreter and dependent DLL loader must also see a short entry path.
 // Deep results are handled separately using extended-length filesystem paths.
 if(wcslen(path)>120){
  if(!alias_root(path,folder,alias)){
   fwprintf(stderr,L"KaKsTurbo: cannot create its temporary installation alias (Windows error %lu). Check TEMP write access; the installation and results have not been moved.\n",GetLastError());return 126;}
  launchroot=alias;
 }
 wchar_t python[32768];swprintf(python,32768,L"%ls\\runtime\\python.exe",launchroot);
 // Reuse untouched command-line quoting after removing argv[0].
 wchar_t *tail=GetCommandLineW();
 if(*tail==L'"'){tail++;while(*tail&&*tail!=L'"')tail++;if(*tail)tail++;}
 else while(*tail&&*tail!=L' '&&*tail!=L'\t')tail++;
 while(*tail==L' '||*tail==L'\t')tail++;
 size_t cap=wcslen(python)+wcslen(tail)+64;wchar_t*cmd=calloc(cap,sizeof(wchar_t));
 if(!cmd){if(alias[0]){RemoveDirectoryW(alias);RemoveDirectoryW(folder);}return 126;}swprintf(cmd,cap,L"\"%ls\" -I -B -m kaksturbo %ls",python,tail);
 STARTUPINFOW si={0};si.cb=sizeof(si);PROCESS_INFORMATION pi={0};
 if(!CreateProcessW(python,cmd,NULL,NULL,TRUE,0,NULL,NULL,&si,&pi)){
  fwprintf(stderr,L"KaKsTurbo: cannot start bundled Python (Windows error %lu). Extract the complete ZIP before running.\n",GetLastError());free(cmd);if(alias[0]){RemoveDirectoryW(alias);RemoveDirectoryW(folder);}return 126;}
 free(cmd);CloseHandle(pi.hThread);WaitForSingleObject(pi.hProcess,INFINITE);DWORD code=1;GetExitCodeProcess(pi.hProcess,&code);CloseHandle(pi.hProcess);
 if(alias[0]){RemoveDirectoryW(alias);RemoveDirectoryW(folder);}return (int)code;
}
#else
#include <unistd.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(int argc,char**argv){
 char path[PATH_MAX],python[PATH_MAX];ssize_t n=readlink("/proc/self/exe",path,sizeof(path)-1);
 if(n<=0||n>=sizeof(path)-1)return 126;path[n]=0;char*slash=strrchr(path,'/');if(!slash)return 126;*slash=0;
 if(snprintf(python,sizeof(python),"%s/runtime/bin/python",path)>=(int)sizeof(python))return 126;
 char library[PATH_MAX];
 if(snprintf(library,sizeof(library),"%s/runtime/lib",path)>=(int)sizeof(library))return 126;
 // Scope the bundled interpreter's shared libraries to this child process.
 // The launcher neither changes a shell profile nor uses conda activation.
 if(setenv("LD_LIBRARY_PATH",library,1)!=0)return 126;
 char**args=calloc(argc+5,sizeof(char*));if(!args)return 126;
 args[0]=python;args[1]="-I";args[2]="-B";args[3]="-m";args[4]="kaksturbo";
 for(int i=1;i<argc;i++)args[i+4]=argv[i];args[argc+4]=NULL;execv(python,args);
 perror("KaKsTurbo: cannot start bundled Python; extract the complete archive");return 126;
}
#endif
