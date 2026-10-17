"""Path-only launch adapter for the unchanged historical executable.

The frozen historical Python driver still owns scheduling, all model settings,
parsing, row checks and quality reporting. Only its Popen dependency is bound
to this adapter: legacy native file APIs see short directory aliases. A junction
references the original directory; no input, output or binary is copied.
"""
from __future__ import annotations
import ctypes as C
from ctypes import wintypes as W
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace


def junction(link, target):
    k=C.WinDLL('kernel32',use_last_error=True)
    k.CreateFileW.argtypes=[W.LPCWSTR,W.DWORD,W.DWORD,C.c_void_p,W.DWORD,W.DWORD,W.HANDLE]
    k.CreateFileW.restype=W.HANDLE
    k.DeviceIoControl.argtypes=[W.HANDLE,W.DWORD,C.c_void_p,W.DWORD,C.c_void_p,W.DWORD,C.POINTER(W.DWORD),C.c_void_p]
    k.DeviceIoControl.restype=W.BOOL
    k.CloseHandle.argtypes=[W.HANDLE];k.CloseHandle.restype=W.BOOL
    target=str(target)
    normal=target[4:] if target.startswith('\\\\?\\') else target
    nt='\\??\\UNC\\'+normal[2:] if normal.startswith('\\\\') else '\\??\\'+normal
    sub=nt.encode('utf-16-le');display=target.encode('utf-16-le')
    strings=sub+b'\0\0'+display+b'\0\0'
    data=struct.pack('<IHHHHHH',0xA0000003,8+len(strings),0,0,len(sub),len(sub)+2,len(display))+strings
    if len(data)>16384:raise OSError('Installation path exceeds the Windows junction target limit')
    Path(link).mkdir()
    handle=k.CreateFileW(str(link),0x40000000,0,None,3,0x02200000,None)
    if handle==W.HANDLE(-1).value:raise C.WinError(C.get_last_error())
    try:
        used=W.DWORD()
        if not k.DeviceIoControl(handle,0x000900A4,data,len(data),None,0,C.byref(used),None):
            raise C.WinError(C.get_last_error())
    finally:k.CloseHandle(handle)


class HistoricalProcess:
    def __init__(self,command,**kwargs):
        if len(command)!=9 or command[1]!='-i' or command[3]!='-o' or command[5:]!=['-m','MA','-c','1']:
            raise ValueError('Unexpected historical command; path adaptation refused')
        self.directory=Path(tempfile.mkdtemp(prefix='kkt-legacy-'))
        self.links=[]
        try:
            mapped=list(command)
            parents={}
            for index in (0,2,4):
                path=Path(command[index])
                if path.parent not in parents:
                    link=self.directory/str(len(parents))
                    self.links.append(link)
                    junction(link,path.parent)
                    parents[path.parent]=link
                mapped[index]=str(parents[path.parent]/path.name)
            # Retained alongside the normal historical logs and checkpoint hash.
            record=Path(command[4]).with_suffix('.windows_path.json')
            record.write_text(json.dumps(dict(schema='kaksturbo.windows_historical_paths/1',
                original_command=command,launch_command=mapped,data_copied=False,
                algorithm_changed=False),indent=2)+'\n',encoding='utf-8')
            self.process=subprocess.Popen(mapped,**kwargs)
        except BaseException:
            self.cleanup()
            raise

    def cleanup(self):
        # rmdir removes a directory junction itself, never its target contents.
        for link in reversed(self.links):
            if os.path.lexists(link):os.rmdir(link)
        self.links=[]
        self.directory.rmdir()

    def __getattr__(self,name):return getattr(self.process,name)

    def wait(self,*args,**kwargs):
        code=self.process.wait(*args,**kwargs)
        if self.links:self.cleanup()
        return code


def main():
    if os.name!='nt':raise RuntimeError('Windows path adapter must not run on other platforms')
    root=Path(os.environ['KAKSFORGE_RUNTIME_ROOT'])
    sys.path.insert(0,str(root/'scripts'))
    import product_v1_engine_reproduce as historical
    historical.subprocess=SimpleNamespace(Popen=HistoricalProcess,STDOUT=subprocess.STDOUT)
    return historical.main()


if __name__=='__main__':raise SystemExit(main())
