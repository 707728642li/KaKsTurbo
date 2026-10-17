# KaKsTurbo Windows launch entry for paths the calling shell cannot launch as EXE.
# GPL-3.0-only. No administrator rights, registry edits, data copying or fitting changes.
$ErrorActionPreference = 'Stop'
$ktArguments = $args
$ktRoot = $PSScriptRoot
if (-not ('KaKsTurboPathAlias' -as [type])) {
    Add-Type -TypeDefinition @'
using System;
using System.IO;
using System.Text;
using System.Runtime.InteropServices;
public static class KaKsTurboPathAlias {
    [DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError=true)]
    static extern IntPtr CreateFileW(string name,uint access,uint share,IntPtr security,uint creation,uint flags,IntPtr template);
    [DllImport("kernel32.dll", SetLastError=true)]
    static extern bool DeviceIoControl(IntPtr file,uint code,byte[] input,int length,IntPtr output,int size,out int used,IntPtr overlapped);
    [DllImport("kernel32.dll")] static extern bool CloseHandle(IntPtr handle);
    [DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError=true)]
    public static extern bool RemoveDirectoryW(string path);
    public static void Create(string link,string target) {
        Directory.CreateDirectory(link);
        string normal=target.StartsWith(@"\\?\") ? target.Substring(4) : target;
        string substitute=normal.StartsWith(@"\\") ? @"\??\UNC\"+normal.Substring(2) : @"\??\"+normal;
        byte[] sub=Encoding.Unicode.GetBytes(substitute), print=Encoding.Unicode.GetBytes(target);
        byte[] data=new byte[16+sub.Length+2+print.Length+2];
        if(data.Length>16384) throw new IOException("Installation path exceeds the Windows junction target limit.");
        Array.Copy(BitConverter.GetBytes(0xA0000003u),0,data,0,4);
        Array.Copy(BitConverter.GetBytes((ushort)(data.Length-8)),0,data,4,2);
        Array.Copy(BitConverter.GetBytes((ushort)sub.Length),0,data,10,2);
        Array.Copy(BitConverter.GetBytes((ushort)(sub.Length+2)),0,data,12,2);
        Array.Copy(BitConverter.GetBytes((ushort)print.Length),0,data,14,2);
        Array.Copy(sub,0,data,16,sub.Length);Array.Copy(print,0,data,18+sub.Length,print.Length);
        IntPtr file=CreateFileW(link,0x40000000,0,IntPtr.Zero,3,0x02200000,IntPtr.Zero);
        if(file==new IntPtr(-1)) throw new System.ComponentModel.Win32Exception(Marshal.GetLastWin32Error());
        try {int used;if(!DeviceIoControl(file,0x000900A4,data,data.Length,IntPtr.Zero,0,out used,IntPtr.Zero))
            throw new System.ComponentModel.Win32Exception(Marshal.GetLastWin32Error());}
        finally {CloseHandle(file);}
    }
}
'@
}
$ktTemporary = Join-Path ([IO.Path]::GetTempPath()) ('kkt-' + [Guid]::NewGuid().ToString('N'))
$ktAlias = Join-Path $ktTemporary 'root'
$ktExitCode = 126
try {
    [IO.Directory]::CreateDirectory($ktTemporary) | Out-Null
    [KaKsTurboPathAlias]::Create($ktAlias, $ktRoot)
    & (Join-Path $ktAlias 'kaksturbo.exe') @ktArguments
    $ktExitCode = $LASTEXITCODE
} catch {
    [Console]::Error.WriteLine('KaKsTurbo launcher: ' + $_.Exception.Message + ' Check TEMP write access. No installation or result files were moved.')
} finally {
    # Remove only our two directory entries; never recurse into the package.
    [KaKsTurboPathAlias]::RemoveDirectoryW($ktAlias) | Out-Null
    [KaKsTurboPathAlias]::RemoveDirectoryW($ktTemporary) | Out-Null
}
exit $ktExitCode
