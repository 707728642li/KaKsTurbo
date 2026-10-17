param(
    [string]$Archive = (Join-Path $PSScriptRoot 'KaKsTurbo-1.1.0rc10-windows-x86_64.zip'),
    [string]$Destination = (Join-Path (Get-Location).Path 'KaKsTurbo-1.1.0rc10')
)
# Long-path-safe ZIP extraction and manifest verification, Windows PowerShell 5.1+.
# GPL-3.0-only. Existing destinations are never overwritten or deleted.
$ErrorActionPreference = 'Stop'
function Extended-Path([string]$Value) {
    $Value = $Value.Replace('/', '\')
    if ($Value.StartsWith('\\.\')) { throw 'Device namespace is not an installation directory.' }
    if (-not $Value.StartsWith('\\?\')) {
        if (-not ($Value -match '^[A-Za-z]:\\' -or $Value.StartsWith('\\'))) {
            $Value = [IO.Path]::GetFullPath($Value)
        }
        if ($Value.StartsWith('\\')) { $Value = '\\?\UNC\' + $Value.Substring(2) }
        else { $Value = '\\?\' + $Value }
    }
    if ($Value.Split('\') -contains '..' -or $Value.Split('\') -contains '.') {
        throw 'Use a destination without dot/dot-dot components.'
    }
    return $Value.TrimEnd('\')
}
function File-SHA256([string]$Name) {
    $stream = [IO.File]::OpenRead($Name)
    $algorithm = [Security.Cryptography.SHA256]::Create()
    try { return ([BitConverter]::ToString($algorithm.ComputeHash($stream))).Replace('-', '').ToLowerInvariant() }
    finally { $algorithm.Dispose(); $stream.Dispose() }
}
$ktTarget = Extended-Path $Destination
$ktArchive = Extended-Path $Archive
$ktZip = $null
$ktStream = $null
try {
    if ([IO.Directory]::Exists($ktTarget) -or [IO.File]::Exists($ktTarget)) {
        throw 'Destination exists. Keep it intact and choose a new installation directory.'
    }
    Add-Type -AssemblyName System.IO.Compression
    $ktStream = [IO.File]::OpenRead($ktArchive)
    $ktZip = [IO.Compression.ZipArchive]::new($ktStream, [IO.Compression.ZipArchiveMode]::Read)
    $ktPrefix = 'KaKsTurbo-1.1.0rc10/'
    foreach ($entry in $ktZip.Entries) {
        if (-not $entry.FullName.StartsWith($ktPrefix) -or $entry.FullName.Contains('\') -or $entry.FullName.Contains(':') -or
            $entry.FullName.Split('/') -contains '..' -or $entry.FullName.Split('/') -contains '.') { throw 'Unsafe archive entry.' }
    }
    [IO.Directory]::CreateDirectory($ktTarget) | Out-Null
    $ktCount = 0
    foreach ($entry in $ktZip.Entries) {
        $ktRelative = $entry.FullName.Substring($ktPrefix.Length).Replace('/', '\')
        if (-not $ktRelative) { continue }
        $ktFile = $ktTarget + '\' + $ktRelative
        if ($entry.FullName.EndsWith('/')) { [IO.Directory]::CreateDirectory($ktFile) | Out-Null; continue }
        [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($ktFile)) | Out-Null
        $inputStream = $entry.Open()
        $outputStream = $null
        try {
            $outputStream = [IO.File]::Open($ktFile, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write)
            $inputStream.CopyTo($outputStream)
        } finally { if ($outputStream) { $outputStream.Dispose() }; $inputStream.Dispose() }
        $ktCount++
        Write-Progress -Activity 'Extracting KaKsTurbo' -Status "$ktCount files" -PercentComplete (100*$ktCount/$ktZip.Entries.Count)
    }
    Write-Progress -Activity 'Extracting KaKsTurbo' -Completed
    $manifest = [IO.File]::ReadAllText($ktTarget + '\FILES.sha256.json') | ConvertFrom-Json
    $ktChecked = 0
    foreach ($entry in $manifest.PSObject.Properties) {
        if ($entry.Name.StartsWith('/') -or $entry.Name.Contains('\') -or $entry.Name.Contains(':') -or $entry.Name.Split('/') -contains '..') {
            throw 'Unsafe manifest entry.'
        }
        $ktFile = $ktTarget + '\' + $entry.Name.Replace('/', '\')
        if ((File-SHA256 $ktFile) -ne $entry.Value.sha256) { throw ('SHA256 mismatch: ' + $entry.Name) }
        $ktChecked++
        Write-Progress -Activity 'Verifying KaKsTurbo' -Status "$ktChecked files" -PercentComplete (100*$ktChecked/@($manifest.PSObject.Properties).Count)
    }
    Write-Progress -Activity 'Verifying KaKsTurbo' -Completed
    Write-Output "Installed and SHA256 verified $ktChecked files at $Destination"
    Write-Output 'For deeply nested installations, use the supplied PowerShell entry; no folder relocation is needed:'
    Write-Output ('powershell.exe -NoProfile -ExecutionPolicy Bypass -File "' + $ktTarget + '\kaksturbo.ps1" doctor --backend cpu')
} catch {
    [Console]::Error.WriteLine('KaKsTurbo installation failed: ' + $_.Exception.Message + ' Existing files and any partial extraction are retained.')
    exit 1
} finally {
    if ($ktZip) { $ktZip.Dispose() }
    if ($ktStream) { $ktStream.Dispose() }
}
