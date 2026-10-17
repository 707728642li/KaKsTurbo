# Windows paths and installation — 1.1.0rc10

Long input/output directories, spaces and Unicode are handled inside the program. Quote paths containing spaces. No registry or global PATH changes are needed. All original numerical settings and scientific assets remain pinned.

Use `Install-KaKsTurbo.ps1` next to the ZIP for long-path-safe extraction and SHA256 verification:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Install-KaKsTurbo.ps1 -Destination "D:\My projects\KaKsTurbo"
& "D:\My projects\KaKsTurbo\kaksturbo.exe" run --input "D:\data\pairs.axt" --output "D:\My projects\results\run 1" --backend cpu --threads 2
```

If the installation path itself exceeds the calling shell's EXE limit, use the packaged PowerShell entry. The installer prints the exact command:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "\\?\D:\very deep directory\KaKsTurbo\kaksturbo.ps1" doctor --backend cpu
```

This prefix is only needed for legacy-shell script invocation; ordinary input/output arguments are normalized automatically. The launcher creates temporary directory junctions in writable local TEMP, without copying the runtime or moving results, and removes its own links on normal exit. ExecutionPolicy Bypass applies only to that process; organizational policy still takes precedence.

Tested on native Windows 11 with LongPathsEnabled=0: 389-character input, 374-character result directory, 519-character deepest result file, 199-character direct-EXE installation and 334-character installation through Windows PowerShell 5.1. Combined tests include spaces and Chinese characters, real CPU/GPU fits, interrupted-checkpoint resume, prepared metadata, gzip input and compatible export. These are small packaging regressions, not performance/convergence claims. Per-component filesystem limits and access rights still apply; network/cloud filesystems and third-party extractors are not certified.

Use the original rc9 binary to resume rc9 checkpoints; product identity checks remain strict. Existing completed results can still be exported read-only. Previous qualified package records are retained; download availability is listed in the repository README.
