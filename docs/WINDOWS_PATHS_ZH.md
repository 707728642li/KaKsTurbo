# Windows 长路径与安装（1.1.0rc10）

**输入和结果目录不再要求用户手工缩短路径。** `run`、`prepare`、`validate`、`status`、`resume`、兼容导出都在内部使用 Windows 扩展路径。带空格的命令行参数加引号即可，中文路径也经过实测。程序不修改注册表、全局 PATH 或系统执行策略。

## 解压和启动

完整 ZIP 与同目录的 `Install-KaKsTurbo.ps1` 配套使用。如果 Windows 资源管理器报告路径过长，使用此解压器，不必搬动或重新规划项目目录。它逐文件验证 SHA256，拒绝覆盖现有目标；出错保留部分文件供检查。

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Install-KaKsTurbo.ps1 -Archive .\KaKsTurbo-1.1.0rc10-docs2-windows-x86_64.zip -Destination "D:\My projects\KaKsTurbo"
& "D:\My projects\KaKsTurbo\kaksturbo.exe" doctor --backend cpu
& "D:\My projects\KaKsTurbo\kaksturbo.exe" run --input "D:\My projects\data\pairs.axt" --output "D:\My projects\results\run 1" --backend cpu --threads 2
```

上面的 `Bypass` 只作用于该 PowerShell 进程，不更改用户/系统策略。如果单位策略禁用脚本，应遵循本机管理要求。

**安装目录本身超过旧式 shell 的 EXE 启动限制时**，使用包内 `kaksturbo.ps1`。解压器会打印完整可复制的启动命令，例如：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "\\?\D:\very deep directory\KaKsTurbo\kaksturbo.ps1" run --input "D:\data\pairs.axt" --output "D:\results\run 1" --backend cpu --threads 2
```

`\\?\` 只在这种旧 shell 的脚本启动位置需要；输入/输出参数无需自己加前缀。程序在用户 TEMP 中创建临时目录链接，让 Python/DLL 及历史 CLI 使用较短的入口；**不复制引擎、不移动输入或结果**。正常退出后只移除自身的临时链接。TEMP 需要是可写的本地目录。

## 验证范围

Windows 11 原生、`LongPathsEnabled=0`：输入 389 字符、输出目录 374 字符、最深结果文件 519 字符；普通 EXE 从 199 字符安装目录启动；PowerShell 5.1 入口从 334 字符安装目录启动。组合包含空格和中文。已验证真实 CPU/GPU、保存一块后中断并恢复、预处理元数据、gzip 输入、22 列兼容导出和原有结果保护。测试使用完整 14 模型、FP64 和原质量阈值；微型测试为安装回归，不是新的速度或生物学准确性证据。

文件系统单个文件名/目录名长度及权限仍适用；未认证网络共享、云盘占位文件或所有第三方 ZIP 工具。Windows shell/解压器的限制与程序内部路径支持分别说明。旧 rc9 检查点必须用原 rc9 继续恢复；新版不绕过产品身份验证。旧结果可继续只读导出。

核心标准表、NA、优化器及完整模型平均均未改变。旧压缩包和失败诊断证据保留。当前下载入口见仓库 README。
