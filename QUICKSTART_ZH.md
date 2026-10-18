<!-- homepage-refresh -->
# 快速入门

已有密码子比对输入，可直接使用下列最小示例；只有原始 CDS/蛋白时，继续阅读下面的比对准备流程。

在完整解压的 [Linux 便携包](https://github.com/707728642li/KaKsTurbo/releases/tag/v1.1.0rc10)目录执行，无需安装额外比对环境：

```bash
./kaksturbo doctor --backend cpu
./kaksturbo run --input examples/three_pair_smoke.axt --output results_demo --backend cpu --threads 8
head -n 4 results_demo/result.tsv
```

该合成示例为三条有向请求，预期两个有限比值和一个相同序列对 NA。[真实输出预览](examples/results/README.md) · [Windows 操作](docs/WINDOWS_PATHS_ZH.md) · [完整中文 CLI](USER_GUIDE_ZH.txt)

自己的 AXT 可直接替换 `--input`；FASTA 必须通过 `--pairs` 指定配对。输出目录须新建。

---

## 从这里开始：原始 CDS 到 Ka/Ks

[English](QUICKSTART.md) · [完整中文教程](docs/RAW_CDS_TUTORIAL_ZH.md) · [批量脚本](scripts/prepare_cds_pairs.py)

把 **Linux 便携包完整解压到新目录**，进入该目录打开终端。CPU 计算不需要显卡或 CUDA。包内 Python 运行 KaKsTurbo；下面独立的可选比对环境运行 MAFFT/PAL2NAL。

## Linux：只需安装一次上游比对软件

已有合适的环境时直接复用，通过 `--env` 指定，不必重复安装。否则在解压目录内执行：

```bash
conda env create -p "$PWD/envs/input-prep" -f workflow/raw_cds_environment.yml
```

配方使用 conda-forge/bioconda，固定本教程验证过的 MAFFT 7.526、PAL2NAL 14.1，并安装 Python 和 GNU parallel。首次安装需要能访问这些源；比对程序未捆绑进主程序包。KaKsTurbo 自带的计算运行环境不受影响。

## Linux：一条命令跑完整示例

```bash
bash scripts/run_raw_cds_demo_linux.sh --output "$PWD/tutorial_demo"
```

依次完成 CPU 诊断、CDS/PEP 一致性检查、蛋白比对、密码子回译、`prepare`、`validate` 和 CPU 计算。先用两个单线程比对任务并发，结束后用两个 CPU 线程拟合。结果见 `tutorial_demo/results_cpu/result.tsv` 和 `quality_summary.txt`。

这是 4 个有向请求的人工示例，包含反向请求及相同序列对；实测得到 3 个有限 Ka/Ks、1 个 NA。示例用于教会输入和运行方式，不作为生物学或性能基准。

复用已有环境，或 GNU parallel 在另一个位置时：

```bash
bash scripts/run_raw_cds_demo_linux.sh \
  --env /absolute/path/to/existing/input-prep \
  --parallel /absolute/path/to/parallel \
  --output "$PWD/tutorial_demo_existing_env"
```

每次指定新的输出目录；出错时停止后续步骤并保留记录。`--jobs` 与 `--threads` 分别控制前后两个阶段。在源码目录运行时，先设置 `KAKSTURBO_BIN=/absolute/path/to/installed/kaksturbo`。全部参数见 `--help`；该示例脚本只运行 CPU。

## 换成自己的数据

准备匹配的 `cds.fa`、`pep.fa` 和 TAB 分隔的 `pairs.tsv`，三列为 `pair_id`、`seq1`、`seq2`。先明确有依据的同源基因配对，再按[完整中文教程](docs/RAW_CDS_TUTORIAL_ZH.md)批量执行。教程包括 ID 映射、末端终止处理、三碱基缺口、失败排查及 CPU/GPU 命令。基础脚本针对标准遗传密码表 1、阅读框完整且翻译与蛋白一致的数据，不会自动推断同源关系或修复移码。

## Windows 怎么用

Windows 包也附带同样的教程、脚本和原始示例。在 Linux 准备输入后，把**整个** `tutorial_demo/prepared/` 复制到 Windows。Windows rc10 自动处理长输入/输出路径、空格和中文路径，无需修改系统长路径开关。深层安装和安全解压入口见 docs/WINDOWS_PATHS_ZH.md；科学引擎与旧 rc9 相同。在 Windows 解压目录中执行：

```powershell
.\kaksturbo.exe validate --input prepared
.\kaksturbo.exe run --input prepared --output D:/kt_run1 --backend cpu --threads 2
```

想立即试跑 Windows 主程序时，可用现成的 `examples/three_pair_smoke.axt` 作为 `--input`，无需上游比对软件。本教程不宣称验证了 Windows 原生 MAFFT/PAL2NAL 安装，也不要求安装 WSL。

GPU 需兼容的 NVIDIA 驱动、设备和足够显存。先运行 `doctor --backend gpu --probe-gpu`，再显式选 `--backend gpu`；详见[设备说明](docs/DEVICE_GUIDE_ZH.txt)和 [Linux 已验证范围](docs/LINUX_COMPATIBILITY.md)。GPU 不可用时会明确提示原因，不悄悄改用 CPU。
