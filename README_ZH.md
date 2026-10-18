<div align="center">

# KaKsTurbo

### 面向 CPU 与 GPU 的高性能模型平均 Ka/Ks 计算

**[下载](https://github.com/707728642li/KaKsTurbo/releases/tag/v1.1.0rc10) · [快速入门](QUICKSTART_ZH.md) · [批量 CDS 教程](docs/RAW_CDS_TUTORIAL_ZH.md) · [English](README.md)**

</div>

KaKsTurbo 面向批量同源基因对的分子进化分析，计算非同义替换率 **Ka**、同义替换率 **Ks** 及 **Ka/Ks**。它将结构化 FP64 矩阵计算、边界约束优化与完整十四模型平均结合，提供共享模型定义和数值策略的 CPU、单 GPU 后端。

软件同时提供模型级诊断、可恢复任务、终端进度和标准 TSV/JSONL 结果。Linux 优先的原始 CDS/蛋白教程可完成 **MAFFT → PAL2NAL → 标准输入 → KaKsTurbo** 的批量流程。

## 计算流程与方法

[![KaKsTurbo 计算流程与数值方法](docs/assets/workflow.png)](docs/assets/workflow.png)

Figure 1 原图保持不变。计算保留全部 61 个有义密码子状态；图中的权重与可行域为示意。独立延长拟合属于验证研究，不是每次常规运行自动执行的步骤。

- **结构化矩阵计算：** 利用生成矩阵及导数结构，保留完整均值速率导数和经过检查的回退路径。
- **边界约束优化：** CPU/GPU 共用有界 BFGS 与 Schur 自由坐标修正，配合线搜索和方向保护。
- **完整模型平均：** 保留全部十四候选，通过 AICc 权重平均参数、重建一个 GTR 过程后再计算 Ka/Ks。
- **可检查的数值状态：** 输出模型权重、边界/封顶标记、投影梯度和 NA 原因。

模型家族及 MA 框架继承已有 KaKs 方法；我们的工作重点是计算实现、优化策略和数值验证，不将其包装为全新的统计模型或显著性检验。[方法说明](docs/METHODS.md)

## 实测性能

[![完整 CLI 耗时与物理核心扩展性](docs/assets/performance.png)](docs/assets/performance.png)

**A：N=3；B/C：N=5 技术重复。** 菱形为中位数，耗时误差条为最小—最大值；KAKS3 为 KaKs_Calculator 3.0 MA。

| 数据与执行环境 | N | 基线耗时 | CPU 耗时 | GPU 耗时 | 中位数耗时之比 |
|:---|---:|---:|---:|---:|:---|
| 公共 1,000 对，同一 Windows 主机 | 3 | 458.04 s | **22.44 s** | **23.06 s** | CPU **20.41×**；GPU **19.86×** |
| 植物 29,028 对，归档 Linux CPU 实验 | 5 | 17,385.79 s | **1,520.30 s** | — | CPU **11.44×** |
| 果蝇 23,571 对，归档 Linux CPU 实验 | 5 | 55,989.67 s | **1,221.29 s** | — | CPU **45.84×** |

以上比较每臂分配 **8 个物理核 / 16 个逻辑处理器**；GPU 臂额外使用一张 GPU。Linux 队列采用独立 CPU 亲和性并发，仍共享主机带宽和存储。另一个固定 10,000 对数据集上，64 物理核相对 2 核为 **22.12×、69.1% 并行效率**。

这些是特定工作负载、版本和执行条件下的完整流程比较，包含与基线不同的估计策略，不能当作普适加速比。主基准中 GPU 略慢于 CPU；未将跨平台 GPU 结果混入同机比较。[全部观测、条件与局限](docs/PERFORMANCE.md)

## 下载与安装

| 平台 | 当前程序版本 | 下载 |
|:---|:---|:---|
| Linux x86-64，glibc ≥2.17 | 1.1.0rc9-tutorial1 | [TAR.GZ](https://github.com/707728642li/KaKsTurbo/releases/download/v1.1.0rc10/KaKsTurbo-1.1.0rc9-tutorial1-docs1-linux-x86_64.tar.gz) |
| Windows x86-64 | 1.1.0rc10 | [ZIP](https://github.com/707728642li/KaKsTurbo/releases/download/v1.1.0rc10/KaKsTurbo-1.1.0rc10-docs1-windows-x86_64.zip) · [长路径解压器](https://github.com/707728642li/KaKsTurbo/releases/download/v1.1.0rc10/Install-KaKsTurbo.ps1) |

`docs1` 是文档打包修订：已包含本首页、中英文教程、方法/性能图和合成结果预览，可在解压目录阅读。Windows/Linux 程序版本和计算文件保持不变，旧包继续保留在同一个 Release。

当前为预发布版。完整解压后即可使用；核心 CPU 计算无需另外安装 Python、Conda 或 CUDA。两个平台的便携包保留各自已验收版本。[PyPI 1.1.0rc10](https://pypi.org/project/kaksturbo/1.1.0rc10/) 已提供 Linux/Windows wheel；[Bioconda 配方](https://github.com/bioconda/bioconda-recipes/pull/70404)已提交，尚待审核和频道上架。[pip 安装及验收说明](docs/PACKAGE_INDEX.md)。[校验值](https://github.com/707728642li/KaKsTurbo/releases/download/v1.1.0rc10/SHA256SUMS-docs1.txt)

## 快速运行

已有密码子比对数据时，在 Linux 解压目录执行：

```bash
./kaksturbo doctor --backend cpu
./kaksturbo run --input examples/three_pair_smoke.axt \
  --output results_demo --backend cpu --threads 8
head -n 4 results_demo/result.tsv
```

用自己的 AXT 替换示例输入即可；FASTA 需要 `--pairs` 指定有向配对。使用新的输出目录。[完整快速入门](QUICKSTART_ZH.md) · [输入输出说明](docs/INPUT_OUTPUT.md)

如果只有原始 CDS/PEP 和同源配对列表，按[批量教程](docs/RAW_CDS_TUTORIAL_ZH.md)先完成比对。主程序不推断同源关系或自动修复移码。

### GPU 与模式选择

`robust` 是默认 CPU/GPU 模式；`reproduce` 为历史 CPU 估计策略，用于兼容性工作，两者数值策略不同。`--backend auto` 当前选择 CPU。显式 GPU 运行前执行：

```bash
nvidia-smi
./kaksturbo doctor --backend gpu --probe-gpu
./kaksturbo run --input examples/three_pair_smoke.axt \
  --output gpu_result --backend gpu --device cuda:0
```

GPU 后端要求 NVIDIA 计算能力 ≥8.6、驱动 CUDA 兼容级别 ≥12.9、至少 8 GiB 空闲显存（含余量）；无需单独安装 CUDA Toolkit。设备、驱动或显存不满足时明确提示原因，不静默切换 CPU。[设备说明](docs/DEVICE_GUIDE_ZH.txt)

## 结果预览

以下是内置合成输入的真实 Windows CPU 结果，选取部分列并显示六位小数。Linux 尾数可能略有不同。

| pair_id | Ka | Ks | Ka_Ks | status |
|:---|---:|---:|---:|:---|
| forward | 0.066008 | 0.065604 | 1.006154 | available |
| reverse | 0.066008 | 0.065604 | 1.006154 | available |
| identical | NA | NA | NA | unidentifiable |

[完整精度 TSV](examples/results/three_pair_cpu.tsv) · [JSONL](examples/results/three_pair_cpu.jsonl) · [质量摘要](examples/results/three_pair_cpu_quality.txt)

完全相同的序列对保留 NA 与原因，不填写虚构的零比值。标准结果还包含序列 ID、密码子数、模型和诊断信息；兼容导出可额外生成 KaKs_Calculator 风格的 22 列表，缺失统计量仍为 NA。文件格式兼容不代表估计器等价。

## 更多文档

- [中文 CLI/API 使用说明](USER_GUIDE_ZH.txt)；[英文 CLI 参考](docs/USAGE.md)。
- [Linux 已验证平台](docs/LINUX_COMPATIBILITY.md)；[Windows 长路径](docs/WINDOWS_PATHS_ZH.md)。
- [数值验证和未解决项](docs/NUMERICAL_VALIDATION.md)；[版本与断点恢复](docs/MIGRATION.md)。
- [兼容导出字段](docs/COMPATIBLE_EXPORT_ZH.txt)；[原始 CDS 批量教程](docs/RAW_CDS_TUTORIAL_ZH.md)。

支持范围为标准遗传密码表 1 下的 MA。有限值、坐标验收、梯度阈值与全局最优是不同概念；Ka/Ks >1 本身不是正选择显著性检验。所有 NA 和未解决诊断均保留。

维护者：**Taishan Li** · **litaishan@caf.ac.cn**。许可：[GPL-3.0-only](LICENSE)。引用请注明软件版本和 Release URL；[CITATION.cff](CITATION.cff) 提供机器可读元数据。问题反馈请通过 [GitHub Issues](https://github.com/707728642li/KaKsTurbo/issues)，使用最小合成示例。
