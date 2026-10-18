# 从原始 CDS / PEP 到 KaKsTurbo：Linux 快速教程

[English](RAW_CDS_TUTORIAL.md) · [输入输出说明](INPUT_OUTPUT.md)

这条流程把**已确认要比较的同源基因对**整理为密码子比对，再计算 Ka、Ks 和 Ka/Ks：

```text
原始 CDS + 对应蛋白 PEP + 显式配对表
    → 检查 CDS 翻译与蛋白一致
    → 每对蛋白用 MAFFT 比对
    → PAL2NAL 回译为密码子比对
    → KaKsTurbo prepare / validate
    → KaKsTurbo run → result.tsv / results.jsonl / 质量汇总
```

主程序的 `prepare` 负责规范化**已比对**输入；原始序列比对由随教程提供的独立脚本 [`prepare_cds_pairs.py`](../scripts/prepare_cds_pairs.py) 调用外部软件完成。它不会修改 KaKsTurbo 的估计方法。下文以 Linux 为主；已生成的标准输入也可以在 Windows 使用。

## 1. 准备三个文件

| 文件 | 内容 | 要求 |
|---|---|---|
| `cds.fa` | 未比对的 CDS，多条 FASTA | 编码方向 5′→3′、不含内含子/UTR、正确阅读框 |
| `pep.fa` | 同一注释版本、同一转录本对应的蛋白 | 与 CDS 使用相同 ID；FASTA 标题的第一个空白前字段作为 ID |
| `pairs.tsv` | 明确要比较的基因对 | 三列 TAB 分隔：`pair_id`、`seq1`、`seq2` |

例如配对表是：

```text
pair_id	seq1	seq2
pair_A_B	spA_tx1	spB_tx1
pair_B_A	spB_tx1	spA_tx1
pair_C_D	spA_tx2	spB_tx2
```

配对表中必须是真正的制表符，不是连续空格。`pair_id` 唯一；请求顺序和反向配对保留。同一个转录本可以出现在多个请求中。多个物种的 FASTA 可以合并，但 ID 必须全局唯一；若添加物种前缀，要同时更新 CDS、PEP 和配对表。若蛋白 ID 与转录本 ID 不同，先按注释映射，不能靠文件顺序猜对应关系。

配对关系应来自已确认的直系同源、共线性或有依据的候选基因分析。本脚本不推断同源性，也不将整个蛋白组中的不同基因家族混在一起比对。同一基因的不同异构体不能随意拼配；先选定对应转录本。

基础流程采用**标准遗传密码表 1**，要求 CDS 为 A/C/G/T，蛋白为 20 种常规氨基酸，不支持含移码、内部终止、X 或其他模糊符号的原始输入。它会报告这些情况，保留所有请求的失败记录，等待用户检查注释；不会自动删掉有问题的配对。这是本教程的输入约束，主程序对已经比对输入的符号/过滤能力见输入输出说明。

## 2. 一次安装可选的比对环境

在 KaKsTurbo **便携包解压目录或源码目录**执行。已有合适环境时直接复用，无需重复安装。环境仅用于上游比对，不替换便携包自带运行环境。

```bash
conda env create -p "$PWD/envs/input-prep" \
  -f workflow/raw_cds_environment.yml

conda run -p "$PWD/envs/input-prep" --no-capture-output mafft --version
conda run -p "$PWD/envs/input-prep" --no-capture-output pal2nal.pl -h
conda run -p "$PWD/envs/input-prep" --no-capture-output parallel --version
```

教程已实测 MAFFT 7.526、Bioconda 的 PAL2NAL 14.1、GNU parallel 20260722。PAL2NAL 上游现另有新版；此配方固定经过本次验证的 Perl 版接口，不假定新版命令完全相同。比对工具由用户安装，未捆绑进 KaKsTurbo 二进制包。

## 3. 先跑随包小示例

[`examples/raw_cds_demo`](../examples/raw_cds_demo) 是**人工构造的格式示例**，含 4 个有向请求：两个不同家族中的配对、一个反向请求和一个相同序列对；用于流程验收，不是生物学或性能基准。

以下命令在 Linux 便携包解压目录执行，`KT="$PWD/kaksturbo"` 选择包内程序。在源码目录使用时，把它改为已安装可执行文件的绝对路径。每次使用新的输出目录。[快速入门](../QUICKSTART_ZH.md)还提供一条命令跑完整示例的脚本。

```bash
set -euo pipefail
PREP_ENV="$PWD/envs/input-prep"
KT="$PWD/kaksturbo"

# 每个比对任务单线程，最多同时处理 2 对。
conda run -p "$PREP_ENV" --no-capture-output python scripts/prepare_cds_pairs.py \
  --cds examples/raw_cds_demo/cds.fa \
  --pep examples/raw_cds_demo/pep.fa \
  --pairs examples/raw_cds_demo/pairs.tsv \
  --output tutorial_alignment --jobs 2 --strip-terminal-stop

# 仅当上一步返回 0、alignment_manifest.json 中 complete=true 时继续。
"$KT" prepare --input tutorial_alignment/aligned_pairs.axt \
  --metadata tutorial_alignment/aligned_pairs.metadata.jsonl --output tutorial_prepared
"$KT" validate --input tutorial_prepared
"$KT" run --input tutorial_prepared --output tutorial_results --backend cpu --threads 2
```

示例 CDS 含末端终止密码子，PEP 含末尾 `*`，因此明确使用 `--strip-terminal-stop`。该选项只允许去掉**一个末端终止密码子及对应的可选蛋白终止标记**，每条处理均记入来源记录；内部终止或翻译不匹配仍报错。没有末端终止的真实输入不必加该选项。原始文件始终不改写，CDS 的大小写保留。

本次 Linux 实测这 4 个请求全部读入并完成 14 模型计算，主结果 3 个有限 Ka/Ks、1 个 NA。相同序列对中的缺失估计保留为 NA，不用零代替。其他数据的质量标记、NA 数量和耗时不由这个示例保证。

## 4. 换成自己的数据，批量处理

只需把输入路径换成自己的文件，并按可用算力调整 `--jobs`。下面最多 8 个独立基因对并发，每个 MAFFT 只使用 1 个线程；GNU parallel 错峰启动任务。不要把“并发配对数”和 KaKsTurbo 的 CPU 工作线程数相乘或混为一谈，两阶段顺序执行。

```bash
set -euo pipefail
PREP_ENV="$PWD/envs/input-prep"
KT="$PWD/kaksturbo"

conda run -p "$PREP_ENV" --no-capture-output python scripts/prepare_cds_pairs.py \
  --cds data/cds.fa --pep data/pep.fa --pairs data/pairs.tsv \
  --output alignment_batch --jobs 8 --strip-terminal-stop

"$KT" prepare --input alignment_batch/aligned_pairs.axt \
  --metadata alignment_batch/aligned_pairs.metadata.jsonl --output prepared_batch
"$KT" validate --input prepared_batch
"$KT" run --input prepared_batch --output results_cpu --backend cpu --threads 8
```

脚本显示真实完成数量、吞吐、耗时及 ETA。`--jobs 1` 可用于单个配对，不需要 GNU parallel。`--timeout 300` 是每对、每个外部比对程序的超时秒数，不是 KaKsTurbo 的拟合迭代数。`--keep-work` 可保留逐对中间文件用于排查；默认成功后合并保留日志和蛋白比对、删除重复中间文件。失败时保留中间文件，不导出部分成功的最终 AXT。修复后使用新输出目录重跑，避免覆盖既有结果。

若希望用 GPU，只替换最后的计算步骤：

```bash
nvidia-smi
"$KT" doctor --backend gpu --probe-gpu
"$KT" run --input prepared_batch --output results_gpu --backend gpu --device cuda:0
```

这里每个 KaKsTurbo 任务使用一张卡。GPU 需要兼容的 NVIDIA 驱动，预编译包不需要额外 CUDA Toolkit；不可用时明确报错，CPU 仍可单独使用。完整设备要求见安装文档。该教程的新增端到端验收采用 Linux CPU，没有声称重跑 GPU 验收。

## 5. 脚本具体做了什么

对于每一个明确请求，脚本给两条序列分配临时内部名 `S1/S2`，避免特殊 ID 影响外部工具，再运行：

```bash
mafft --amino --auto --thread 1 --inputorder pair.pep.fa > pair.pep.aln.fa
pal2nal.pl pair.pep.aln.fa pair.cds.fa -output fasta -codontable 1 > pair.codon.aln.fa
```

PAL2NAL 输入的蛋白与 CDS 名称和顺序均一致。脚本先验证完整 CDS 翻译，再核对蛋白比对没有丢失残基、PAL2NAL 输出逐密码子覆盖完整处理后 CDS。**不使用 `-nogap` 或 `-nomismatch`**；三碱基缺口保留，最终由 KaKsTurbo 的既定配对密码子过滤规则处理。流程不会根据比对得分、序列长度或最终 Ka/Ks 选择性丢弃请求。

这里使用逐对蛋白比对，适合独立的两序列 Ka/Ks 请求。如果研究需要多物种共享同一比对坐标，应先在每个同源组内建立统一多序列密码子比对，再按已确认配对提取输入；两种分析设计应清楚区分。

## 6. 得到了哪些文件

| 文件 | 用途 |
|---|---|
| `aligned_pairs.axt` | 主程序可读取的三行一组 KaKs AXT |
| `aligned_pairs.metadata.jsonl` | 原始两条序列 ID、方向、顺序和序列指纹，随 AXT 一起使用 |
| `alignment_manifest.json` | 成功状态、原始文件哈希、软件路径/版本和处理策略 |
| `qc.tsv` | 每对比对长度、含缺口密码子数、共同可用密码子数、蛋白一致性比例 |
| `errors.tsv` | 失败配对及原因；完全成功时只有表头 |
| `pair_provenance.jsonl` | 每对命令、处理时间、末端终止去除等明确变换 |
| `protein_alignments.jsonl.gz`、`tool_logs.jsonl.gz` | 压缩的蛋白比对和外部程序日志，供回看 |
| `parallel.tsv` | 并行任务日志，`--jobs > 1` 时生成 |

AXT 示例结构（不带 FASTA 的 `>`，各组间空行）：

```text
pair_A_B
ATGGCT---TTT
ATGGCCGGATTC

```

这是 KaKs 三行 AXT，不是含染色体坐标的 UCSC AXT。`prepare` 会进一步生成带哈希和元数据绑定的标准 `prepared_batch/`，建议完整保存或移动这个目录。

最终查看 `results_cpu/result.tsv`、`results.jsonl` 和 `quality_summary.txt`。NA 不等于 0，有限 Ka/Ks 不代表充分收敛，Ka/Ks > 1 本身也不是显著正选择检验。`qc.tsv` 可以帮助挑选需要人工检查的比对；脚本不会自动凭一个一致性阈值删除它们。

## 7. 常见问题

| 提示 | 应该检查什么 |
|---|---|
| `ID missing` / `Duplicate FASTA ID` | ID 是否全局唯一，CDS 与 PEP 是否来自相同转录本，配对表名称是否准确 |
| `CDS translation != peptide` | 注释版本、转录本映射、阅读框、CDS 方向以及是否误用了带 UTR/内含子的序列 |
| `length ... multiple of 3` / `internal stop` | 截断、移码、错误 CDS 提取；不要靠删碱基或换密码表“修到通过” |
| `terminal stop found` | 确认是合法末端终止后，显式加 `--strip-terminal-stop` |
| 模糊碱基或 `X` | 回到原始注释/QC；本基础脚本不自动掩蔽、修复或抛弃这些请求 |
| 外部程序报错或输出为空 | 查看 `tool_logs.jsonl.gz`、`errors.tsv` 和保留的 `.work/`；返回码 0 但无有效比对也会被拒绝 |
| prepared 元数据/哈希错误 | 保持 AXT 与配套元数据、manifest 一起移动；不要手动修改其中一份 |

在 Windows 计算时，可先在 Linux 完成准备，把**整个** `prepared_batch/` 复制到 Windows。Windows rc10 自动处理长输入/输出路径、空格和中文路径，无需修改系统长路径开关。深层安装和安全解压入口见 WINDOWS_PATHS_ZH.md；科学引擎与旧 rc9 相同。下面盘符按实际情况调整，并使用新输出目录：

```powershell
.\kaksturbo.exe validate --input prepared_batch
.\kaksturbo.exe run --input prepared_batch --output D:/kt_run1 --backend cpu --threads 8
```

## 方法来源与验证范围

- [MAFFT 官方说明](https://mafft.cbrc.jp/alignment/software/)及[线程设置](https://mafft.cbrc.jp/alignment/software/multithreading.html)：蛋白比对；本配方每任务单线程。
- [PAL2NAL 官方页面](https://www.bork.embl.de/pal2nal/)与 [Suyama et al. (2006)](https://doi.org/10.1093/nar/gkl315)：蛋白比对到密码子比对的转换。上述严格一致性检查由本教程额外执行。
- [GNU parallel](https://www.gnu.org/software/parallel/)：独立配对的批量并发；使用时请按其引用要求注明软件。

准备脚本保持计算引擎、14 模型、FP64 和数值阈值不变。公开的[流程验证记录](RAW_CDS_VALIDATION.md)与[机器可读摘要](RAW_CDS_VALIDATION.json)说明已测试的平台、输入与范围；这些检查用于验证教程，不代表性能比较或生物学准确性。

本次验收的具体平台、通过项与局限见 [验证记录](RAW_CDS_VALIDATION.md)。
