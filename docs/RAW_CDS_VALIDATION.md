# Raw CDS tutorial qualification — 2026-10-18

The tutorial was exercised primarily on **native Ubuntu 22.04.5 LTS, x86_64, glibc 2.35**, using real external tools and the installed KaKsTurbo 1.1.0rc9 Linux executable. This is a small synthetic workflow check, not a performance comparison or biological-accuracy study.

| Check | Observed result |
|---|---|
| Input and transformation unit tests | 14/14 passed |
| Real MAFFT 7.526 + Bioconda PAL2NAL 14.1 | Four directed requests completed |
| GNU parallel 20260722, two jobs versus serial | AXT and metadata SHA256 identical; input order retained |
| Installed Linux `prepare` / `validate` / CPU `run --threads 2` | All four requests retained; three finite Ka/Ks, one NA |
| Original CDS/PEP and pair table | SHA256 unchanged |
| Wrong CDS/PEP translation | Batch rejected; no final AXT |
| External tool exits zero but returns invalid alignment | All four failures reported; no final AXT; work retained |
| Tampered pair fingerprint | Installed CLI rejects the metadata |
| Success cleanup | Compressed logs/alignment provenance retained; duplicate work removed |
| Windows rc9 input-only check | Linux prepared directory and AXT + sidecar both accepted; no new Windows fitting |

The identical-sequence control remains NA. The tiny example contains 56 model slots: 42 finite saved endpoints and 14 slots with unavailable diagnostics from the NA pair. The saved diagnostics flag all 42 finite endpoints at parameter bounds; all 42 have projected gradient at or below 1e-8 and none hit the iteration cap. These observations describe this fixture only and do not certify global optima or other datasets.

The fixture covers original ID mapping, a reverse request, codon gaps and lowercase CDS. It uses a strict genetic-code-1, frame-intact input policy. It does not establish support for ambiguous raw DNA, frameshifts, alternate codes, arbitrary transcript mismatches or every MAFFT/PAL2NAL release. No GPU fit was added for this tutorial. Existing scientific source and all five original rc9 download archives retain their SHA256 values.

MAFFT/PAL2NAL/Python were installed in the project preparation environment; an already-installed GNU parallel was reused through an explicit path. The environment recipe can install these optional tools together on a new system. No alignment program is bundled into the KaKsTurbo binary runtime.

Machine-readable summary: [RAW_CDS_VALIDATION.json](RAW_CDS_VALIDATION.json). Tutorials: [English](RAW_CDS_TUTORIAL.md), [中文](RAW_CDS_TUTORIAL_ZH.md). The published summary records tool versions and passed/failed checks; raw local execution logs are not part of the software download.

该流程已在 Linux 使用真实软件和现有安装包跑通；这份记录是上游输入准备的验收，不增加论文性能或生物学准确性主张。原始结果、原发布归档及科学标准保持不变。
