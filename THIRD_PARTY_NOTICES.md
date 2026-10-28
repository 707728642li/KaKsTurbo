# Third-party notices

KaKsTurbo incorporates/adapts KaKs_Calculator source under GNU GPL version 3. The project license is GPL-3.0-only; original copyright and source notices are retained. Third-party components keep their respective licenses.

## Retained numerical runtimes

Both platforms use ILP64 FP64 OpenBLAS. The previously published Windows portable package uses the NumPy 2.4.6 runtime, and the earlier Linux portable package uses the pinned NumPy OpenBLAS distribution. Their license bundles remain in licenses/Windows-OpenBLAS-bundle and licenses/OpenBLAS-bundle. The 1.1.0 Conda build instead links to Conda-managed OpenBLAS and compiler runtimes; it does not bundle those third-party binaries. The corresponding KaKsTurbo ABI bridge sources are included. No Intel MKL, Intel OpenMP or TBB binary is included in the new Windows engine. Historical Intel notices retained as provenance do not indicate active redistribution.

Portable downloads include CPython and only the runtime libraries needed by their bundled interpreter. See licenses/CPython-runtime and its component inventory, or licenses/recovered-v1 for the Windows runtime notices. Microsoft runtime components retain their separate redistribution terms; no GPL label is applied to them. Windows Expat, OpenSSL, SQLite, zlib, libffi, bzip2, xz and MinGW pthread notices are retained. The operating system and NVIDIA driver are external prerequisites; no NVIDIA driver or CUDA Toolkit redistributable is shipped. Generated PTX is built from the supplied project CUDA source.

The degree-18 matrix-exponential coefficients retain their PyTorch 2.5.1 BSD-style source notice and Bader, Blanes and Casas (2019) provenance. Native corresponding sources, compiler commands and asset manifests are retained. The GPL source archive must accompany binary publication; corresponding-source archives accompany the GitHub binary release. Third-party source URLs and package hashes are recorded in the inventories. This component audit is not a legal opinion or a claim to relicense third-party software.

The historical NVIDIA cuda.h file is not redistributed. A metadata-only adaptation reads its original SHA256 from external-sdk-provenance.json instead of requiring that unused file. Numerical statements, native binaries and PTX are unchanged. The retained PyTorch Reduce.cuh file is covered by its original BSD notice. Linux omits unused readline, curses, NIS and crypt modules and their optional libraries; the active CPython component inventory identifies the included files and notices.
