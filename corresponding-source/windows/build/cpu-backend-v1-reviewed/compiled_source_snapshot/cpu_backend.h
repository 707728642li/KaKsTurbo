#ifndef KAKSCUDA_CPU_BACKEND_H
#define KAKSCUDA_CPU_BACKEND_H
#include "shared_core.h"
#include <stddef.h>
#define KAKS_CPU_ABI_VERSION 1
#define KAKS_CPU_DIAGNOSTIC_WIDTH 8
/* diagnostics: norm1(A), actual squarings, adaptive fallback used, GEMM count,
 * minimum observed joint probability, maximum active transition row-sum error,
 * minimum transition entry, maximum MKL threads observed (must be one).
 * F is the positive negative log likelihood, source-order pattern left fold.
 * This is the new pure-rate / StrictPositive model, not legacy reproduction. */
#ifdef __cplusplus
extern "C" {
#endif
KAKS_CORE_API uint32_t kaks_cpu_abi_version(void);
KAKS_CORE_API uint64_t kaks_cpu_workspace_doubles(void);
/* Loads this absolute MKL runtime path. Workers persist until close; each owns
 * one reusable arena. MKL thread-local count is set to one and then restored.
 * workers: 1..16; primary_max_s: 0..16. allow_adaptive_fallback means compute
 * the actual required squarings beyond primary_max_s, never clamp them.
 * Only synchronous calls are supported; overlapping calls on one session are
 * rejected. Close and pair free require all calls/readers already completed. */
KAKS_CORE_API int kaks_cpu_create(const wchar_t *mkl_dll, uint32_t workers,
    uint32_t primary_max_s, uint32_t allow_adaptive_fallback, void **session,
    char *error, uint32_t error_capacity);
KAKS_CORE_API void kaks_cpu_free(void *session);
KAKS_CORE_API int kaks_cpu_runtime_info(const void *session, wchar_t *loaded_path,
    uint32_t path_capacity, char *mkl_version, uint32_t version_capacity,
    uint32_t *workers, char *error, uint32_t error_capacity);
/* Caller owns and keeps live pair handles, input and output buffers until the
 * call returns. Repeated slots are allowed. Each slot maps pair=slot/14,
 * model=slot%14. Each request has seven contiguous raw doubles. capacity is in
 * rows and applies to F, diagnostics[rows,8] and codes[rows]. Invalid metadata
 * or parameters reject before any output mutation. Per-row numerical errors
 * return 2, preserving other rows with code=0 and failed rows code=1/F=NaN.
 * No objective memoization, fitting, RNG consumption or model dropping occurs. */
KAKS_CORE_API int kaks_cpu_evaluate_batch(void *session,
    const void *const *pairs, uint32_t pair_count, const int64_t *slots,
    const double *parameters, uint64_t count, double *objectives,
    double *diagnostics, uint32_t *codes, uint64_t capacity,
    char *error, uint32_t error_capacity);
/* Single evaluation/debug interface with caller-owned reusable workspace.
 * Optional transition61x61 receives the full stationary similarity transform,
 * inactive states embedded as identity. No probability clipping is performed. */
KAKS_CORE_API int kaks_cpu_evaluate_one(void *session, const void *pair,
    uint32_t model, const double x[7], double *workspace, uint64_t workspace_size,
    double *objective, double diagnostics[8], double *transition61x61,
    char *error, uint32_t error_capacity);
/* MA/state evaluation consumes expanded rates exactly once. pi64 means packed
 * sense61 + three zero padding values, already normalized (tolerance 1e-12).
 * Every observed endpoint must have positive pi. These frequencies are neither
 * reestimated nor replaced by the pair's original frequencies. */
KAKS_CORE_API int kaks_cpu_evaluate_expanded(void *session, const void *pair,
    const double pi64[64], const double rates6[6], double omega, double codon_time,
    double *workspace, uint64_t workspace_size, double *objective,
    double diagnostics[8], double *transition61x61,
    char *error, uint32_t error_capacity);
/* Independent general 61x61 FP64 exponential, for primitive certification.
 * diagnostics[0:4] have the same meaning; [4:7] are NaN. Matrix and output may
 * alias, workspace may not alias either. All input entries must be finite. */
KAKS_CORE_API int kaks_cpu_matrix_exp(void *session, const double *matrix61x61,
    double *output61x61, double *workspace, uint64_t workspace_size,
    double diagnostics[8], char *error, uint32_t error_capacity);
#ifdef __cplusplus
}
#endif
#endif
