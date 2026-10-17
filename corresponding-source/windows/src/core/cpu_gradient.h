#ifndef KAKSCUDA_CPU_GRADIENT_H
#define KAKSCUDA_CPU_GRADIENT_H
#include "shared_core.h"
#include <stddef.h>
#define KAKS_CPU_GRADIENT_ABI_VERSION 1
#define KAKS_CPU_GRADIENT_DIAGNOSTIC_WIDTH 12
/* diagnostics 0..7 match cpu_backend.h; 3 counts all actual GEMMs.
 * 8=forward GEMMs, 9=reverse GEMMs, 10=active raw dimensions, 11=request kind.
 * kind 0 requests F only and writes exactly zero g_log; kind 1 requests F and
 * g_log[j]=raw_x[j]*dF/draw_x[j]. Neither result is divided by codon count L.
 * Inactive gradient entries are exactly zero. StrictPositive/pure rates only. */
#ifdef __cplusplus
extern "C" {
#endif
KAKS_CORE_API uint32_t kaks_cpu_gradient_abi_version(void);
KAKS_CORE_API uint64_t kaks_cpu_gradient_workspace_doubles(uint32_t tape_max_s);
/* workers 1..16, primary_max_s 0..16, tape_max_s primary_max_s..64.
 * Adaptive fallback computes the actual required squarings up to tape_max_s;
 * exceeding either a disabled primary budget or tape budget is an error.
 * No truncation, probability clipping, altered coefficients or fast-math.
 * Each worker owns a persistent arena; MKL thread-local count is one/restored.
 * All calls are synchronous. Overlapping calls on one session are rejected.
 * Close/pair free require all calls/readers completed. */
KAKS_CORE_API int kaks_cpu_gradient_create(const wchar_t *mkl_dll,
    uint32_t workers, uint32_t primary_max_s, uint32_t tape_max_s,
    uint32_t allow_adaptive_fallback, void **session,
    char *error, uint32_t error_capacity);
KAKS_CORE_API void kaks_cpu_gradient_free(void *session);
KAKS_CORE_API int kaks_cpu_gradient_runtime_info(const void *session,
    wchar_t *loaded_path, uint32_t path_capacity,
    char *mkl_version, uint32_t version_capacity, uint32_t *workers,
    uint32_t *tape_max_s, char *error, uint32_t error_capacity);
/* Row slot=pair_index*14+model_id; repeated slots/mixed kinds are allowed.
 * All buffers/const PairData handles remain alive until the call returns.
 * Capacity applies to F[count],g_log[count,7],diagnostics[count,12],codes[count].
 * Invalid metadata/kinds/raw parameters reject before output mutation.
 * Per-row numerical errors return 2, failed rows code=1/F=g_log=NaN, other
 * rows retained code=0. No memoization, fitting, RNG or dropped models. */
KAKS_CORE_API int kaks_cpu_gradient_evaluate_batch(void *session,
    const void *const *pairs, uint32_t pair_count, const int64_t *slots,
    const double *parameters, const uint32_t *kinds, uint64_t count,
    double *objectives, double *gradient_log, double *diagnostics,
    uint32_t *codes, uint64_t capacity, char *error, uint32_t error_capacity);
/* Single point with caller-owned reusable workspace. Optional g_raw receives
 * unnormalized raw-coordinate derivatives, optional transition receives the
 * stationary similarity transform with inactive identity embedding. */
KAKS_CORE_API int kaks_cpu_gradient_evaluate_one(void *session, const void *pair,
    uint32_t model, const double x[7], uint32_t kind,
    double *workspace, uint64_t workspace_size, double *objective,
    double gradient_log[7], double *gradient_raw, double diagnostics[12],
    double *transition61x61, char *error, uint32_t error_capacity);
#ifdef __cplusplus
}
#endif
#endif
