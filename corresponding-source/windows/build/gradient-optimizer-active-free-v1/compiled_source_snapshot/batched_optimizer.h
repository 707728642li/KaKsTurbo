#ifndef KAKSCUDA_BATCHED_OPTIMIZER_H
#define KAKSCUDA_BATCHED_OPTIMIZER_H
#include <stdint.h>
#ifdef _WIN32
#define KAKS_OPT_API __declspec(dllexport)
#else
#define KAKS_OPT_API
#endif
#ifdef __cplusplus
extern "C" {
#endif
/* Independent reverse-communication, log-coordinate bounded BFGS.
 * All context operations are serial on its creating thread. Evaluation may
 * run elsewhere; no callback, Windows fiber, matrix, or process global state.
 * Immutable config v1: central log-coordinate probes, safeguarded Armijo,
 * projected gradients and an explicit final coordinate improvement test.
 * Values submitted are RAW negative log likelihood; scale is typically L.
 * Status: 0 running, 1 finite-coordinate gate passed, 2 resource cap,
 * 3 rejected objective. Status 1 is not a global-optimality certificate.
 * Completed results' points were evaluated. Running/status0 rows are only
 * diagnostics and must not be consumed as scientific fits. Request tickets
 * must not be reused. Numeric derivative failure is explicit status3. */
KAKS_OPT_API uint32_t kaks_opt_abi_version(void);
KAKS_OPT_API int kaks_opt_create(uint32_t max_jobs,uint32_t max_iterations,
    double gradient_tolerance,double raw_quality_tolerance,void **context,
    char *error,uint32_t capacity);
KAKS_OPT_API int kaks_opt_add(void *context,int64_t slot,uint32_t dimensions,
    const double initial[7],const double lower[7],const double upper[7],
    double objective_scale,char *error,uint32_t capacity);
KAKS_OPT_API int kaks_opt_poll(void *context,uint32_t capacity,int64_t *slots,
    uint64_t *tickets,double *parameters,uint32_t *count,uint32_t *completed,
    char *error,uint32_t error_capacity);
KAKS_OPT_API int kaks_opt_submit(void *context,uint32_t count,const int64_t *slots,
    const uint64_t *tickets,const double *objectives,char *error,uint32_t capacity);
/* result width=18: x7, F, initial_F, last_gradient_projected_per_scale,
 * last_quality_improvement, iterations, evaluations, quality_rounds,
 * status, inverse-Hessian resets, best_observed_F, returned_F-best_observed_F.
 * F/x are the point actually tested by the final gate; the best probe can
 * differ by <=raw_quality_tolerance in a successful result. The last gradient
 * can precede a quality-stage promotion; it is NOT a returned-point KKT
 * certificate. One row per add. */
KAKS_OPT_API int kaks_opt_results(void *context,uint32_t capacity,double *results,
    char *error,uint32_t error_capacity);
KAKS_OPT_API void kaks_opt_free(void *context);
#ifdef __cplusplus
}
#endif
#endif
