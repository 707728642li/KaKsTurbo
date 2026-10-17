#ifndef KAKS_FIBER_OPTIMIZER_H
#define KAKS_FIBER_OPTIMIZER_H
#include <stdint.h>
#define KAKS_FIBER_API __declspec(dllexport)
#define KAKS_FIBER_ABI_VERSION 1
#define KAKS_FIBER_REQUEST_META_WIDTH 6
#define KAKS_FIBER_RESULT_WIDTH 96
#define KAKS_FIBER_STATUS_WIDTH 4
#ifdef __cplusplus
extern "C" {
#endif
KAKS_FIBER_API uint32_t kaks_fiber_abi_version(void);
/* Single session, single owning Windows OS thread. No GPU work occurs here.
 * Defaults in the Python wrapper: 64 exact-cache entries, 128KiB reserved and
 * 4KiB initially committed stack, at most 16 pairs. max_pairs may be 1..1000.
 * Windows may round/grow stack allocations. GY94 objects/workspace use heap.
 * The attempted 64KiB reserve overflowed during GCC cancellation unwinding;
 * reserves below the verified 128KiB minimum are rejected. */
KAKS_FIBER_API int kaks_fiber_create(uint32_t cache_entries,
    uint32_t stack_reserve, uint32_t stack_commit, uint32_t max_pairs,
    void **session, char *error, uint32_t error_capacity);
/* Raw aligned codon sequences; uses the original paired codon mask/GC order.
 * Add all pairs before the first poll. Each pair owns all 14 canonical models. */
KAKS_FIBER_API int kaks_fiber_add_pair(void *session, const char *pair_id,
    const char *seq_a, const char *seq_b, uint32_t *pair_index,
    char *error, uint32_t error_capacity);
/* Runs ready fibers until their next uncached objective request or completion.
 * meta rows={slot,pair_index,model_index,np,Iround,value_is_negative_lnL=1}.
 * x rows have 7 doubles. Only the first np are active. tickets prevent stale
 * submissions. Pending requests may be returned again until supplied.
 * The original lfun returns the minimization objective -lnL (usually positive).
 * Torch evaluator output[0] is lnL (usually negative), so negate it on submit. */
KAKS_FIBER_API int kaks_fiber_poll(void *session, uint32_t capacity,
    int32_t *meta, double *x, uint64_t *tickets, uint32_t *request_count,
    uint32_t *completed_count, char *error, uint32_t error_capacity);
/* Atomic validation before accepting any row. objectives are -lnL, never lnL.
 * eigen_warning_delta is optional; library backends with different diagnostics
 * must not claim that zero proves equivalence to original Householder warnings. */
KAKS_FIBER_API int kaks_fiber_submit(void *session, uint32_t count,
    const int32_t *slots, const uint64_t *tickets, const double *objectives,
    const int32_t *eigen_warning_delta, char *error, uint32_t error_capacity);
/* Original CPU lfun on each pending request; test provider, without fitting. */
KAKS_FIBER_API int kaks_fiber_mock_evaluate(void *session, uint32_t count,
    const int32_t *slots, const uint64_t *tickets, double *objectives,
    int32_t *eigen_warning_delta, char *error, uint32_t error_capacity);
/* Completed models only. raw96/status4 match the existing adapter ABI and can
 * be passed to its finalizer in pair/model order. No Fisher/MA runs here. */
KAKS_FIBER_API int kaks_fiber_results(void *session, uint32_t task_capacity,
    double *results, int32_t *statuses, char *error, uint32_t error_capacity);
/* Short uncached direct original optimizer oracle, using the same source class
 * and heap workspace but no fiber switches. Do not use as large CPU fallback. */
KAKS_FIBER_API int kaks_fiber_cpu_oracle(void *session, uint32_t pair_index,
    uint32_t model_index, double result[96], int32_t status[4],
    char *error, uint32_t error_capacity);
/* Per-task uint64 rows: requests, cache_hits, valid_cache_entries, completed,
 * Iround, eigen_warnings. Buffer capacity is in tasks. */
KAKS_FIBER_API int kaks_fiber_metrics(void *session, uint32_t task_capacity,
    uint64_t *metrics, char *error, uint32_t error_capacity);
/* Cancels outstanding fibers by resuming their suspended hook with an
 * exception, allowing scratch cleanup; then deletes fibers and heap objects. */
KAKS_FIBER_API int kaks_fiber_close(void *session, char *error, uint32_t error_capacity);
#ifdef __cplusplus
}
#endif
#endif
