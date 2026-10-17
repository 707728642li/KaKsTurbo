#ifndef KAKSCUDA_SHARED_CORE_H
#define KAKSCUDA_SHARED_CORE_H
#include <stdint.h>
#ifdef _WIN32
#define KAKS_CORE_API __declspec(dllexport)
#else
#define KAKS_CORE_API
#endif

/* Version 1 is prepared-data and matrix construction only, never fitting.
 * Preparation inputs come from the verified original native adapter. */
#define KAKS_CORE_ABI_VERSION 1
#define KAKS_CORE_MODEL_COUNT 14
#define KAKS_CORE_NCODE 61
#define KAKS_CORE_Q_SIZE 3721
#define KAKS_CORE_Q_OFFSET 0
#define KAKS_CORE_S_OFFSET 3721
#define KAKS_CORE_A_OFFSET 7442
#define KAKS_CORE_PI_OFFSET 11163
#define KAKS_CORE_SQRT_PI_OFFSET 11227
#define KAKS_CORE_RATES_OFFSET 11288
/* scalars: rs, ra0, ra, mr, omega, raw kappa, x0, active_count, np,
 * support_policy, support_cutoff */
#define KAKS_CORE_SCALARS_OFFSET 11294
#define KAKS_CORE_SNAPSHOT_WIDTH 11305
/* Legacy mode is for upstream snapshot certification only. Its positive
 * discarded frequencies can leave the retained transition substochastic.
 * Publication/new-model mode retains every strictly positive frequency. */
#define KAKS_CORE_LEGACY_CUTOFF 0
#define KAKS_CORE_STRICT_POSITIVE 1

#ifdef __cplusplus
extern "C" {
#endif
KAKS_CORE_API uint32_t kaks_core_abi_version(void);
KAKS_CORE_API uint32_t kaks_core_edge_count(void);
/* meta4/values66/GC and triangular patterns follow kaks_adapter.h exactly.
 * Arrays start at zero locally even when meta[0] is a nonzero global offset.
 * The core owns a copy. After creation inputs may be released or modified.
 * Concurrent reads/builds on a live pair are safe with distinct output buffers.
 * Free must occur only after all reads/builds have completed. No stale-pointer
 * registry, process globals, hidden matrix allocations or locks are used. */
KAKS_CORE_API int kaks_core_create_prepared_pair(const char *pair_id,
    const int32_t meta[4], const double values[66], const int8_t *z0,
    const int8_t *z1, const double *weights, uint32_t count,
    const double gc[4], void **pair, char *error, uint32_t error_capacity);
KAKS_CORE_API int kaks_core_export_prepared_pair(const void *pair,
    int32_t meta[4], double values[66], int8_t *z0, int8_t *z1,
    double *weights, uint32_t capacity, double gc[4],
    char *error, uint32_t error_capacity);
KAKS_CORE_API void kaks_core_free_prepared_pair(void *pair);
/* Allocation-free original-order Q construction. No eigensolve/optimizer.
 * Full Q determines mr before explicit support masking. S embeds the original
 * active lower triangle and its mirror in 61 dimensions, inactive rows zero.
 * A=(S*x0)/mr; exp(A) must embed inactive identity afterwards, as in robust.
 * Expanded rates are a pure raw-x mapping matching one original parse pass.
 * The new model's objective/stats must reuse that same expanded rate object.
 * It does not reproduce the legacy repeated in-place rate transformations.
 * The frequencies are the original prepared hybrid empirical/F3x4 estimate;
 * this interface does not normalize or reestimate them. pi64 is packed sense61
 * plus padding, not the original 64-codon ID order. StrictPositive's tail3 is
 * zero; LegacyCutoff mirrors the unused source tail for equal-frequency models.
 * Inactive identity after exp is an embedding convention for stationary
 * likelihoods, not the full-Q transition from a zero-frequency start state. */
KAKS_CORE_API int kaks_core_q_snapshot(const void *pair, uint32_t model,
    uint32_t support_policy, const double x[7], double *output, uint32_t capacity,
    char *error, uint32_t error_capacity);
#ifdef __cplusplus
}
#endif
#endif
