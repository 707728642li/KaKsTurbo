#ifndef KAKS_CUDA_NATIVE_ADAPTER_H
#define KAKS_CUDA_NATIVE_ADAPTER_H

#include <stdint.h>
#ifdef _WIN32
#define KAKS_ADAPTER_API __declspec(dllexport)
#else
#define KAKS_ADAPTER_API
#endif

#define KAKS_ADAPTER_ABI_VERSION 1
#define KAKS_ADAPTER_MODEL_COUNT 14
#define KAKS_ADAPTER_META_WIDTH 4
#define KAKS_ADAPTER_VALUE_WIDTH 66
#define KAKS_ADAPTER_RESULT_WIDTH 96
#define KAKS_ADAPTER_STATUS_WIDTH 4

enum KaksAdapterStatus {
    KAKS_ADAPTER_OK = 0,
    KAKS_ADAPTER_INVALID_ARGUMENT = 1,
    KAKS_ADAPTER_INVALID_INPUT = 2,
    KAKS_ADAPTER_NO_USABLE_CODONS = 3,
    KAKS_ADAPTER_BUFFER_TOO_SMALL = 4,
    KAKS_ADAPTER_MISSING_MODEL = 5,
    KAKS_ADAPTER_GPU_MODEL_FAILED = 6,
    KAKS_ADAPTER_NONFINITE_MODEL = 7,
    KAKS_ADAPTER_INTERNAL_ERROR = 8
};

#ifdef __cplusplus
extern "C" {
#endif

KAKS_ADAPTER_API uint32_t kaks_adapter_abi_version(void);
KAKS_ADAPTER_API const char *kaks_adapter_schema(void);
KAKS_ADAPTER_API const char *kaks_adapter_model_name(uint32_t model_index);

/* seq_a/seq_b are concatenated AXT sequences, without line breaks. */
KAKS_ADAPTER_API int kaks_prepare_pair(
    const char *pair_id, const char *seq_a, const char *seq_b, void **out_context,
    char *error, uint32_t error_capacity);

KAKS_ADAPTER_API int kaks_get_pair_sizes(
    void *context, uint32_t *raw_nt_length, uint32_t *cleaned_nt_length,
    uint32_t *pattern_count, char *error, uint32_t error_capacity);

/*
 * meta={pattern_offset, npatt, cleaned_codon_count, 0};
 * values={cleaned_nt_length, SNP, pi[0..60], 0,0,0}.
 * z0/z1/weights receive this pair's triangular patterns; write them to global
 * arrays at pattern_offset. Pattern arrays have capacity in elements.
 * GC is computed from the original concatenated sequences before masking.
 */
KAKS_ADAPTER_API int kaks_export_pair(
    void *context, int32_t pattern_offset, int32_t meta[4], double values[66],
    int8_t *z0, int8_t *z1, double *weights, uint32_t pattern_capacity,
    double gc[4], char *error, uint32_t error_capacity);

/*
 * results[14][96]: indices 0=lnL,1=AICc,2=3*(x0/3) as original MLResult.t,
 * 3=omega,4=Ka,5=Ks,
 * 6=S,7=N,8=Sd,9=Nd,10=GY.t,11..16=rate6,17..80=pi64.
 * statuses[14][4]: full_job_id, ming2_result, Iround, eigenfail_count.
 * Host must initialize statuses to -1 before GPU dispatch. ming2_result 0/1
 * is diagnostic, not a success/failure flag in the upstream optimizer.
 * This source version's ming2 returns only 0/1; all other return codes fail
 * finalization. -1 identifies an unfinished GPU fit, and -2 an unfitted
 * evaluation. An iteration limit or an eigen-status warning remains eligible
 * with a 0/1 return; callers report these diagnostics separately.
 * Every model must be present; no CPU fitting fallback exists.
 * output is one six-significant-digit MA TSV row, ending in LF, without header.
 * required_capacity includes NUL. CPU timing covers injected MS and original
 * MA postprocessing, including its final likelihood/eigen and Fisher calls.
 */
KAKS_ADAPTER_API int kaks_finalize_pair(
    void *context, const double *results, uint32_t model_count,
    const int32_t *statuses, char *output, uint32_t output_capacity,
    uint32_t *required_capacity, double *cpu_postprocess_seconds,
    char *error, uint32_t error_capacity);

/* One original CPU likelihood/eigen/statistics evaluation, without fitting.
 * status={model_index,-2,0,eigen_warning_count}; output follows the GPU ABI.
 * This is for primitive parity, not a fitted MA candidate. */
KAKS_ADAPTER_API int kaks_evaluate_pair(
    void *context, uint32_t model_index, const double x[7], double results[96],
    int32_t status[4], char *error, uint32_t error_capacity);

/* Last finalization metadata: legacy negative ming2 count (always zero after
 * valid finalization), iteration-limit count, GPU eigen warning count,
 * CPU MA eigen warning count. Reporting only. */
KAKS_ADAPTER_API int kaks_get_postprocess_diagnostics(
    void *context, int32_t diagnostics[4], char *error, uint32_t error_capacity);

/* Caller must not free a context while another call is using it. */
KAKS_ADAPTER_API void kaks_free_pair(void *context);

#ifdef __cplusplus
}
#endif
#endif
