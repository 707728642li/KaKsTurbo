#ifndef KAKSCUDA_STATISTICS_H
#define KAKSCUDA_STATISTICS_H
#include "shared_core.h"
#include <stdint.h>
#ifdef _WIN32
#define KAKS_STAT_API __declspec(dllexport)
#else
#define KAKS_STAT_API
#endif
#define KAKS_STAT_ABI_VERSION 1
#define KAKS_STAT_VALUE_COUNT 21

enum KaksStatStatus { KAKS_STAT_AVAILABLE=0, KAKS_STAT_UNIDENTIFIABLE=1,
    KAKS_STAT_UNAVAILABLE=2 };
enum KaksStatFlag {
    KAKS_STAT_ZERO_OBSERVED=1u, KAKS_STAT_ZERO_SCALE=2u,
    KAKS_STAT_NO_SYNONYMOUS_FLOW=4u, KAKS_STAT_NO_NONSYNONYMOUS_FLOW=8u,
    KAKS_STAT_SHORT_AICC=16u, KAKS_STAT_SCORE_UNAVAILABLE=32u,
    KAKS_STAT_MA_AICC_UNDEFINED=64u, KAKS_STAT_OUTSIDE_SOURCE_BOX=128u,
    KAKS_STAT_FIXED_PARAMETER_ONLY=256u, KAKS_STAT_ZERO_TIME=512u
};
enum KaksMaFlag {
    KAKS_MA_INCOMPLETE=1u, KAKS_MA_CANDIDATE_UNAVAILABLE=2u,
    KAKS_MA_SCORE_PENDING=4u, KAKS_MA_RATE_MASS_DEFAULTED=8u,
    KAKS_MA_FIXED_PARAMETER_ONLY=16u, KAKS_MA_PROCESS_UNAVAILABLE=32u
};
/* values = lnL,AICc,x0,x0/3,omega,rs,ra0,ra,mr,S,N,Ks,Ka,Ka/Ks,
 * expected_S_events,expected_N_events,observed_rescaled_S,observed_rescaled_N,
 * codon_count,observed_SNP,k. Unavailable mathematical values are NaN.
 * math_mask marks mathematically computed fields; report_mask additionally
 * suppresses unidentifiable substitution-derived fields for observed SNP=0.
 * Parameters/sites/weights remain conditional diagnostics, not identified
 * estimates for zero inputs. Raw supplied scores are NOT recomputed here.
 * pair_fingerprint detects ordinary cross-pair mixups, not adversarial input.
 * No result from this API establishes fitting/convergence or quality. */
typedef struct KaksModelStatistics {
    uint32_t model, support_policy, status, flags;
    uint64_t pair_fingerprint, math_mask, report_mask;
    double values[KAKS_STAT_VALUE_COUNT], rates[6], pi[64], x[7];
} KaksModelStatistics;
typedef struct KaksMaStatistics {
    uint32_t status, flags, present_mask, unavailable_mask;
    int32_t best_model;
    uint32_t rate_default_mask;
    double weights[14], rate_weight_masses[5];
    KaksModelStatistics averaged;
} KaksMaStatistics;

#ifdef __cplusplus
extern "C" {
#endif
KAKS_STAT_API uint32_t kaks_statistics_abi_version(void);
KAKS_STAT_API uint32_t kaks_statistics_model_size(void);
KAKS_STAT_API uint32_t kaks_statistics_ma_size(void);
/* Return code reports argument/ABI errors; result.status reports scientific
 * unavailability. A valid finite score is supplied lnL (not positive F).
 * If score_available=0, lnL/AICc are NaN and unavailable. If mr=0 the supplied
 * score is deliberately discarded: this normalized process is undefined.
 * fit-box violations are flagged, not silently clamped by fixed statistics. */
KAKS_STAT_API int kaks_statistics_fixed(const void *prepared_pair,uint32_t model,
    uint32_t support_policy,const double x[7],uint32_t score_available,double lnL,
    KaksModelStatistics *output,char *error,uint32_t error_capacity);
/* Requires all fourteen unique models from this pair and support policy with
 * every conditional plug-in AICc available; otherwise structured unavailable
 * output, no subset renormalization. Model input order may vary.
 * Pure model parameters/frequencies are revalidated against the prepared pair.
 * Historical conditional free-rate masks and <1e-50 fallback-to-one are
 * explicit, with default mask. Averaged parameters form a GTR ModelState.
 * MA score starts unavailable: never substitute a weighted model lnL. */
KAKS_STAT_API int kaks_statistics_average(const void *prepared_pair,
    const KaksModelStatistics *models,uint32_t model_count,
    KaksMaStatistics *output,char *error,uint32_t error_capacity);
/* Build averaged Q/S/A without expanding already expanded rates or replacing
 * averaged pi with the original prepared frequencies. */
KAKS_STAT_API int kaks_statistics_ma_q(const void *prepared_pair,
    const KaksMaStatistics *averaged,double *output,uint32_t capacity,
    char *error,uint32_t error_capacity);
/* After an independent same-state likelihood evaluation, attach its lnL.
 * MA AICc stays unavailable because its parameter count is not defined. */
KAKS_STAT_API int kaks_statistics_ma_score(const void *prepared_pair,
    KaksMaStatistics *averaged,double lnL,char *error,uint32_t error_capacity);
#ifdef __cplusplus
}
#endif
#endif
