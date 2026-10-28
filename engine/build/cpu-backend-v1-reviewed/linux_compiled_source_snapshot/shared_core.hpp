#ifndef KAKSCUDA_SHARED_CORE_HPP
#define KAKSCUDA_SHARED_CORE_HPP
#include "shared_core.h"
#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace kakscuda {
struct Pattern { int8_t z0, z1; double weight; };
/* Construct once, then share const PairData across all fourteen models and
 * CPU workers. Sequence cleaning and GC are deliberately upstream in v1. */
class PairData final {
public:
    PairData(const char *id, const int32_t *meta, const double *values,
        const int8_t *z0, const int8_t *z1, const double *weight,
        uint32_t count, const double *gc);
    const std::string &id() const noexcept { return id_; }
    const std::array<int32_t,4> &meta() const noexcept { return meta_; }
    const std::array<double,66> &values() const noexcept { return values_; }
    const std::array<double,4> &gc() const noexcept { return gc_; }
    const std::vector<Pattern> &patterns() const noexcept { return patterns_; }
private:
    std::string id_;
    std::array<int32_t,4> meta_;
    std::array<double,66> values_;
    std::array<double,4> gc_;
    std::vector<Pattern> patterns_;
};
enum class SupportPolicy : uint32_t { LegacyCutoff=0, StrictPositive=1 };
struct ModelConfig {
    uint32_t id; int nkappa; bool equal_frequency;
    SupportPolicy support_policy; double support_cutoff;
};
/* Small request-local state. Future optimizer state/cache is separate; no
 * matrix is owned by a resident model. This construction consumes no RNG. */
struct ModelState {
    ModelConfig config;
    std::array<double,7> x;
    std::array<double,6> rates;
    std::array<double,64> pi;
    double kappa, omega;
};
ModelConfig model_config(uint32_t model, SupportPolicy);
/* Pure mapping. Every consumer receives the same immutable expanded rates;
 * no stats/output function may expand an expanded vector again. */
std::array<double,6> expanded_rates(uint32_t model, const double *x);
ModelState model_state(const PairData &, uint32_t model, SupportPolicy, const double *x);
/* The output is caller/worker-owned storage. Q, S and A never alias. */
void build_q_snapshot(const ModelState &, double *output);
}
#endif
