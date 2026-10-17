#ifndef KAKSCUDA_STATISTICS_HPP
#define KAKSCUDA_STATISTICS_HPP
#include "statistics.h"
#include "shared_core.hpp"
namespace kakscuda {
/* Consumers accept immutable expanded ModelState, never a second rate parse.
 * No matrix/likelihood, RNG, optimizer, allocation or shared mutable state. */
KaksModelStatistics statistics_from_state(const PairData &,const ModelState &,
    bool score_available,double lnL,bool is_ma=false);
KaksMaStatistics average_statistics(const PairData &,const KaksModelStatistics *,uint32_t);
ModelState averaged_model_state(const PairData &,const KaksMaStatistics &);
}
#endif
