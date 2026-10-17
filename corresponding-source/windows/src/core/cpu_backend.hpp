#ifndef KAKSCUDA_CPU_BACKEND_HPP
#define KAKSCUDA_CPU_BACKEND_HPP
#include "cpu_backend.h"
#include "shared_core.hpp"
namespace kakscuda {
/* Backend-local MKL implementation. The state evaluator deliberately does not
 * call model_state or expanded_rates: statistics and MA may supply an already
 * expanded, coherent immutable state. Worker/session ownership is in the C ABI. */
class CpuSession;
void cpu_evaluate_state(CpuSession &, const PairData &, const ModelState &,
    double *workspace, double &objective, double *diagnostics,
    double *optional_transition);
}
#endif
