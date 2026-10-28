/* Independent execution bound: retain per-column positive basis sums.
 * No Q, likelihood, derivative, polynomial, or optimizer is evaluated here.
 * Pair/parity stride = 61*12 column coefficients followed by 12 mr terms.
 */
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <new>
#include <vector>

namespace {
constexpr size_t stride = 61 * 12 + 12;
constexpr uint32_t max_pairs = 1000; // This new experiment is development-only.
constexpr double theta = 1.090863719290036;
const int nkappa[14] = {0,0,1,1,2,2,2,2,3,3,4,4,5,5};
const int rate_map[7][6] = {{5,5,5,5,5,5},{0,0,5,5,5,5},
    {0,1,5,5,5,5},{0,0,1,1,5,5},{0,1,2,2,5,5},
    {0,0,1,2,3,5},{0,1,2,3,4,5}};
struct Context {
    uint32_t pairs;
    std::vector<double> coefficients;
    Context(uint32_t n, const double* values)
        : pairs(n), coefficients(values, values + static_cast<size_t>(n)*2*stride) {}
};
int fail(const char* message, char* error, uint32_t capacity) {
    if (error && capacity) {
        const size_t n = std::min(std::strlen(message), static_cast<size_t>(capacity-1));
        std::memcpy(error, message, n); error[n] = 0;
    }
    return 1;
}
int choose(Context* c, int64_t slot, const double* x, int limit, int adaptive,
           int* level, double* bound, char* error, uint32_t capacity) {
    if (slot < 0 || slot >= static_cast<int64_t>(c->pairs)*14)
        return fail("Invalid tight-bound slot", error, capacity);
    *level = limit; *bound = 0.;
    if (!adaptive) return 0;
    // Invalid numeric inputs retain the maximum-budget fallback. Validation
    // remains the old host's responsibility; the device guard is authoritative.
    for (int k=0; k<7; ++k) if (!std::isfinite(x[k])) return 0;
    const int model = static_cast<int>(slot%14), nk = nkappa[model];
    const double omega = x[nk+1];
    if (x[0] < 0. || omega <= 0.) return 0;
    double raw[6], weights[12];
    for (int k=0; k<6; ++k) raw[k] = k<nk ? x[k+1] : 1.;
    for (int k=0; k<6; ++k) {
        weights[k] = raw[rate_map[model/2][k]];
        if (weights[k] <= 0.) return 0;
        weights[k+6] = weights[k]*omega;
        if (!std::isfinite(weights[k+6])) return 0;
    }
    const double* coeff = &c->coefficients[(static_cast<size_t>(slot/14)*2+model%2)*stride];
    double mr = 0.;
    // Exactly the old host's twelve-term ascending left fold; full unmasked mr.
    for (int k=0; k<12; ++k) mr += coeff[61*12+k]*weights[k];
    if (mr <= 0. || !std::isfinite(mr)) return 0;
    double norm = 0.;
    for (int column=0; column<61; ++column) {
        double sum = 0.;
        for (int k=0; k<12; ++k) sum += coeff[column*12+k]*weights[k];
        if (!std::isfinite(sum)) return 0;
        if (sum > norm) norm = sum;
    }
    const double value = (x[0]/mr)*norm;
    if (!std::isfinite(value)) return 0;
    *bound = value;
    if (value <= theta/(1.+1e-12)) { *level=0; return 0; }
    // Avoid float-to-int overflow for extreme finite requests: retain limit.
    const double scaled = value*(1.+1e-12)/theta;
    if (!std::isfinite(scaled)) return 0;
    const double needed = std::ceil(std::log2(scaled));
    if (!std::isfinite(needed) || needed >= limit) return 0;
    *level = needed <= 0. ? 0 : static_cast<int>(needed);
    return 0;
}
}

extern "C" __declspec(dllexport) int gy94_tight_host_create(uint32_t pairs,
    const double* coefficients, void** output, char* error, uint32_t capacity) {
    if (output) *output = nullptr;
    if (!pairs || pairs>max_pairs || !coefficients || !output)
        return fail("Tight host requires 1..1000 development pairs", error, capacity);
    const size_t count = static_cast<size_t>(pairs)*2*stride;
    for (size_t i=0; i<count; ++i)
        if (!std::isfinite(coefficients[i]) || coefficients[i]<0.)
            return fail("Tight coefficients must be finite and nonnegative", error, capacity);
    try { *output = new Context(pairs, coefficients); }
    catch (...) { return fail("Tight host allocation failed", error, capacity); }
    return 0;
}
extern "C" __declspec(dllexport) void gy94_tight_host_free(void* context) {
    delete static_cast<Context*>(context);
}
extern "C" __declspec(dllexport) int gy94_graph_host_group(void* raw,
    const int64_t* slots, const double* x, uint32_t count, int limit, int adaptive,
    int64_t* order, int32_t* levels, uint32_t* offsets, double* maximum_bound,
    char* error, uint32_t capacity) {
    Context* c = static_cast<Context*>(raw);
    if (!c || limit<0 || limit>16 || !offsets || !maximum_bound ||
        (count && (!slots || !x || !order || !levels)))
        return fail("Invalid tight grouped request", error, capacity);
    uint32_t histogram[17]={0}, cursor[17]={0}; *maximum_bound=0.;
    for (uint32_t row=0; row<count; ++row) {
        int level; double bound;
        const int code=choose(c,slots[row],x+static_cast<size_t>(row)*7,
            limit,adaptive,&level,&bound,error,capacity);
        if (code) return code;
        levels[row]=level; ++histogram[level];
        if (bound>*maximum_bound) *maximum_bound=bound;
    }
    offsets[0]=0;
    for (int k=0; k<17; ++k) { offsets[k+1]=offsets[k]+histogram[k]; cursor[k]=offsets[k]; }
    for (uint32_t row=0; row<count; ++row) order[cursor[levels[row]]++]=row;
    return 0;
}
