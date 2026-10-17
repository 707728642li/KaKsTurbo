#ifndef KAKS_FIBER_OPTIMIZER_BRIDGE_H
#define KAKS_FIBER_OPTIMIZER_BRIDGE_H
bool kaks_fiber_objective(void *model, double *x, int np, double *objective);
void kaks_fiber_record_fit(void *model, int result, const double *x, int np, int round);
void kaks_fiber_record_eigen(int status);
/* Original PairwiseCodon scratch is heap allocated before its first callback.
 * This guard frees it on cancellation without altering normal arithmetic. */
class KaksFiberScratchScope {
    double *data;
public:
    explicit KaksFiberScratchScope(double *value) : data(value) {}
    ~KaksFiberScratchScope() { delete [] data; }
    void release() { data=0; }
private:
    KaksFiberScratchScope(const KaksFiberScratchScope &);
    KaksFiberScratchScope &operator=(const KaksFiberScratchScope &);
};
#endif
