#ifndef KAKS_FIBER_OPTIMIZER_NEAR_BRIDGE_H
#define KAKS_FIBER_OPTIMIZER_NEAR_BRIDGE_H
#include <cstring>
// Local to a single ming2 invocation; never shares a window across restarts.
struct KaksNearWindow {
    int count,nfree;
    int marks[7],ix[7];
    double anchor[7],low[7],high[7],best[7];
    double previous_f,start_f,gain,best_f,max_scaled_span;
    bool have_best;
    KaksNearWindow() : count(0),nfree(0),previous_f(0),start_f(0),gain(0),
        best_f(0),max_scaled_span(0),have_best(false) {}
};
void kaks_fiber_near_begin(void *model,KaksNearWindow &window);
void kaks_fiber_near_clear(void *model,KaksNearWindow &window);
bool kaks_fiber_near_step(void *model,KaksNearWindow &window,int round,int np,
    bool incoming_identity,const double *incoming_x,double incoming_f,
    const double *incoming_g,double incoming_size,const int *incoming_marks,
    const int *incoming_ix,int incoming_nfree,double *x,double *f,
    const double *g,double size,const int *marks,const int *ix,int nfree,
    const double xb[][2],double e,double small_diff);
#endif
