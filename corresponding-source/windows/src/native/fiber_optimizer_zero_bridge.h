#ifndef KAKS_FIBER_OPTIMIZER_ZERO_BRIDGE_H
#define KAKS_FIBER_OPTIMIZER_ZERO_BRIDGE_H
bool kaks_fiber_zero_seed(void *model,double *f,double *x,const double xb[][2],double e,int np);
void kaks_fiber_zero_record(void *model,double f,const double *x,const double xb[][2],
    int np,int round,int *status);
#endif
