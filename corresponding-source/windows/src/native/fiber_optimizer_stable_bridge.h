#ifndef KAKS_FIBER_OPTIMIZER_STABLE_BRIDGE_H
#define KAKS_FIBER_OPTIMIZER_STABLE_BRIDGE_H
/* Optional research policy, in an independent staged source/DLL only. */
void kaks_fiber_stabilize(void *model,double *f,double *x,double xb[][2],
                          double *space,double e,int np);
#endif
