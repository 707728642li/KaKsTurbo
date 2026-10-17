#ifndef KAKS_FIBER_OPTIMIZER_PROTECTED_BRIDGE_H
#define KAKS_FIBER_OPTIMIZER_PROTECTED_BRIDGE_H
/* Two immutable-until-replaced exact objective entries, outside the FIFO.
 * Slot 0 protects the feasible baseline; slot 1 the accepted best point. */
void kaks_fiber_pin_objective(void *model,const double *x,int np,double value,int entry);
#endif
