#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
uint32_t kaks_opt_gradient_abi_version(void);
int kaks_opt_poll_kinds(void *,uint32_t,int64_t *,uint64_t *,double *,uint32_t *,uint32_t *,uint32_t *,char *,uint32_t);
/* gradient_log contains count*7 doubles, raw F derivatives w.r.t. log(raw x).
 * Only kind1 active cells are consumed. Neither F nor gradient is divided by L. */
int kaks_opt_submit_mixed(void *,uint32_t,const int64_t *,const uint64_t *,const double *,const double *,char *,uint32_t);
#ifdef __cplusplus
}
#endif
