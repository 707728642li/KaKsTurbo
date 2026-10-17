#ifndef KAKS_TORCH_GRAPH_HOST_GROUPED_H
#define KAKS_TORCH_GRAPH_HOST_GROUPED_H
#include "torch_graph_host.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Stable permutation of a validated wave by each row's bound level.
 * order[count] int64 row positions, levels[count] int32, offsets[18] uint32.
 * Context/inputs use the unchanged base host ABI. */
KAKS_HOST_EXPORT int gy94_graph_host_group(void *context,const int64_t *slots,
    const double *parameters,uint32_t count,int max_squarings,int adaptive,
    int64_t *order,int32_t *levels,uint32_t *offsets,double *maximum_bound,
    char *error,uint32_t capacity);
#ifdef __cplusplus
}
#endif
#endif
