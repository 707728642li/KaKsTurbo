#ifndef KAKS_TORCH_GRAPH_HOST_H
#define KAKS_TORCH_GRAPH_HOST_H
#include <stdint.h>
#ifdef _WIN32
#define KAKS_HOST_EXPORT __declspec(dllexport)
#else
#define KAKS_HOST_EXPORT
#endif
#ifdef __cplusplus
extern "C" {
#endif
/* bounds[pair, parity,24]: norm12 then mr12; copied into the context. */
KAKS_HOST_EXPORT int gy94_graph_host_create(uint32_t pairs,const double *bounds,
    void **out,char *error,uint32_t capacity);
KAKS_HOST_EXPORT void gy94_graph_host_free(void *context);
/* One validation per wave, including duplicates across different chunks. */
KAKS_HOST_EXPORT int gy94_graph_host_validate(void *context,const int64_t *slots,
    const double *parameters,uint32_t count,char *error,uint32_t capacity);
/* Uses already validated immutable CPU tensors; no allocation or mutation. */
KAKS_HOST_EXPORT int gy94_graph_host_choose(void *context,const int64_t *slots,
    const double *parameters,uint32_t count,int max_squarings,int adaptive,
    int *squarings,double *maximum_bound,char *error,uint32_t capacity);
/* result[count,3] = objective,valid_flag,norm1. Invalid flags are collected;
 * finite objectives are required only for valid rows until explicit fallback. */
KAKS_HOST_EXPORT int gy94_graph_host_analyze(const double *result,uint32_t count,
    int32_t *bad_rows,uint32_t *bad_count,double *maximum_norm,char *error,uint32_t capacity);
#ifdef __cplusplus
}
#endif
#endif
