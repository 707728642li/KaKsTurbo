/* Conda ILP64 OpenBLAS adapter. The estimator-facing ABI is unchanged. */
#include <cstdint>
#include <cstdio>

extern "C" {
void cblas_dgemm64_(int, int, int, int64_t, int64_t, int64_t, double,
                    const double *, int64_t, const double *, int64_t,
                    double, double *, int64_t);
int openblas_set_num_threads_local(int);
int openblas_get_num_threads64_();
char *openblas_get_config64_();

void cblas_dgemm_64(int layout, int ta, int tb, int64_t m, int64_t n,
                    int64_t k, double alpha, const double *a, int64_t lda,
                    const double *b, int64_t ldb, double beta, double *c,
                    int64_t ldc) {
    cblas_dgemm64_(layout, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
int MKL_Set_Num_Threads_Local(int count) { return openblas_set_num_threads_local(count); }
int MKL_Get_Max_Threads() { return openblas_get_num_threads64_(); }
void MKL_Get_Version_String(char *out, int capacity) {
    if (out && capacity > 0)
        std::snprintf(out, static_cast<size_t>(capacity),
                      "KaKsTurbo Conda OpenBLAS64 ABI bridge; %s", openblas_get_config64_());
}
}
