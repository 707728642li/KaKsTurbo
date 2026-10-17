#ifndef KAKS_CUDA_ADAPTER_BRIDGE_H
#define KAKS_CUDA_ADAPTER_BRIDGE_H

#include <string>

/* Internal bridge used only by the separate macro-hooked MSMA source copy. */
struct KaksNativeInjectedModel {
    std::string row;
    double aicc;
    double freq[64];
    double rates[6];
    double omega;
    double time_x0;
};

void kaks_native_get_injected_model(const char *model_name,
                                    KaksNativeInjectedModel &result);
void kaks_native_record_eigen_status(int status);

#endif
