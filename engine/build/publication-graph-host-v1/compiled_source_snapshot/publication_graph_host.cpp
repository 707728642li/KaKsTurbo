/* Publication-mode extension. The old v2/grouped host sources are immutable.
 * Repeated slots are required for simultaneous finite-difference probes. */
#include "torch_graph_host_grouped.cpp"
extern "C" __declspec(dllexport) int kaks_publication_validate(void *raw,
    const int64_t *slots,const double *x,uint32_t count,char *error,uint32_t capacity){
    if(!raw||(count&&(!slots||!x)))return fail("Missing publication request",error,capacity);
    for(uint32_t i=0;i<count;i++){
        const int status=gy94_graph_host_validate(raw,slots+i,x+static_cast<size_t>(i)*7,1,error,capacity);
        if(status)return status;
        const int nk=nkappa[slots[i]%14];const double *v=x+static_cast<size_t>(i)*7;
        if(v[0]<0.||v[nk+1]<=0.)return fail("Nonpositive publication time/omega",error,capacity);
        for(int j=0;j<nk;j++)if(v[j+1]<=0.)return fail("Nonpositive publication exchangeability",error,capacity);
    }return 0;
}
