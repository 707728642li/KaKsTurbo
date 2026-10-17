// CUDA request transport only: copy FP64 bytes, do not evaluate the model.
#include <cstdint>
#include <cstring>
#include <cmath>
#include <limits>
#include <vector>
#define API extern "C" __declspec(dllexport)

API int drv_split(const int64_t* order, uint32_t count, const uint32_t* kinds,
                  const uint32_t* offsets, int64_t* output, uint32_t* out_offsets) {
    if (!order || !kinds || !offsets || !output || !out_offsets || count>1000000 ||
        offsets[0]!=0 || offsets[17]!=count) return 1;
    std::vector<uint8_t> seen(count,0);
    for (uint32_t i=0;i<count;++i) {
        if (order[i]<0 || uint64_t(order[i])>=count || seen[order[i]] || kinds[order[i]]>1) return 2;
        seen[order[i]]=1;
    }
    for (int level=0;level<17;++level) if(offsets[level]>offsets[level+1]) return 3;
    uint32_t cursor=0;out_offsets[0]=0;
    for(int level=0;level<17;++level) for(uint32_t kind=0;kind<2;++kind) {
        for(uint32_t i=offsets[level];i<offsets[level+1];++i)
            if(kinds[order[i]]==kind) output[cursor++]=order[i];
        out_offsets[level*2+kind+1]=cursor;
    }
    return cursor==count?0:4;
}

API int drv_pack(const int64_t* slots,const double* x,const int64_t* order,
                 uint32_t count,uint32_t offset,uint32_t real,uint32_t size,
                 int64_t* out_slots,double* out_x) {
    if(!slots || !x || !order || !out_slots || !out_x || count>1000000 ||
       real==0 || size<real || size>1024 || (size&(size-1)) || offset>count || real>count-offset) return 1;
    for(uint32_t j=0;j<real;++j) if(order[offset+j]<0 || uint64_t(order[offset+j])>=count) return 2;
    for(uint32_t j=0;j<size;++j) {
        auto row=order[offset+(j<real?j:0)];
        std::memcpy(out_slots+j,slots+row,sizeof(int64_t));
        std::memcpy(out_x+j*7,x+row*7,7*sizeof(double));
    }
    return 0;
}

API int drv_scatter(const int64_t* order,uint32_t count,uint32_t offset,uint32_t real,
                    const double* packed,double* F,double* g,double* diagnostics,uint32_t* codes) {
    if(!order || !packed || !F || !g || !diagnostics || !codes || count>1000000 ||
       offset>count || real>count-offset) return 1;
    for(uint32_t j=0;j<real;++j) if(order[offset+j]<0 || uint64_t(order[offset+j])>=count) return 2;
    for(uint32_t j=0;j<real;++j) {
        auto row=order[offset+j];const double* p=packed+j*11;
        bool bad=p[1]!=1. || !std::isfinite(p[0]);
        for(int k=4;k<11;++k) if(!std::isfinite(p[k])) bad=true;
        std::memcpy(diagnostics+row*4,p,4*sizeof(double));codes[row]=bad?1:0;
        if(bad) {
            F[row]=std::numeric_limits<double>::quiet_NaN();
            for(int k=0;k<7;++k) g[row*7+k]=std::numeric_limits<double>::quiet_NaN();
        } else {
            std::memcpy(F+row,p,sizeof(double));
            std::memcpy(g+row*7,p+4,7*sizeof(double));
        }
    }
    return 0;
}
