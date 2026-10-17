/* Compile the unchanged v2 helper and an independent grouping extension. */
#include "torch_graph_host.cpp"
#include "torch_graph_host_grouped.h"

extern "C" int gy94_graph_host_group(void *context,const int64_t *slots,
    const double *parameters,uint32_t count,int limit,int adaptive,int64_t *order,
    int32_t *levels,uint32_t *offsets,double *maximum_bound,char *error,uint32_t capacity) {
    if(!context||limit<0||limit>16||!offsets||!maximum_bound||
       (count&&(!slots||!parameters||!order||!levels)))
        return fail("Invalid grouped host request",error,capacity);
    uint32_t histogram[17]={0},cursor[17]={0};
    *maximum_bound=0.;
    for(uint32_t row=0;row<count;row++) {
        int level;double bound;
        const int status=gy94_graph_host_choose(context,slots+row,
            parameters+static_cast<size_t>(row)*7,1,limit,adaptive,&level,&bound,error,capacity);
        if(status)return status;
        if(level<0||level>limit)return fail("Unexpected host squaring level",error,capacity);
        levels[row]=level;histogram[level]++;
        if(bound>*maximum_bound)*maximum_bound=bound;
    }
    offsets[0]=0;
    for(int s=0;s<17;s++) {offsets[s+1]=offsets[s]+histogram[s];cursor[s]=offsets[s];}
    for(uint32_t row=0;row<count;row++)order[cursor[levels[row]]++]=row;
    return 0;
}
