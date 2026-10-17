#include "torch_graph_host.h"
#include <cmath>
#include <cstring>
#include <vector>
#include <new>

namespace {
const double theta=1.090863719290036;
const int nkappa[14]={0,0,1,1,2,2,2,2,3,3,4,4,5,5};
const int rate_map[7][6]={{5,5,5,5,5,5},{0,0,5,5,5,5},{0,1,5,5,5,5},
    {0,0,1,1,5,5},{0,1,2,2,5,5},{0,0,1,2,3,5},{0,1,2,3,4,5}};
struct Context {
    uint32_t pairs,generation;
    std::vector<double> coefficients;
    std::vector<uint32_t> seen;
    Context(uint32_t n,const double *bounds):pairs(n),generation(0),
        coefficients(bounds,bounds+static_cast<size_t>(n)*48),seen(static_cast<size_t>(n)*14,0) {}
};
int fail(const char *message,char *error,uint32_t capacity) {
    if(error&&capacity) {size_t n=std::strlen(message);if(n>=capacity)n=capacity-1;
        std::memcpy(error,message,n);error[n]=0;}
    return 1;
}
bool finite(double x) {return std::isfinite(x);}
}

extern "C" int gy94_graph_host_create(uint32_t pairs,const double *bounds,void **out,
    char *error,uint32_t capacity) {
    if(out)*out=0;
    if(!pairs||pairs>1000||!bounds||!out)return fail("Invalid host bound context",error,capacity);
    for(size_t i=0;i<static_cast<size_t>(pairs)*48;i++)
        if(!finite(bounds[i])||bounds[i]<0.)return fail("Bounds must be finite and nonnegative",error,capacity);
    try {*out=new Context(pairs,bounds);}catch(...){return fail("Host bound allocation failed",error,capacity);}
    return 0;
}
extern "C" void gy94_graph_host_free(void *context) {delete static_cast<Context *>(context);}
extern "C" int gy94_graph_host_validate(void *raw,const int64_t *slots,const double *x,
    uint32_t count,char *error,uint32_t capacity) {
    Context *c=static_cast<Context *>(raw);
    if(!c||(count&&(!slots||!x)))return fail("Null host validation input",error,capacity);
    if(++c->generation==0) {std::fill(c->seen.begin(),c->seen.end(),0);c->generation=1;}
    for(uint32_t i=0;i<count;i++) {
        const int64_t slot=slots[i];
        if(slot<0||slot>=static_cast<int64_t>(c->pairs)*14)return fail("Request slots must be valid",error,capacity);
        if(c->seen[slot]==c->generation)return fail("Request slots must be unique",error,capacity);
        c->seen[slot]=c->generation;
        for(int j=0;j<7;j++)if(!finite(x[static_cast<size_t>(i)*7+j]))
            return fail("Parameters must contain seven finite doubles per slot",error,capacity);
    }
    return 0;
}
extern "C" int gy94_graph_host_choose(void *raw,const int64_t *slots,const double *x,
    uint32_t count,int limit,int adaptive,int *squarings,double *maximum_bound,
    char *error,uint32_t capacity) {
    Context *c=static_cast<Context *>(raw);
    if(!c||!squarings||!maximum_bound||limit<0||limit>16||(count&&(!slots||!x)))
        return fail("Invalid host bound request",error,capacity);
    *squarings=limit;*maximum_bound=0.;
    if(!adaptive)return 0;
    double maximum=0.;
    for(uint32_t row=0;row<count;row++) {
        const int64_t slot=slots[row];
        if(slot<0||slot>=static_cast<int64_t>(c->pairs)*14)return fail("Invalid bound slot",error,capacity);
        const int model=static_cast<int>(slot%14),nk=nkappa[model];
        const double *parameter=x+static_cast<size_t>(row)*7;
        const double omega=parameter[nk+1];
        double raw_rates[6],weights[12];
        for(int i=0;i<6;i++)raw_rates[i]=i<nk?parameter[i+1]:1.;
        if(parameter[0]<0.||omega<=0.)return 0;
        for(int i=0;i<6;i++) {
            weights[i]=raw_rates[rate_map[model/2][i]];
            if(weights[i]<=0.)return 0;
            weights[i+6]=weights[i]*omega;
        }
        const double *coeff=&c->coefficients[(static_cast<size_t>(slot/14)*2+model%2)*24];
        // Ascending left folds preserve the v1 Python sum order.
        double mr=0.,norm=0.;
        for(int i=0;i<12;i++)mr+=coeff[12+i]*weights[i];
        if(mr<=0.||!finite(mr))return 0;
        for(int i=0;i<12;i++)norm+=coeff[i]*weights[i];
        const double bound=(parameter[0]/mr)*norm;
        if(bound>maximum)maximum=bound;
    }
    *maximum_bound=maximum;
    if(!finite(maximum))return 0;
    if(maximum<=theta/(1.+1e-12)) {*squarings=0;return 0;}
    int value=static_cast<int>(std::ceil(std::log2(maximum*(1.+1e-12)/theta)));
    if(value<0)value=0;if(value>limit)value=limit;
    *squarings=value;return 0;
}
extern "C" int gy94_graph_host_analyze(const double *result,uint32_t count,
    int32_t *bad_rows,uint32_t *bad_count,double *maximum_norm,char *error,uint32_t capacity) {
    if(!bad_count||!maximum_norm||(count&&(!result||!bad_rows)))
        return fail("Null host result input",error,capacity);
    *bad_count=0;*maximum_norm=0.;
    for(uint32_t i=0;i<count;i++) {
        const double *row=result+static_cast<size_t>(i)*3;
        if(finite(row[2])&&row[2]>*maximum_norm)*maximum_norm=row[2];
        if(row[1]!=1.)bad_rows[(*bad_count)++]=static_cast<int32_t>(i);
        else if(!finite(row[0]))return fail("Nonfinite accepted T18 likelihood",error,capacity);
    }
    return 0;
}
