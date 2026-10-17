// First-order full FP64 T18 graph helpers. No approximate arithmetic.
__device__ __constant__ double THETA=1.090863719290036;
__device__ __constant__ double LOG2THETA=0.12547087821039307;
extern "C" __global__ void native_norm_scale(const double* a,double* z,double* diagnostic,int rows,int budget){
    int row=blockIdx.x,tid=threadIdx.x;if(row>=rows)return;
    __shared__ double columns[256];__shared__ double scale,valid;
    double col=0.;
    if(tid<61)for(int i=0;i<61;i++)col=col+fabs(a[(long long)row*3721+i*61+tid]);
    columns[tid]=tid<61?col:0.;__syncthreads();
    for(int stride=128;stride>0;stride/=2){
        if(tid<stride){double b=columns[tid+stride];columns[tid]=(isnan(columns[tid])||isnan(b))?__longlong_as_double(0x7ff8000000000000ULL):fmax(columns[tid],b);}
        __syncthreads();
    }
    if(tid==0){
        double norm=columns[0],required=ceil(log2(norm)-LOG2THETA);
        if(required<0.)required=0.;
        required=required+((norm*pow(2.,-required)>THETA)?1.:0.);
        valid=isfinite(norm)&&isfinite(required)&&required<=budget;
        double selected=valid?required:0.;
        // selected is finite integer0..16 after valid/fallback; cast never sees NaN/Inf.
        scale=__longlong_as_double((unsigned long long)(1023-(int)selected)<<52);
        diagnostic[row*4]=valid;diagnostic[row*4+1]=norm;
        diagnostic[row*4+2]=required;diagnostic[row*4+3]=selected;
    }
    __syncthreads();
    for(int p=tid;p<3721;p+=256){long long idx=(long long)row*3721+p;z[idx]=(valid?a[idx]:0.)*scale;}
}
extern "C" __global__ void native_add2(const double* a,const double* b,double* out,long long n){
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[i]=a[i]+b[i];
}
extern "C" __global__ void native_add3(const double* a,const double* b,const double* c,double* out,long long n){
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[i]=(a[i]+b[i])+c[i];
}
extern "C" __global__ void native_add_pair(const double* a,const double* b,const double* c,const double* d,
 double* ab,double* cd,long long n){
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i<n){ab[i]=a[i]+b[i];cd[i]=c[i]+d[i];}
}
extern "C" __global__ void native_square_select(const double* previous,const double* square,
 const double* diagnostic,double* out,long long n,int step){
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[i]=diagnostic[(i/3721)*4+3]>step?square[i]:previous[i];
}
extern "C" __global__ void native_valid_exp(const double* exponential,const double* diagnostic,double* out,long long n){
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n)out[i]=diagnostic[(i/3721)*4]!=0.?exponential[i]:__longlong_as_double(0x7ff8000000000000ULL);
}
extern "C" __global__ void native_mask_gradient(const double* g,const double* diagnostic,double* out,long long n,int step,int mode){
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n){bool mask=mode==0?diagnostic[(i/3721)*4]!=0.:diagnostic[(i/3721)*4+3]>step;out[i]=mask?g[i]:0.;}
}
extern "C" __global__ void native_reverse_square(const double* left,const double* right,
 const double* incoming,const double* diagnostic,double* out,long long n,int step){
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n){double passthrough=diagnostic[(i/3721)*4+3]>step?0.:incoming[i];out[i]=(left[i]+right[i])+passthrough;}
}
extern "C" __global__ void native_add3_scale(const double* a,const double* b,const double* c,
 const double* diagnostic,double* out,long long n){
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n){const double* d=diagnostic+(i/3721)*4;if(d[0]!=0.){
        // Norm guard proves selected is an integer0..16, so exponent bits give exact2^-selected.
        double scale=__longlong_as_double((unsigned long long)(1023-(int)d[3])<<52);
        out[i]=((a[i]+b[i])+c[i])*scale;
    }else out[i]=0.;}
}
extern "C" __global__ void native_pack(const double* f,const double* diagnostic,const double* raw,
 const double* x,const long long* indices,const long long* nkappa,double* out,int rows,int kind){
    int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=rows)return;
    int active=(int)nkappa[indices[row]]+2;
    bool valid=diagnostic[row*4]!=0.&&isfinite(f[row]);out[row*11]=f[row];
    out[row*11+2]=diagnostic[row*4+1];out[row*11+3]=diagnostic[row*4+2];
    for(int k=0;k<7;k++){
        double v=kind&&k<active?raw[row*7+k]*x[row*7+k]:0.;
        if(!isfinite(v))valid=false;out[row*11+4+k]=v;
    }
    out[row*11+1]=valid?1.:0.;
}
