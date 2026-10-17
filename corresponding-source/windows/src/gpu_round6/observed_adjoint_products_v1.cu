// FP64 contractions of the first objective adjoint on exact prepared support.
// Every output remains dense 61x61. Unsafe/nonfinite/off-support inputs retain
// all original ascending k0..63 __fma_rn operations and positive-zero padding.
#define N 61

extern "C" __global__ void observed_adjoint_safe(const double* g,const double* d,
 const long long* slots,const bool* support,int* safe,
 long long gb,long long gr,long long gc,long long db,long long dr,long long dc,
 long long ib,int rows,int pairs,int force_full){
    const int row=blockIdx.x,tid=threadIdx.x;
    if(row>=rows)return;
    const long long slot=slots[(long long)row*ib];
    const bool valid=slot>=0 && slot<(long long)pairs*14;
    const long long pair=valid?slot/14:0;
    __shared__ int bad[256];
    int failure=!valid || force_full;
    for(int p=tid;p<N*N;p+=256){
        const int i=p/N,j=p%N;
        const double gv=g[(long long)row*gb+(long long)i*gr+(long long)j*gc];
        const double dv=d[(long long)row*db+(long long)i*dr+(long long)j*dc];
        failure|=!isfinite(gv) || !isfinite(dv);
        if(valid && !support[pair*N*N+p] && gv!=0.)failure=1;
    }
    bad[tid]=failure;__syncthreads();
    for(int stride=128;stride>0;stride/=2){
        if(tid<stride)bad[tid]|=bad[tid+stride];
        __syncthreads();
    }
    if(tid==0)safe[row]=!bad[0];
}

extern "C" __global__ void observed_adjoint_left(const double* g,const double* d,
 const long long* slots,const int* pointers,const int* columns,const int* safe,
 double* out,long long gb,long long gr,long long gc,
 long long db,long long dr,long long dc,long long ib,int rows){
    const int batch=blockIdx.z,i=blockIdx.y*4+threadIdx.y;
    const int j=blockIdx.x*32+threadIdx.x;
    if(batch>=rows || i>=N || j>=N)return;
    g+=(long long)batch*gb;d+=(long long)batch*db;
    double acc=0.;
    if(safe[batch]){
        const long long pair=slots[(long long)batch*ib]/14;
        const int begin=pointers[pair*(N+1)+i],end=pointers[pair*(N+1)+i+1];
        for(int p=begin;p<end;p++){
            const int k=columns[p];
            acc=__fma_rn(g[(long long)i*gr+(long long)k*gc],
                         d[(long long)k*dr+(long long)j*dc],acc);
        }
        // Original k61..63 use +0 for both operands. A +0 add has no effect
        // on nonzero finite values and normalizes a signed-zero accumulator.
        acc=__fma_rn(0.,0.,acc);
    }else{
        for(int k=0;k<64;k++){
            const double gv=k<N?g[(long long)i*gr+(long long)k*gc]:0.;
            const double dv=k<N?d[(long long)k*dr+(long long)j*dc]:0.;
            acc=__fma_rn(gv,dv,acc);
        }
    }
    out[(long long)batch*N*N+i*N+j]=acc;
}

extern "C" __global__ void observed_adjoint_right(const double* g,const double* d,
 const long long* slots,const int* pointers,const int* columns,const int* safe,
 double* out,long long gb,long long gr,long long gc,
 long long db,long long dr,long long dc,long long ib,int rows){
    const int batch=blockIdx.z,i=blockIdx.y*4+threadIdx.y;
    const int j=blockIdx.x*32+threadIdx.x;
    if(batch>=rows || i>=N || j>=N)return;
    g+=(long long)batch*gb;d+=(long long)batch*db;
    double acc=0.;
    if(safe[batch]){
        const long long pair=slots[(long long)batch*ib]/14;
        const int begin=pointers[pair*(N+1)+j],end=pointers[pair*(N+1)+j+1];
        for(int p=begin;p<end;p++){
            const int k=columns[p];
            acc=__fma_rn(d[(long long)i*dr+(long long)k*dc],
                         g[(long long)k*gr+(long long)j*gc],acc);
        }
        acc=__fma_rn(0.,0.,acc);
    }else{
        for(int k=0;k<64;k++){
            const double dv=k<N?d[(long long)i*dr+(long long)k*dc]:0.;
            const double gv=k<N?g[(long long)k*gr+(long long)j*gc]:0.;
            acc=__fma_rn(dv,gv,acc);
        }
    }
    out[(long long)batch*N*N+i*N+j]=acc;
}
