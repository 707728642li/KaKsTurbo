// New immutable layout candidate. Same exact observed CSR/CSC and FP64 FMA.
// One CTA per request; finite/support certificate is fused with shared D load.
#define N 61
#define PITCH 65

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

template<bool RIGHT>
__device__ void observed_shared_product(const double* g,const double* d,
 const long long* slots,const int* pointers,const int* columns,const bool* support,
 int* row_safety,double* out,long long gb,long long gr,long long gc,
 long long db,long long dr,long long dc,long long ib,int rows,int pairs,int force_full){
    const int batch=blockIdx.x,tid=threadIdx.x;
    if(batch>=rows)return;
    const long long slot=slots[(long long)batch*ib];
    const bool valid=slot>=0 && slot<(long long)pairs*14;
    const long long pair=valid?slot/14:0;
    g+=(long long)batch*gb;d+=(long long)batch*db;
    __shared__ double dense[N*PITCH];
    __shared__ int bad[256];
    int failure=!valid || force_full;
    // Load along the lower-stride physical dimension. The logical shared
    // matrix is the same for NN/NT/TN/TT and arbitrary nonnegative views.
    for(int p=tid;p<N*N;p+=256){
        const int u=p/N,v=p%N;
        const int i=dr<dc?v:u,j=dr<dc?u:v;
        const double dv=d[(long long)i*dr+(long long)j*dc];
        dense[i*PITCH+j]=dv;
        const double gv=g[(long long)u*gr+(long long)v*gc];
        failure|=!isfinite(gv) || !isfinite(dv);
        if(valid && !support[pair*N*N+p] && gv!=0.)failure=1;
    }
    bad[tid]=failure;__syncthreads();
    for(int stride=128;stride>0;stride/=2){
        if(tid<stride)bad[tid]|=bad[tid+stride];
        __syncthreads();
    }
    if(tid==0)row_safety[batch]=!bad[0];
    const bool safe=!bad[0];
    for(int p=tid;p<N*N;p+=256){
        const int i=p/N,j=p%N;
        double acc=0.;
        if(safe){
            const int key=RIGHT?j:i;
            const int begin=pointers[pair*(N+1)+key],end=pointers[pair*(N+1)+key+1];
            for(int position=begin;position<end;position++){
                const int k=columns[position];
                if(RIGHT){
                    acc=__fma_rn(dense[i*PITCH+k],g[(long long)k*gr+(long long)j*gc],acc);
                }else{
                    acc=__fma_rn(g[(long long)i*gr+(long long)k*gc],dense[k*PITCH+j],acc);
                }
            }
            acc=__fma_rn(0.,0.,acc);
        }else{
            for(int k=0;k<64;k++){
                if(RIGHT){
                    const double dv=k<N?dense[i*PITCH+k]:0.;
                    const double gv=k<N?g[(long long)k*gr+(long long)j*gc]:0.;
                    acc=__fma_rn(dv,gv,acc);
                }else{
                    const double gv=k<N?g[(long long)i*gr+(long long)k*gc]:0.;
                    const double dv=k<N?dense[k*PITCH+j]:0.;
                    acc=__fma_rn(gv,dv,acc);
                }
            }
        }
        out[(long long)batch*N*N+p]=acc;
    }
}

extern "C" __global__ void observed_adjoint_left_shared(const double* g,const double* d,
 const long long* slots,const int* pointers,const int* columns,const bool* support,
 int* row_safety,double* out,long long gb,long long gr,long long gc,
 long long db,long long dr,long long dc,long long ib,int rows,int pairs,int force_full){
    observed_shared_product<false>(g,d,slots,pointers,columns,support,row_safety,out,gb,gr,gc,db,dr,dc,ib,rows,pairs,force_full);
}

extern "C" __global__ void observed_adjoint_right_shared(const double* g,const double* d,
 const long long* slots,const int* pointers,const int* columns,const bool* support,
 int* row_safety,double* out,long long gb,long long gr,long long gc,
 long long db,long long dr,long long dc,long long ib,int rows,int pairs,int force_full){
    observed_shared_product<true>(g,d,slots,pointers,columns,support,row_safety,out,gb,gr,gc,db,dr,dc,ib,rows,pairs,force_full);
}
