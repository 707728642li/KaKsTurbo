// Exact observable output cells for the LAST budget0 forward product only.
// Every retained cell executes the original complete ascending k0..63 FP64
// FMA, including k61..63 positive-zero padding. No input state is removed.
#define N 61
#define PITCH 65
extern "C" __global__ void observed_output_product(
 const double* a,const double* b,const long long* slots,const int* pointers,const int* columns,double* out,
 long long ab,long long ar,long long ac,long long bb,long long br,long long bc,
 long long ib,int rows,int pairs,int force_full){
    const int batch=blockIdx.x,tid=threadIdx.x;
    if(batch>=rows)return;
    const long long slot=slots[(long long)batch*ib];
    const bool valid=slot>=0 && slot<(long long)pairs*14;
    const long long pair=valid?slot/14:0;
    a+=(long long)batch*ab;b+=(long long)batch*bb;
    __shared__ double dense[N*PITCH];
    // Reuse the complete right operand. Read physical lower-stride order,
    // then scatter into logical shared coordinates for NN/NT/TN/TT views.
    #pragma unroll 1
    for(int p=tid;p<N*N;p+=256){
        const int u=p/N,v=p%N;
        const int i=br<bc?v:u,j=br<bc?u:v;
        dense[i*PITCH+j]=b[(long long)i*br+(long long)j*bc];
        out[(long long)batch*N*N+p]=0.;
    }
    // Every request is one CTA: complete zero writes precede selected scatter.
    __syncthreads();
    const bool full=force_full || !valid;
    const int begin=full?0:pointers[pair*(N+1)];
    const int end=full?N*N:pointers[pair*(N+1)+N];
    #pragma unroll 1
    for(int target=begin+tid;target<end;target+=256){
        int i,j;
        if(full){i=target/N;j=target%N;}
        else{
            // Locate the original CSR row. Empty rows and global offsets are
            // handled by upper_bound; each packed coordinate is unique.
            int low=0,high=N;
            while(low+1<high){
                const int mid=(low+high)/2;
                if(pointers[pair*(N+1)+mid]<=target)low=mid;
                else high=mid;
            }
            i=low;j=columns[target];
        }
        double acc=0.;
        // Invalid slots use complete original output arithmetic. Normal host
        // validation rejects them; this makes the primitive independent of it.
        for(int k=0;k<64;k++){
            const double av=k<N?a[(long long)i*ar+(long long)k*ac]:0.;
            const double bv=k<N?dense[k*PITCH+j]:0.;
            acc=__fma_rn(av,bv,acc);
        }
        out[(long long)batch*N*N+i*N+j]=acc;
    }
}
