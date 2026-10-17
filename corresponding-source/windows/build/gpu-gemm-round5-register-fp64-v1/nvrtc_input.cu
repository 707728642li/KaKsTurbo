// Execution-layout candidates only: the round2 serial k=0..63 __fma_rn tape.
// Every output starts at +0 and includes exactly the same three zero pads.
// All physical operands, accumulators and outputs are double. No split-K.
#define N 61
template<int M,int P,int TX,int TY,bool AT,bool BT>
__device__ void register_gemm(const double* a,const double* b,double* c,
    long long ab,long long ar,long long ac,
    long long bb,long long br,long long bc,int batches) {
    const int batch=blockIdx.z;
    if(batch>=batches)return;
    const int tx=threadIdx.x,ty=threadIdx.y,tid=ty*TX+tx;
    const int row0=blockIdx.y*M,col0=blockIdx.x*P;
    const int R=M/TY,S=P/TX;
    __shared__ double sa[M][17];
    __shared__ double sb[16][P+1];
    double acc[R][S];
    #pragma unroll
    for(int r=0;r<R;r++) {
        #pragma unroll
        for(int s=0;s<S;s++)acc[r][s]=0.;
    }
    a+=batch*ab;b+=batch*bb;c+=(long long)batch*N*N;
    for(int base=0;base<N;base+=16) {
        #pragma unroll
        for(int offset=0;offset<M*16;offset+=TX*TY) {
            int pos=tid+offset;
            int ai=AT?pos%M:pos/16,ak=AT?pos/M:pos%16;
            sa[ai][ak]=(row0+ai<N && base+ak<N)?a[(row0+ai)*ar+(base+ak)*ac]:0.;
        }
        #pragma unroll
        for(int offset=0;offset<16*P;offset+=TX*TY) {
            int pos=tid+offset;
            int bk=BT?pos%16:pos/P,bj=BT?pos/16:pos%P;
            sb[bk][bj]=(base+bk<N && col0+bj<N)?b[(base+bk)*br+(col0+bj)*bc]:0.;
        }
        __syncthreads();
        #pragma unroll
        for(int k=0;k<16;k++) {
            #pragma unroll
            for(int r=0;r<R;r++) {
                const double av=sa[ty+r*TY][k];
                #pragma unroll
                for(int s=0;s<S;s++)acc[r][s]=__fma_rn(av,sb[k][tx+s*TX],acc[r][s]);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int r=0;r<R;r++) {
        int row=row0+ty+r*TY;
        #pragma unroll
        for(int s=0;s<S;s++) {
            int col=col0+tx+s*TX;
            if(row<N && col<N)c[row*N+col]=acc[r][s];
        }
    }
}
#define ENTRY(NAME,M,P,TX,TY,AT,BT) \
extern "C" __global__ void NAME(const double* a,const double* b,double* c, \
long long ab,long long ar,long long ac,long long bb,long long br,long long bc,int batches) { \
register_gemm<M,P,TX,TY,AT,BT>(a,b,c,ab,ar,ac,bb,br,bc,batches); }
ENTRY(gemm_whole64_nn,64,64,16,16,false,false)
ENTRY(gemm_whole64_nt,64,64,16,16,false,true)
ENTRY(gemm_whole64_tn,64,64,16,16,true,false)
ENTRY(gemm_whole64_tt,64,64,16,16,true,true)
ENTRY(gemm_rect32x64_nn,32,64,32,8,false,false)
ENTRY(gemm_rect32x64_nt,32,64,32,8,false,true)
ENTRY(gemm_rect32x64_tn,32,64,32,8,true,false)
ENTRY(gemm_rect32x64_tt,32,64,32,8,true,true)
