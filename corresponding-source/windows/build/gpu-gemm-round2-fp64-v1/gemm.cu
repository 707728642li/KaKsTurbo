// Experimental batched 61x61 FP64 GEMM. Each input can be a transposed view.
// No fast math. Products accumulate with explicit IEEE double FMA.
// Four output registers/thread and shared-memory input reuse; padding is zero.
#define N 61
template<int TILE, int ROWS, int COLS>
__device__ void tiled_gemm(const double* a, const double* b, double* c,
    long long ab, long long ar, long long ac,
    long long bb, long long br, long long bc, int batches) {
    const int batch=blockIdx.z;
    if (batch>=batches) return;
    const int tx=threadIdx.x, ty=threadIdx.y;
    const int tid=ty*16+tx;
    const int row0=blockIdx.y*TILE, col0=blockIdx.x*TILE;
    __shared__ double sa[TILE][16];
    __shared__ double sb[16][TILE];
    double acc[ROWS][COLS];
    #pragma unroll
    for (int r=0;r<ROWS;r++) {
        #pragma unroll
        for (int s=0;s<COLS;s++) acc[r][s]=0.;
    }
    a+=batch*ab; b+=batch*bb; c+=(long long)batch*N*N;
    for (int base=0;base<N;base+=16) {
        #pragma unroll
        for (int offset=0;offset<TILE*16;offset+=256) {
            int pos=tid+offset, ai=pos/16, ak=pos%16;
            int bk=pos/TILE, bj=pos%TILE;
            sa[ai][ak]=(row0+ai<N && base+ak<N)?a[(row0+ai)*ar+(base+ak)*ac]:0.;
            sb[bk][bj]=(base+bk<N && col0+bj<N)?b[(base+bk)*br+(col0+bj)*bc]:0.;
        }
        __syncthreads();
        #pragma unroll
        for (int k=0;k<16;k++) {
            #pragma unroll
            for (int r=0;r<ROWS;r++) {
                double av=sa[ty+r*16][k];
                #pragma unroll
                for (int s=0;s<COLS;s++) acc[r][s]=__fma_rn(av,sb[k][tx+s*16],acc[r][s]);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int r=0;r<ROWS;r++) {
        int ri=row0+ty+r*16;
        #pragma unroll
        for (int s=0;s<COLS;s++) {
            int ci=col0+tx+s*16;
            if (ri<N && ci<N) c[ri*N+ci]=acc[r][s];
        }
    }
}
extern "C" __global__ void gemm_tile16(const double* a,const double* b,double* c,
 long long ab,long long ar,long long ac,long long bb,long long br,long long bc,int batches) {
    tiled_gemm<16,1,1>(a,b,c,ab,ar,ac,bb,br,bc,batches);
}
extern "C" __global__ void gemm_tile32(const double* a,const double* b,double* c,
 long long ab,long long ar,long long ac,long long bb,long long br,long long bc,int batches) {
    tiled_gemm<32,2,2>(a,b,c,ab,ar,ac,bb,br,bc,batches);
}
