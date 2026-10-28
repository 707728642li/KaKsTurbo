// Structural code1 codon-graph products, including every diagonal.
// Sorted CSR is rendered from the exact immutable codon_edges() contract.
// Only structural zero primal terms are skipped; outputs/adjoints stay dense.
// Explicit IEEE FP64 FMA, matching the existing gemm32 k-order on retained k.
#define N 61
__device__ const int SP_ROW_PTR[62]={/*ROW_POINTERS*/};
__device__ const int SP_COLUMN[587]={/*COLUMN_INDICES*/};

extern "C" __global__ void sparse_left(const double* s,const double* d,double* out,
    long long sb,long long sr,long long sc,
    long long db,long long dr,long long dc,int batches){
    const int batch=blockIdx.z;
    const int row=blockIdx.y*4+threadIdx.y;
    const int col=blockIdx.x*32+threadIdx.x;
    if(batch>=batches || row>=N || col>=N)return;
    s+=(long long)batch*sb;d+=(long long)batch*db;
    double accumulator=0.;
    const int begin=__ldg(&SP_ROW_PTR[row]),end=__ldg(&SP_ROW_PTR[row+1]);
    for(int p=begin;p<end;p++){
        const int k=__ldg(&SP_COLUMN[p]);
        accumulator=__fma_rn(s[(long long)row*sr+(long long)k*sc],
                             d[(long long)k*dr+(long long)col*dc],accumulator);
    }
    out[(long long)batch*N*N+row*N+col]=accumulator;
}

extern "C" __global__ void sparse_right(const double* s,const double* d,double* out,
    long long sb,long long sr,long long sc,
    long long db,long long dr,long long dc,int batches){
    const int batch=blockIdx.z;
    if(batch>=batches)return;
    const int tx=threadIdx.x,ty=threadIdx.y;
    const int row0=blockIdx.y*4,col=blockIdx.x*32+tx,row=row0+ty;
    s+=(long long)batch*sb;d+=(long long)batch*db;
    // Reuse each dense row across the32 output columns in its warp. Threads
    // outside the final output rows still participate in the shared barrier.
    __shared__ double dense_rows[4][N];
    for(int pos=ty*32+tx;pos<4*N;pos+=128){
        const int r=pos/N,k=pos%N;
        dense_rows[r][k]=(row0+r<N)?d[(long long)(row0+r)*dr+(long long)k*dc]:0.;
    }
    __syncthreads();
    if(row>=N || col>=N)return;
    double accumulator=0.;
    // CSR row col is also the structural column col. The value uses s[k,col]
    // with its own strides; no numerical symmetry is assumed, even for views.
    const int begin=__ldg(&SP_ROW_PTR[col]),end=__ldg(&SP_ROW_PTR[col+1]);
    for(int p=begin;p<end;p++){
        const int k=__ldg(&SP_COLUMN[p]);
        accumulator=__fma_rn(dense_rows[ty][k],
                             s[(long long)k*sr+(long long)col*sc],accumulator);
    }
    out[(long long)batch*N*N+row*N+col]=accumulator;
}
