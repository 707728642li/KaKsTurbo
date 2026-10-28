// Structural code1 codon-graph products, including every diagonal.
// Sorted CSR is rendered from the exact immutable codon_edges() contract.
// Only structural zero primal terms are skipped; outputs/adjoints stay dense.
// Explicit IEEE FP64 FMA, matching the existing gemm32 k-order on retained k.
#define N 61
__device__ const int SP_ROW_PTR[62]={0,10,20,28,37,47,57,65,74,82,90,99,108,116,126,136,146,156,166,176,186,196,206,216,225,234,244,254,263,273,283,293,303,313,323,333,343,353,363,373,382,391,401,411,420,430,440,450,460,470,480,490,500,510,520,530,539,548,558,568,577,587};
__device__ const int SP_COLUMN[587]={0,1,2,3,4,8,10,13,29,45,0,1,2,3,5,9,11,14,30,46,0,1,2,3,6,15,31,47,0,1,2,3,7,12,16,32,48,0,4,5,6,7,8,10,17,33,49,1,4,5,6,7,9,11,18,34,50,2,4,5,6,7,19,35,51,3,4,5,6,7,12,20,36,52,0,4,8,9,10,21,37,53,1,5,8,9,11,22,38,54,0,4,8,10,11,12,25,41,57,1,5,9,10,11,12,26,42,58,3,7,10,11,12,28,44,60,0,13,14,15,16,17,21,25,29,45,1,13,14,15,16,18,22,26,30,46,2,13,14,15,16,19,23,27,31,47,3,13,14,15,16,20,24,28,32,48,4,13,17,18,19,20,21,25,33,49,5,14,17,18,19,20,22,26,34,50,6,15,17,18,19,20,23,27,35,51,7,16,17,18,19,20,24,28,36,52,8,13,17,21,22,23,24,25,37,53,9,14,18,21,22,23,24,26,38,54,15,19,21,22,23,24,27,39,55,16,20,21,22,23,24,28,40,56,10,13,17,21,25,26,27,28,41,57,11,14,18,22,25,26,27,28,42,58,15,19,23,25,26,27,28,43,59,12,16,20,24,25,26,27,28,44,60,0,13,29,30,31,32,33,37,41,45,1,14,29,30,31,32,34,38,42,46,2,15,29,30,31,32,35,39,43,47,3,16,29,30,31,32,36,40,44,48,4,17,29,33,34,35,36,37,41,49,5,18,30,33,34,35,36,38,42,50,6,19,31,33,34,35,36,39,43,51,7,20,32,33,34,35,36,40,44,52,8,21,29,33,37,38,39,40,41,53,9,22,30,34,37,38,39,40,42,54,23,31,35,37,38,39,40,43,55,24,32,36,37,38,39,40,44,56,10,25,29,33,37,41,42,43,44,57,11,26,30,34,38,41,42,43,44,58,27,31,35,39,41,42,43,44,59,12,28,32,36,40,41,42,43,44,60,0,13,29,45,46,47,48,49,53,57,1,14,30,45,46,47,48,50,54,58,2,15,31,45,46,47,48,51,55,59,3,16,32,45,46,47,48,52,56,60,4,17,33,45,49,50,51,52,53,57,5,18,34,46,49,50,51,52,54,58,6,19,35,47,49,50,51,52,55,59,7,20,36,48,49,50,51,52,56,60,8,21,37,45,49,53,54,55,56,57,9,22,38,46,50,53,54,55,56,58,23,39,47,51,53,54,55,56,59,24,40,48,52,53,54,55,56,60,10,25,41,45,49,53,57,58,59,60,11,26,42,46,50,54,57,58,59,60,27,43,47,51,55,57,58,59,60,12,28,44,48,52,56,57,58,59,60};

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
