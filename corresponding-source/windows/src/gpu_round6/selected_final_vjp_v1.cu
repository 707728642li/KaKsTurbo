// Complete parameter-Jacobian observation of the final FP64 T18 adjoint.
// Upstream adjoints remain full dense. Only the final587 observable entries
// use the fast path; uncertified rows execute every original final entry.
#define N 61
#define NN 3721
#define NT 587
__device__ __constant__ int ROW_PTR[62]={ROW_POINTER_VALUES};
__device__ __constant__ int COLUMN[587]={COLUMN_VALUES};
__device__ __constant__ int TARGET_I[587]={TARGET_I_VALUES};
__device__ __constant__ int TARGET_J[587]={TARGET_J_VALUES};
__device__ __constant__ int LOOKUP[3721]={LOOKUP_VALUES};

// Every instruction/order below matches the observed entry of the original
// dense64 product, sparse right/left CSR products, add_pair and add3_scale.
__device__ __forceinline__ double tail_entry(const double* gz,
 const double* ga3,const double* a2,const double* ga2,const double* z,
 const double* diagnostic,int row,int i,int j){
 const long long off=(long long)row*NN;
 const double* d=diagnostic+row*4;
 if(d[0]==0.)return 0.;
 double dense=0.;
 #pragma unroll
 for(int k=0;k<64;k++){
  double left=k<N?ga3[off+i*N+k]:0.;
  double right=k<N?a2[off+j*N+k]:0.;
  dense=__fma_rn(left,right,dense);
 }
 double gz_updated=gz[off+i*N+j]+dense;
 double right_product=0.;
 for(int p=ROW_PTR[j];p<ROW_PTR[j+1];p++){
  int k=COLUMN[p];
  right_product=__fma_rn(ga2[off+i*N+k],z[off+j*N+k],right_product);
 }
 double left_product=0.;
 for(int p=ROW_PTR[i];p<ROW_PTR[i+1];p++){
  int k=COLUMN[p];
  left_product=__fma_rn(z[off+k*N+i],ga2[off+k*N+j],left_product);
 }
 double scale=__longlong_as_double((unsigned long long)(1023-(int)d[3])<<52);
 return ((gz_updated+right_product)+left_product)*scale;
}

extern "C" __global__ void selected_row_certificate(const double* gz,
 const double* ga3,const double* a2,const double* ga2,const double* z,
 const double* s,const double* a,int* safe,int rows,int force_full){
 int row=blockIdx.x,tid=threadIdx.x;
 if(row>=rows)return;
 __shared__ double maxima[5][256];
 __shared__ int invalid[256];
 double m0=0.,m1=0.,m2=0.,m3=0.,m4=0.;int bad=0;
 long long off=(long long)row*NN;
 for(int p=tid;p<NN;p+=256){
  double v0=gz[off+p],v1=ga3[off+p],v2=a2[off+p],v3=ga2[off+p],v4=z[off+p];
  bad|=!isfinite(v0)||!isfinite(v1)||!isfinite(v2)||!isfinite(v3)||!isfinite(v4);
  m0=fmax(m0,fabs(v0));m1=fmax(m1,fabs(v1));m2=fmax(m2,fabs(v2));
  m3=fmax(m3,fabs(v3));m4=fmax(m4,fabs(v4));
  // Nonedges must be original exact positive-zero S/A. Any unexpected
  // support/NaN/Inf/signed-zero uses the original full contraction path.
  bad|=!isfinite(s[off+p])||!isfinite(a[off+p]);
  if(LOOKUP[p]<0)bad|=__double_as_longlong(s[off+p])!=0LL||__double_as_longlong(a[off+p])!=0LL;
 }
 maxima[0][tid]=m0;maxima[1][tid]=m1;maxima[2][tid]=m2;
 maxima[3][tid]=m3;maxima[4][tid]=m4;invalid[tid]=bad;__syncthreads();
 for(int step=128;step>0;step/=2){
  if(tid<step){
   #pragma unroll
   for(int k=0;k<5;k++)maxima[k][tid]=fmax(maxima[k][tid],maxima[k][tid+step]);
   invalid[tid]|=invalid[tid+step];
  }
  __syncthreads();
 }
 if(tid==0){
  // Upward-rounded bounds plus2x slack cover all64 FMAs, both <=10-term
  // sparse products, independently rounded intermediate additions, and
  // subnormal underflow. An inconclusive/overflowing bound selects fallback.
  double dense=__dmul_ru(128.,__dmul_ru(maxima[1][0],maxima[2][0]));
  double sparse=__dmul_ru(40.,__dmul_ru(maxima[3][0],maxima[4][0]));
  double bound=__dmul_ru(2.,__dadd_ru(__dadd_ru(maxima[0][0],dense),sparse));
  safe[row]=!force_full && !invalid[0] && isfinite(bound);
 }
}

extern "C" __global__ void selected_tail_packed(const double* gz,
 const double* ga3,const double* a2,const double* ga2,const double* z,
 const double* diagnostic,const int* safe,double* packed,int rows){
 int row=blockIdx.y,target=blockIdx.x*blockDim.x+threadIdx.x;
 if(row>=rows||target>=NT||!safe[row])return;
 packed[(long long)row*NT+target]=tail_entry(gz,ga3,a2,ga2,z,diagnostic,row,TARGET_I[target],TARGET_J[target]);
}

extern "C" __global__ void selected_tail_full_fallback(const double* gz,
 const double* ga3,const double* a2,const double* ga2,const double* z,
 const double* diagnostic,const int* safe,double* full,int rows){
 int row=blockIdx.y,p=blockIdx.x*blockDim.x+threadIdx.x;
 if(row>=rows||p>=NN||safe[row])return;
 full[(long long)row*NN+p]=tail_entry(gz,ga3,a2,ga2,z,diagnostic,row,p/N,p%N);
}

extern "C" __global__ void selected_contract_reduce(const double* packed,
 const double* full,const int* safe,const double* s,const double* a,
 const double* mr,double* contractions,int rows){
 int row=blockIdx.x,tid=threadIdx.x;
 if(row>=rows)return;
 __shared__ double sm[256],st[256];
 double dot_a=0.,dot_s=0.;
 long long off=(long long)row*NN;
 for(int p=tid;p<NN;p+=256){
  int target=LOOKUP[p];
  if(safe[row]&&target<0){
   dot_a=dot_a+0.;dot_s=dot_s+0.;
  }else{
   double g=safe[row]?packed[(long long)row*NT+target]:full[off+p];
   dot_a=dot_a+g*a[off+p];dot_s=dot_s+g*s[off+p];
  }
 }
 sm[tid]=dot_a;st[tid]=dot_s;__syncthreads();
 for(int step=128;step>0;step/=2){
  if(tid<step){sm[tid]=sm[tid]+sm[tid+step];st[tid]=st[tid]+st[tid+step];}
  __syncthreads();
 }
 if(tid==0){contractions[row*2]=st[0]/mr[row];contractions[row*2+1]=-sm[0]/mr[row];}
}

// The original all263-edge kernel is inserted verbatim except its g-address
// lookup, which selects packed587 or the complete3721 fallback row.
ORIGINAL_EDGE_KERNEL
