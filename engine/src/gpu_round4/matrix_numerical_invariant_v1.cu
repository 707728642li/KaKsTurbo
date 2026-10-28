// All-FP64 sparse Q construction with dense symmetric S/A output.
// Same original per-edge multiplies, full (unmasked) rs and ra reductions.
// Round4 uses the row0/x32 mean-rate addition schedule for EVERY request.
// The schedule is independent of row ordinal, graph bucket, and partition.
// Real-number formulas and every per-edge FP64 multiply remain unchanged.
__device__ __constant__ int CODE[3721]={EDGE_CODE_VALUES};
__device__ __constant__ int EI[263]={EDGE_I_VALUES};
__device__ __constant__ int EJ[263]={EDGE_J_VALUES};
__device__ __constant__ int ER[263]={EDGE_R_VALUES};
__device__ __constant__ int ES[263]={EDGE_SYN_VALUES};

__device__ __forceinline__ double offdiag_q(int i,int j,const double* pi,
 const double* rates,double omega){
    int code=CODE[i*61+j];
    if(code<0)return 0.;
    double multiplier=code<6?1.:omega;
    return (rates[code%6]*pi[j])*multiplier;
}
__device__ __forceinline__ void mr_term(int edge,const double* pi,
 const double* rates,double omega,double &rs,double &ra){
    double lower=rates[ER[edge]]*pi[EJ[edge]];
    double pij=(2.*pi[EI[edge]])*lower;
    rs=ES[edge]?pij:0.;ra=ES[edge]?0.:pij*omega;
}
extern "C" __global__ void matrix_forward_prepare(const double* all_pi,
 const double* x,const long long* indices,const long long* nkappa,
 const long long* rate_map,double* pi,double* sqrt_pi,double* mr,
 double* rates,double* diagonal,int rows,int pi_stride){
    int row=blockIdx.x,tid=threadIdx.x,lane=tid%32,warp=tid/32;
    if(row>=rows)return;
    long long slot=indices[row];int nk=(int)nkappa[slot];
    double omega=x[row*7+nk+1];
    __shared__ double pr[61],rr[6],rs_shared[64],ra_shared[64];
    if(tid<61){
        double p=all_pi[slot*pi_stride+tid];pr[tid]=p;
        pi[row*61+tid]=p;sqrt_pi[row*61+tid]=p>0.?__dsqrt_rn(p):1.;
    }
    if(tid<6){
        int mapped=(int)rate_map[slot*6+tid];
        double rate=mapped<nk?x[row*7+mapped+1]:1.;
        rr[tid]=rate;rates[row*6+tid]=rate;
    }
    __syncthreads();
    // Fixed 263-edge mean-rate reduction. Preserve old row0/x32 operation
    // order for EVERY row: no dependence on row count, ordinal, or address.
    // This deliberately removes the old Torch alignment-emulation schedule.
    const int width=32;
    double rs=0.,ra=0.;
    if(tid<width){
        double sv[4]={0.,0.,0.,0.},av[4]={0.,0.,0.,0.};
        const int shift=0,head=0;
        if(shift && tid>=shift && tid<4)mr_term(tid-shift,pr,rr,omega,sv[0],av[0]);
        int end=263-head;
        for(int idx=tid;idx*4+3<end;idx+=width){
            #pragma unroll
            for(int k=0;k<4;k++){
                double syn,nonsyn;mr_term(head+idx*4+k,pr,rr,omega,syn,nonsyn);
                sv[k]=sv[k]+syn;av[k]=av[k]+nonsyn;
            }
        }
        int tail=end-end%4;
        if(tail+tid<end){
            double syn,nonsyn;mr_term(head+tail+tid,pr,rr,omega,syn,nonsyn);
            sv[0]=sv[0]+syn;av[0]=av[0]+nonsyn;
        }
        rs=sv[0];ra=av[0];
        #pragma unroll
        for(int k=1;k<4;k++){rs=rs+sv[k];ra=ra+av[k];}
        rs_shared[tid]=rs;ra_shared[tid]=ra;
    }
    __syncthreads();
    if(tid<32){
        if(width==64){rs=rs+rs_shared[tid+32];ra=ra+ra_shared[tid+32];}
        #pragma unroll
        for(int offset=1;offset<32;offset*=2){
            rs=rs+__shfl_down_sync(0xffffffff,rs,offset);
            ra=ra+__shfl_down_sync(0xffffffff,ra,offset);
        }
        if(tid==0)mr[row]=rs+ra;
    }
    // 61-wide Q row sum uses lane k + lane k+32 then ascending warp
    // shuffle1,2,4,8,16. Every absent edge and original Q diagonal is zero.
    for(int i=warp;i<61;i+=8){
        double v=offdiag_q(i,lane,pr,rr,omega);
        double second=lane+32<61?offdiag_q(i,lane+32,pr,rr,omega):0.;
        v=(0.+v)+(0.+second);v=v+0.;v=v+0.;
        #pragma unroll
        for(int offset=1;offset<32;offset*=2)v=v+__shfl_down_sync(0xffffffff,v,offset);
        if(lane==0)diagonal[row*61+i]=-v;
    }
}
extern "C" __global__ void matrix_forward_dense(const double* pi,
 const double* sqrt_pi,const double* mr,const double* rates,const double* diagonal,
 const double* x,const long long* indices,const long long* nkappa,
 double* symmetric,double* a,bool* live_outer,int rows){
    long long p=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(p>=(long long)rows*3721)return;
    int row=(int)(p/3721),local=(int)(p%3721),i=local/61,j=local%61;
    const double* pr=pi+(long long)row*61;const double* sr=sqrt_pi+(long long)row*61;
    bool live=pr[i]>0. && pr[j]>0.;live_outer[p]=live;
    // Only a structural nonedge with finite positive factors may skip its
    // exactly +0 result. Invalid support/time/mr and time0 keep old arithmetic.
    const double time=x[row*7],mean_rate=mr[row];
    const bool safe_support=isfinite(pr[i]) && isfinite(pr[j]) && pr[i]>=0. && pr[j]>=0.
        && isfinite(sr[i]) && isfinite(sr[j]) && sr[i]>0. && sr[j]>0.
        && (pr[i]>0. || sr[i]==1.) && (pr[j]>0. || sr[j]==1.);
    if(i!=j && CODE[i*61+j]<0 && safe_support && isfinite(time) && time>0.
        && isfinite(mean_rate) && mean_rate>0.){
        symmetric[p]=0.;a[p]=0.;return;
    }
    double value=0.;
    if(live){
        if(i==j)value=__dadd_rn(__dadd_rn(0.,0.),diagonal[row*61+i]);
        else{
            int lower_i=i>j?i:j,lower_j=i>j?j:i;
            double omega=x[row*7+(int)nkappa[indices[row]]+1];
            double q=offdiag_q(lower_i,lower_j,pr,rates+(long long)row*6,omega);
            value=(q*sr[lower_i])/sr[lower_j];
        }
    }
    symmetric[p]=value;a[p]=(value*x[row*7])/mr[row];
}
