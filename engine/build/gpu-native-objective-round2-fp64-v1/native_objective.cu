// Full FP64 likelihood and dense exponential adjoint, one block per request.
// Each positive endpoint pattern has one unique dense-weight cell. No atomics,
// clipping, probability floor, support truncation or missing stationary factor.
extern "C" __global__ void native_objective(const double* exponential,
 const double* pi,const double* sqrt_pi,const bool* live_outer,
 const long long* indices,const double* weights,double* objective,double* gexp,
 long long eb,long long er,long long ec,long long pb,long long pc,
 long long sb,long long sc,long long lb,long long lr,long long lc,
 long long ib,int rows,int pairs,int kind){
    const int row=blockIdx.x,tid=threadIdx.x;
    if(row>=rows)return;
    const long long slot=indices[(long long)row*ib];
    const bool valid_slot=slot>=0 && slot<(long long)pairs*14;
    const long long pair=valid_slot?slot/14:0;
    __shared__ double sums[256];
    double total=0.;
    for(int p=tid;p<3721;p+=256){
        const int i=p/61,j=p%61;
        double term=0.,gradient=0.;
        if(valid_slot){
            const double weight=weights[pair*3721+p];
            if(weight>=1e-6){
                const double pi_i=pi[(long long)row*pb+(long long)i*pc];
                const double sqrt_i=sqrt_pi[(long long)row*sb+(long long)i*sc];
                const double sqrt_j=sqrt_pi[(long long)row*sb+(long long)j*sc];
                const bool live=live_outer[(long long)row*lb+(long long)i*lr+(long long)j*lc];
                double transition;
                if(live){
                    transition=exponential[(long long)row*eb+(long long)i*er+(long long)j*ec]/sqrt_i;
                    transition=transition*sqrt_j;
                }else transition=(i==j)?1.:0.;
                const double fh=transition*pi_i;
                // Original safe=where(weight>=1e-6,fh,1). This branch is the
                // selected fh branch; all omitted terms have log(1)*weight=0.
                term=log(fh)*weight;
                if(kind && live){
                    // Exact original Torch chain, including its finite-range
                    // failure semantics; never simplify this to -weight/exp.
                    gradient=-weight/fh;
                    gradient=gradient*pi_i;
                    gradient=gradient*sqrt_j;
                    gradient=gradient/sqrt_i;
                }
            }
        }
        total=total+term;
        if(kind)gexp[(long long)row*3721+p]=gradient;
    }
    sums[tid]=total;__syncthreads();
    for(int step=128;step>0;step/=2){
        if(tid<step)sums[tid]=sums[tid]+sums[tid+step];
        __syncthreads();
    }
    if(tid==0)objective[row]=valid_slot?-sums[0]:__longlong_as_double(0x7ff8000000000000LL);
}
