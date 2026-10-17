// Experimental reverse Jacobian for the unchanged pure-rate Q -> symmetric A.
// Two stages: full dense quotient contractions, then all codon-edge derivatives.
// Input/output and every accumulation are double. Compile without contraction.
extern "C" __global__ void matrix_contract_reduce(const double* g,
 const double* s,const double* a,const double* mr,double* contractions,int rows) {
    int row=blockIdx.x,tid=threadIdx.x;
    if(row>=rows)return;
    __shared__ double sm[256],st[256];
    double dot_a=0.,dot_s=0.;
    long long offset=(long long)row*3721;
    for(int p=tid;p<3721;p+=256){
        dot_a=dot_a+g[offset+p]*a[offset+p];
        dot_s=dot_s+g[offset+p]*s[offset+p];
    }
    sm[tid]=dot_a;st[tid]=dot_s;__syncthreads();
    for(int step=128;step>0;step/=2){
        if(tid<step){sm[tid]=sm[tid]+sm[tid+step];st[tid]=st[tid]+st[tid+step];}
        __syncthreads();
    }
    if(tid==0){
        contractions[row*2]=st[0]/mr[row];
        contractions[row*2+1]=-sm[0]/mr[row];
    }
}
extern "C" __global__ void matrix_contract_edges(const double* g,
 const double* pi,const double* sqrt_pi,const double* mr,const double* rates,
 const double* x,const long long* indices,const long long* nkappa,
 const long long* rate_map,const long long* edge_i,const long long* edge_j,
 const long long* edge_rate,const bool* edge_syn,const double* contractions,
 double* gradient,int rows,int edges){
    int row=blockIdx.x,tid=threadIdx.x;
    if(row>=rows)return;
    long long slot=indices[row];int nk=(int)nkappa[slot];
    double time=x[row*7],omega=x[row*7+nk+1],mean_rate=mr[row];
    double mradj=contractions[row*2+1];
    const double* gr=g+(long long)row*3721;
    const double* pr=pi+(long long)row*61;
    const double* sr=sqrt_pi+(long long)row*61;
    __shared__ double sums[6][256];
    double acc[6]={0.,0.,0.,0.,0.,0.};
    for(int edge=tid;edge<edges;edge+=256){
        int i=(int)edge_i[edge],j=(int)edge_j[edge],r=(int)edge_rate[edge];
        double pii=pr[i],pij=pr[j];
        double qlower=0.;
        if(pii>0. && pij>0.)
            qlower=(((gr[i*61+j]+gr[j*61+i])*sr[i])/sr[j])*time/mean_rate;
        double di=(pii>0.)?gr[i*61+i]*time/mean_rate:0.;
        double dj=(pij>0.)?gr[j*61+j]*time/mean_rate:0.;
        double base=(qlower-di)*pij-dj*pii;
        bool syn=edge_syn[edge];double multiplier=syn?1.:omega;
        int mapped=(int)rate_map[slot*6+r];
        if(mapped<nk){
            double contribution=base*multiplier+(mradj*((2.*pii)*pij))*multiplier;
            acc[mapped]=acc[mapped]+contribution;
        }
        if(!syn){
            double rate=rates[row*6+r];
            double contribution=base*rate+(mradj*(2.*pii))*(rate*pij);
            acc[nk]=acc[nk]+contribution;
        }
    }
    #pragma unroll
    for(int k=0;k<6;k++)sums[k][tid]=acc[k];
    __syncthreads();
    for(int step=128;step>0;step/=2){
        if(tid<step){
            #pragma unroll
            for(int k=0;k<6;k++)sums[k][tid]=sums[k][tid]+sums[k][tid+step];
        }
        __syncthreads();
    }
    if(tid==0){
        gradient[row*7]=contractions[row*2];
        #pragma unroll
        for(int k=0;k<6;k++)gradient[row*7+k+1]=(k<=nk)?sums[k][0]:0.;
    }
}
