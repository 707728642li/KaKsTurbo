
__device__ __constant__ double COEFF[25]={0.0,-0.10036558103014462,-0.00802924648241157,-0.00089213849804573,0.0,0.0,0.3978497494996451,1.3678377846041172,0.49828962252538267,-0.0006378981945947233,-10.967639605296206,1.680158138789062,0.05717798464788655,-0.0069821012248805206,3.349750170860705e-05,-0.09043168323908106,-0.06764045190713819,0.06759613017704597,0.029555257042931552,-1.391802575160607e-05,0.0,0.0,-0.09233646193671186,-0.016936493900208172,-1.400867981820361e-05};
extern "C" __global__ void t18_fold(const double* eye,const double* z,
 const double* a2,const double* a3,const double* a6,double* out,long long n){
 long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
 if(i>=n)return;
 double p0=eye[i%3721],p1=z[i],p2=a2[i],p3=a3[i],p4=a6[i];
 #pragma unroll
 for(int r=0;r<5;r++){
  double v=0.;v=v+p0*COEFF[r*5];v=v+p1*COEFF[r*5+1];
  v=v+p2*COEFF[r*5+2];v=v+p3*COEFF[r*5+3];v=v+p4*COEFF[r*5+4];
  out[(long long)r*n+i]=v;
 }
}
extern "C" __global__ void t18_fold_backward(const double* g0,const double* g1,
 const double* g2,const double* g3,const double* g4,double* out,long long n){
 long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
 if(i>=n)return;
 double a=g0[i],b=g1[i],c=g2[i],d=g3[i],e=g4[i];
 #pragma unroll
 for(int k=1;k<5;k++){
  double v=0.;v=v+a*COEFF[k];v=v+b*COEFF[5+k];v=v+c*COEFF[10+k];
  v=v+d*COEFF[15+k];v=v+e*COEFF[20+k];out[(long long)(k-1)*n+i]=v;
 }
}
