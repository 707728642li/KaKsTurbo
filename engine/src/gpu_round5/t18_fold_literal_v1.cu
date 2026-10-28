// Every original FP64 multiply/add retained in its original left-fold order.
extern "C" __global__ void t18_fold(const double* eye,const double* z,
 const double* a2,const double* a3,const double* a6,double* out,long long n){
 long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
 if(i>=n)return;
 double p0=eye[i%3721],p1=z[i],p2=a2[i],p3=a3[i],p4=a6[i];
 {double v=0.;
  v=v+p0*(0.0);
  v=v+p1*(-0.10036558103014462);
  v=v+p2*(-0.00802924648241157);
  v=v+p3*(-0.00089213849804573);
  v=v+p4*(0.0);
  out[(long long)0*n+i]=v;}
 {double v=0.;
  v=v+p0*(0.0);
  v=v+p1*(0.3978497494996451);
  v=v+p2*(1.3678377846041172);
  v=v+p3*(0.49828962252538267);
  v=v+p4*(-0.0006378981945947233);
  out[(long long)1*n+i]=v;}
 {double v=0.;
  v=v+p0*(-10.967639605296206);
  v=v+p1*(1.680158138789062);
  v=v+p2*(0.05717798464788655);
  v=v+p3*(-0.0069821012248805206);
  v=v+p4*(3.349750170860705e-05);
  out[(long long)2*n+i]=v;}
 {double v=0.;
  v=v+p0*(-0.09043168323908106);
  v=v+p1*(-0.06764045190713819);
  v=v+p2*(0.06759613017704597);
  v=v+p3*(0.029555257042931552);
  v=v+p4*(-1.391802575160607e-05);
  out[(long long)3*n+i]=v;}
 {double v=0.;
  v=v+p0*(0.0);
  v=v+p1*(0.0);
  v=v+p2*(-0.09233646193671186);
  v=v+p3*(-0.016936493900208172);
  v=v+p4*(-1.400867981820361e-05);
  out[(long long)4*n+i]=v;}
}
extern "C" __global__ void t18_fold_backward(const double* g0,const double* g1,
 const double* g2,const double* g3,const double* g4,double* out,long long n){
 long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
 if(i>=n)return;
 double a=g0[i],b=g1[i],c=g2[i],d=g3[i],e=g4[i];
 {double v=0.;
  v=v+a*(-0.10036558103014462);
  v=v+b*(0.3978497494996451);
  v=v+c*(1.680158138789062);
  v=v+d*(-0.06764045190713819);
  v=v+e*(0.0);
  out[(long long)0*n+i]=v;}
 {double v=0.;
  v=v+a*(-0.00802924648241157);
  v=v+b*(1.3678377846041172);
  v=v+c*(0.05717798464788655);
  v=v+d*(0.06759613017704597);
  v=v+e*(-0.09233646193671186);
  out[(long long)1*n+i]=v;}
 {double v=0.;
  v=v+a*(-0.00089213849804573);
  v=v+b*(0.49828962252538267);
  v=v+c*(-0.0069821012248805206);
  v=v+d*(0.029555257042931552);
  v=v+e*(-0.016936493900208172);
  out[(long long)2*n+i]=v;}
 {double v=0.;
  v=v+a*(0.0);
  v=v+b*(-0.0006378981945947233);
  v=v+c*(3.349750170860705e-05);
  v=v+d*(-1.391802575160607e-05);
  v=v+e*(-1.400867981820361e-05);
  out[(long long)3*n+i]=v;}
}
