/* Independent translation unit: the unchanged, audited fiber host is reused.
 * Selection of this DLL explicitly selects the research refinement policy.
 */
#define kaks_fiber_close kaks_fiber_original_host_close
#include "fiber_optimizer.cpp"
#undef kaks_fiber_close
#include "fiber_optimizer_stable_bridge.h"
#include <map>

namespace {
enum { STABLE_DIAGNOSTIC_COLUMNS=17 };
struct StableDiagnostic {
    double values[STABLE_DIAGNOSTIC_COLUMNS];
    StableDiagnostic() { for(int i=0;i<STABLE_DIAGNOSTIC_COLUMNS;i++) values[i]=0; }
};
std::map<Task *,StableDiagnostic> stable_diagnostics;
bool stable_inside(const double *x,const double xb[][2],int np) {
    for(int i=0;i<np;i++) if(!_finite(x[i])||x[i]<xb[i][0]||x[i]>xb[i][1])return false;
    return true;
}
}

void kaks_fiber_stabilize(void *raw,double *f,double *x,double xb[][2],
                          double *space,double e,int np) {
    if(!current||current->model!=raw||np!=current->model->com.np||np<2||np>7)
        throw std::runtime_error("Unexpected optional refinement hook");
    GY94 *gy=static_cast<GY94 *>(raw);
    Task *t=current;
    if(t->oracle)return; // The explicitly named original-CPU oracle stays original.
    StableDiagnostic &diag=stable_diagnostics[t]; diag=StableDiagnostic();
    diag.values[1]=gy->Iround; diag.values[7]=diag.values[8]=*f;
    diag.values[14]=*f;
    diag.values[12]=static_cast<double>(t->requests);
    diag.values[13]=static_cast<double>(t->hits);
    const int primary_round=gy->Iround;
    const bool capped=primary_round>=2000;
    const bool repeated=t->hits>=64&&t->requests&&t->hits>8*t->requests;
    const bool outside=!stable_inside(x,xb,np);
    if(!capped&&!repeated&&!outside)return;
    if(!_finite(*f))throw std::runtime_error("Refinement requires a finite primary fit");
    diag.values[0]=(capped?1:0)+(repeated?2:0)+(outside?4:0);
    const double initial_f=*f;
    double best[7],candidate[7];
    std::memcpy(best,x,7*sizeof(double));
    double best_f=initial_f;
    if(outside) {
        // Original x0+alpha*p can land a few ulps beyond a box face. Repair
        // the actual vector and rescore its feasible baseline. Refinement must
        // not worsen that score. No tolerance exempts an out-of-box estimate.
        for(int i=0;i<np;i++) {
            if(!_finite(best[i]))throw std::runtime_error("Nonfinite primary parameter");
            if(best[i]<xb[i][0])best[i]=xb[i][0];
            if(best[i]>xb[i][1])best[i]=xb[i][1];
        }
        best_f=gy->lfun2dSdN(best,np);
        if(!_finite(best_f))throw std::runtime_error("Nonfinite projected objective");
    }
    // Compare refinement with an actually feasible baseline. An unbounded
    // source estimate may have a lower score than every nearby feasible point.
    const double feasible_f=best_f;
    diag.values[14]=feasible_f;diag.values[15]=feasible_f-initial_f;
    for(int pass=0;pass<2;pass++) {
        const double pass_f=best_f;
        // The same fixed policy applies to every model/pair/active parameter.
        // Bounds are taken directly from original PairwiseCodon, never from a
        // reference result. Faces are tested in parameter, lower/upper order.
        for(int parameter=0;parameter<np;parameter++) for(int side=0;side<2;side++) {
            if(best[parameter]==xb[parameter][side])continue;
            std::memcpy(candidate,best,7*sizeof(double));
            candidate[parameter]=xb[parameter][side];
            const double value=gy->lfun2dSdN(candidate,np); diag.values[2]++;
            if(_finite(value)&&value<best_f) {
                best_f=value;std::memcpy(best,candidate,7*sizeof(double));diag.values[3]++;
            }
        }
        // Re-enter the identical original BFGS. Its entry reconstructs the
        // working set and identity inverse Hessian from the chosen point.
        std::memcpy(candidate,best,7*sizeof(double));
        double value=best_f;
        const int result=gy->ming2(&value,candidate,xb,space,e,np);
        diag.values[4]++;diag.values[5]=gy->Iround;
        diag.values[6]+=gy->Iround;diag.values[11]=result;
        // A round-off displacement beyond a face is projected, then rescored.
        // The projected candidate is still subject to strict improvement.
        for(int i=0;i<np;i++) {
            if(candidate[i]<xb[i][0])candidate[i]=xb[i][0];
            if(candidate[i]>xb[i][1])candidate[i]=xb[i][1];
        }
        value=gy->lfun2dSdN(candidate,np);
        if(_finite(value)&&stable_inside(candidate,xb,np)&&value<best_f) {
            best_f=value;std::memcpy(best,candidate,7*sizeof(double));
        } else diag.values[10]++;
        if(best_f==pass_f)break;
    }
    std::memcpy(x,best,7*sizeof(double));
    // Even exact-cache hits restore raw kappa/omega, equal pi and parse once.
    // Original PairwiseCodon then performs its unchanged stats/output parses.
    *f=gy->lfun2dSdN(x,np);
    if(!_finite(*f)||*f>feasible_f||!stable_inside(x,xb,np))
        throw std::runtime_error("Refinement violated its objective/bounds invariant");
    diag.values[8]=*f;diag.values[9]=initial_f-*f;
    diag.values[16]=feasible_f-*f;
    gy->Iround=primary_round; // ABI status reports primary fit; extra rounds above.
}

extern "C" __declspec(dllexport) uint32_t kaks_fiber_stable_policy_version(void) {return 2;}
extern "C" __declspec(dllexport) uint32_t kaks_fiber_stable_diagnostic_columns(void) {return STABLE_DIAGNOSTIC_COLUMNS;}
extern "C" __declspec(dllexport) int kaks_fiber_stable_diagnostics(void *raw,uint32_t capacity,
    double *out,char *error,uint32_t error_capacity) {
    FIBER_BEGIN {
        Session *s=checked(raw);
        if(capacity<s->tasks.size()||!out)throw std::invalid_argument("Stable diagnostic capacity too small");
        for(size_t i=0;i<s->tasks.size();i++) {
            std::map<Task *,StableDiagnostic>::const_iterator it=stable_diagnostics.find(s->tasks[i]);
            if(s->tasks[i]->state!=DONE||it==stable_diagnostics.end())
                throw std::invalid_argument("Finish all models before stable diagnostics");
            std::memcpy(out+i*STABLE_DIAGNOSTIC_COLUMNS,it->second.values,STABLE_DIAGNOSTIC_COLUMNS*sizeof(double));
        }
    } FIBER_END;
}
extern "C" __declspec(dllexport) int kaks_fiber_close(void *raw,char *error,uint32_t error_capacity) {
    FIBER_BEGIN {
        Session *s=checked(raw);
        const std::vector<Task *> keys=s->tasks;
        const int code=kaks_fiber_original_host_close(raw,error,error_capacity);
        // The base close can destroy the session yet return an error from
        // ConvertFiberToThread. Clear addresses whenever destruction occurred.
        if(live!=s)for(size_t i=0;i<keys.size();i++)stable_diagnostics.erase(keys[i]);
        if(code)return code;
    } FIBER_END;
}
