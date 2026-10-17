/* Independent translation unit: the unchanged, audited fiber host is reused.
 * Selection of this DLL explicitly selects the research refinement policy.
 */
#define kaks_fiber_close kaks_fiber_original_host_close
#include "fiber_optimizer_stagnation_host_v8.cpp"
#undef kaks_fiber_close
#include "fiber_optimizer_stable_bridge.h"
#include "fiber_optimizer_stagnation_bridge.h"
#include <map>
#include <sstream>
#include "fiber_optimizer_protected_bridge.h"

namespace {
enum { STABLE_DIAGNOSTIC_COLUMNS=23 };
struct StableDiagnostic {
    double values[STABLE_DIAGNOSTIC_COLUMNS];
    StableDiagnostic() { for(int i=0;i<STABLE_DIAGNOSTIC_COLUMNS;i++) values[i]=0; }
};
std::map<Task *,StableDiagnostic> stable_diagnostics;
enum { POLISH_METRIC_COLUMNS=16 };
struct PolishMetric { double values[POLISH_METRIC_COLUMNS]; PolishMetric(){for(int i=0;i<POLISH_METRIC_COLUMNS;i++)values[i]=0.;} };
std::map<Task *,PolishMetric> polish_metrics;
bool stable_inside(const double *x,const double xb[][2],int np) {
    for(int i=0;i<np;i++) if(!_finite(x[i])||x[i]<xb[i][0]||x[i]>xb[i][1])return false;
    return true;
}
}

#include "fiber_optimizer_near_impl.inc"
#include "fiber_optimizer_zero_impl.inc"

void kaks_fiber_stabilize(void *raw,double *f,double *x,double xb[][2],
                          double *space,double e,int np) {
    if(!current||current->model!=raw||np!=current->model->com.np||np<2||np>7)
        throw std::runtime_error("Unexpected optional refinement hook");
    GY94 *gy=static_cast<GY94 *>(raw);
    Task *t=current;
    if(t->oracle)return; // The explicitly named original-CPU oracle stays original.
    near_metrics[t].refining=true;
    StableDiagnostic &diag=stable_diagnostics[t];
    const double primary_stagnation=diag.values[17],primary_stagnation_round=diag.values[19];
    diag=StableDiagnostic();diag.values[17]=primary_stagnation;diag.values[18]=primary_stagnation_round;diag.values[19]=primary_stagnation_round;diag.values[20]=64;
    diag.values[1]=gy->Iround; diag.values[7]=diag.values[8]=*f;
    diag.values[14]=*f;
    diag.values[12]=static_cast<double>(t->requests);
    diag.values[13]=static_cast<double>(t->hits);
    const int primary_round=gy->Iround;
    const bool capped=primary_round>=2000;
    const bool repeated=t->hits>=64&&t->requests&&t->hits>8*t->requests;
    const bool outside=!stable_inside(x,xb,np);
    const bool zero_seeded=zero_metrics[t].values[0]!=0.;
    const bool old_trigger=!zero_seeded&&(capped||repeated||outside||primary_stagnation);
    if(!_finite(*f))throw std::runtime_error("Refinement requires a finite primary fit");
    diag.values[0]=(capped?1:0)+(repeated?2:0)+(outside?4:0)+(primary_stagnation?8:0);
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
    kaks_fiber_pin_objective(raw,best,np,feasible_f,0);
    kaks_fiber_pin_objective(raw,best,np,best_f,1);
    if(old_trigger) for(int pass=0;pass<2;pass++) {
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
                kaks_fiber_pin_objective(raw,best,np,best_f,1);
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
            kaks_fiber_pin_objective(raw,best,np,best_f,1);
        } else diag.values[10]++;
        if(best_f==pass_f)break;
    }
    // All models receive the same finite-coordinate quality measurement.
    // Only improvements larger than the source e trigger one-dimensional
    // polishing; every accepted evaluation still uses a strict comparison.
    PolishMetric &polish=polish_metrics[t];polish=PolishMetric();
    const int max_rounds=64,max_expand=20,max_golden=50;
    polish.values[0]=1.;polish.values[1]=e;
    polish.values[13]=max_rounds;polish.values[14]=max_golden;polish.values[15]=max_expand;
    double probe_gain=0.;
    for(int polish_round=0;polish_round<=max_rounds;polish_round++) {
        double winner[7],origin[7];std::memcpy(origin,best,7*sizeof(double));
        const double origin_f=best_f;double winner_f=best_f;
        int winner_parameter=-1;probe_gain=0.;
        for(int parameter=0;parameter<np;parameter++) {
            const double fine=1e-6*(fabs(origin[parameter])+1.);
            for(int scale=0;scale<3;scale++)for(int side=0;side<2;side++) {
                std::memcpy(candidate,origin,7*sizeof(double));
                if(scale==0)candidate[parameter]=xb[parameter][side];
                else {
                    const double step=scale==1?fine:pow(fine,.67);
                    candidate[parameter]+=(side?step:-step);
                    if(candidate[parameter]<xb[parameter][0])candidate[parameter]=xb[parameter][0];
                    if(candidate[parameter]>xb[parameter][1])candidate[parameter]=xb[parameter][1];
                }
                if(candidate[parameter]==origin[parameter])continue;
                const double value=gy->lfun2dSdN(candidate,np);polish.values[8]++;
                if(_finite(value)&&value<winner_f) {
                    winner_f=value;winner_parameter=parameter;
                    std::memcpy(winner,candidate,7*sizeof(double));
                }
            }
        }
        probe_gain=origin_f-winner_f;
        if(polish_round==0)polish.values[2]=probe_gain;
        if(probe_gain<=e) {polish.values[4]=1.;break;}
        diag.values[0]=static_cast<int>(diag.values[0])|16;
        if(polish_round==max_rounds) {polish.values[11]=1.;break;}
        if(winner_parameter<0)throw std::runtime_error("Missing improving polish coordinate");
        // Verify/rescore the selected point immediately before pinning its
        // objective/warning pair. A provider variation can never worsen best.
        std::memcpy(candidate,winner,7*sizeof(double));
        double value=gy->lfun2dSdN(candidate,np);polish.values[8]++;
        if(!_finite(value)||!stable_inside(candidate,xb,np)||value>=best_f) {
            polish.values[12]++;break;
        }
        best_f=value;std::memcpy(best,candidate,7*sizeof(double));
        kaks_fiber_pin_objective(raw,best,np,best_f,1);polish.values[9]++;
        polish.values[5]++;polish.values[10]++;
        const int parameter=winner_parameter;
        double previous=origin[parameter],current_point=best[parameter],current_f=best_f;
        double left=previous<current_point?previous:current_point;
        double right=previous>current_point?previous:current_point;
        const double bound=current_point>previous?xb[parameter][1]:xb[parameter][0];
        // The origin's other coordinates remain fixed during this search.
        // Every evaluated feasible improvement is accepted and pinned at once.
#define POLISH_EVALUATE(destination) do { \
        destination=gy->lfun2dSdN(candidate,np); \
        if(_finite(destination)&&stable_inside(candidate,xb,np)&&destination<best_f) { \
            best_f=destination;std::memcpy(best,candidate,7*sizeof(double)); \
            kaks_fiber_pin_objective(raw,best,np,best_f,1);polish.values[9]++; \
        } } while(0)
        for(int expansion=0;expansion<max_expand&&current_point!=bound;expansion++) {
            double next=origin[parameter]+2.*(current_point-origin[parameter]);
            if(next<xb[parameter][0])next=xb[parameter][0];
            if(next>xb[parameter][1])next=xb[parameter][1];
            if(next==current_point)break;
            std::memcpy(candidate,origin,7*sizeof(double));candidate[parameter]=next;
            double next_f;POLISH_EVALUATE(next_f);polish.values[6]++;
            left=previous<next?previous:next;right=previous>next?previous:next;
            if(!_finite(next_f)||next_f>=current_f)break;
            previous=current_point;current_point=next;current_f=next_f;
        }
        if(left<right) {
            const double phi=.6180339887498948482;
            double c=right-phi*(right-left),d=left+phi*(right-left),fc,fd;
            std::memcpy(candidate,origin,7*sizeof(double));candidate[parameter]=c;
            POLISH_EVALUATE(fc);polish.values[7]++;
            std::memcpy(candidate,origin,7*sizeof(double));candidate[parameter]=d;
            POLISH_EVALUATE(fd);polish.values[7]++;
            for(int golden=0;golden<max_golden;golden++) {
                if(right-left<=1e-12*(1.+fabs(left)+fabs(right)))break;
                if(_finite(fc)&&(!_finite(fd)||fc<fd)) {
                    right=d;d=c;fd=fc;c=right-phi*(right-left);
                    std::memcpy(candidate,origin,7*sizeof(double));candidate[parameter]=c;
                    POLISH_EVALUATE(fc);
                } else {
                    left=c;c=d;fc=fd;d=left+phi*(right-left);
                    std::memcpy(candidate,origin,7*sizeof(double));candidate[parameter]=d;
                    POLISH_EVALUATE(fd);
                }
                polish.values[7]++;
            }
        }
#undef POLISH_EVALUATE
    }
    polish.values[3]=probe_gain;
    diag.values[21]=probe_gain;diag.values[22]=probe_gain==0.?1.:0.;
    std::memcpy(x,best,7*sizeof(double));
    // Even exact-cache hits restore raw kappa/omega, equal pi and parse once.
    // Original PairwiseCodon then performs its unchanged stats/output parses.
    *f=gy->lfun2dSdN(x,np);
    if(!_finite(*f)||*f>feasible_f||!stable_inside(x,xb,np)) {
        std::ostringstream detail;detail.precision(17);
        detail<<"Refinement violated its objective/bounds invariant: slot="<<t->slot
              <<" pair="<<t->pair->id<<" model="<<MODELS[t->model_id]<<" np="<<np
              <<" trigger="<<diag.values[0]<<" primary_round="<<primary_round
              <<" initial_F="<<initial_f<<" feasible_F="<<feasible_f
              <<" accepted_best_F="<<best_f<<" restored_F="<<*f
              <<" restored_minus_feasible="<<(*f-feasible_f)
              <<" finite="<<(_finite(*f)!=0)<<" inside="<<stable_inside(x,xb,np);
        for(int i=0;i<np;i++)detail<<" x"<<i<<"="<<x[i]<<" box=["<<xb[i][0]<<","<<xb[i][1]<<"]";
        throw std::runtime_error(detail.str());
    }
    diag.values[8]=*f;diag.values[9]=initial_f-*f;
    diag.values[16]=feasible_f-*f;
    gy->Iround=primary_round; // ABI status reports primary fit; extra rounds above.
}

bool kaks_fiber_record_stagnation(void *raw,int round,int streak) {
    if(!current||current->model!=raw||streak<64)throw std::runtime_error("Invalid stagnation hook");
    if(current->oracle)return false;
    StableDiagnostic &diag=stable_diagnostics[current];
    diag.values[17]++;diag.values[19]=round;diag.values[20]=streak;return true;
}
extern "C" __declspec(dllexport) uint32_t kaks_fiber_stable_policy_version(void) {return 3;}
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
extern "C" __declspec(dllexport) uint32_t kaks_fiber_polish_metric_columns(void) {return POLISH_METRIC_COLUMNS;}
extern "C" __declspec(dllexport) int kaks_fiber_polish_metrics(void *raw,uint32_t capacity,double *out,char *error,uint32_t error_capacity) {
    FIBER_BEGIN {
        Session *s=checked(raw);if(capacity<s->tasks.size()||!out)throw std::invalid_argument("Polish metric capacity too small");
        for(size_t i=0;i<s->tasks.size();i++) {
            std::map<Task *,PolishMetric>::const_iterator it=polish_metrics.find(s->tasks[i]);
            if(s->tasks[i]->state!=DONE||it==polish_metrics.end())throw std::invalid_argument("Finish all models before polish metrics");
            std::memcpy(out+i*POLISH_METRIC_COLUMNS,it->second.values,POLISH_METRIC_COLUMNS*sizeof(double));
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
        if(live!=s)for(size_t i=0;i<keys.size();i++){stable_diagnostics.erase(keys[i]);polish_metrics.erase(keys[i]);near_metrics.erase(keys[i]);zero_metrics.erase(keys[i]);}
        if(code)return code;
    } FIBER_END;
}
