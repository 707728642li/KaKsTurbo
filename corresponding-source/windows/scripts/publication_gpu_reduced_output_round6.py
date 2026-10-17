"""Exact observable output product for final budget0 T18 forward evaluation.

All upstream matrices and reverse operations retain the tested v2 tape. Only
the last budget0 D@C output computes prepared observed cells; every selected
cell uses all64 original FP64 FMAs. Later squaring always uses full GEMM.
"""
from __future__ import annotations
import argparse
import hashlib
import inspect
import json
from pathlib import Path

from gpu_observed_output_round6 import ObservedOutputRuntime,build_observed_output
from publication_gpu_gradient import GPUGradientError
from publication_gpu_native_driver_round3 import sha
from publication_gpu_reduced_shared_round6 import (PublicationGPUGradient as PackedProvider,
    SharedObservedSelectedT18Runtime,static_tape_contract as packed_static_tape_contract)

SOURCE=Path(__file__).resolve()
SCHEMA='kaksforge.observed_output_packed_structural_vjp_gradient_round6/3'
ENGINE='CUDA Driver API; single context exact observed-output plus packed observed-adjoint and structural587 VJP; cold-result reuse; no Torch import; no CUDA timing events'
CANDIDATE='observed_output_packed_plus_structural587'
EXPECTED_PARENT_METHOD_SHA256='da9458e795a50437411cf222daaf3c1e0915554bd8a39e61837481f9efffc526'

class OutputObservedSelectedT18Runtime(SharedObservedSelectedT18Runtime):
    def forward_gradient(self,indices,x,budget,kind):
        if kind not in (0,1) or not isinstance(budget,int) or not 0<=budget<=16:raise ValueError('Invalid native graph kind/budget')
        H=self.helpers;mm=self.gemm.product;left=self.sparse.product_left;right=self.sparse.product_right
        a,pi,sqrt_pi,live,s,mr,rates=self.matrix.forward(indices,x)
        z,diagnostic=H.norm_scale(a,budget)
        a2=left(z,z);a3=left(z,a2);a6=mm(a3,a3)
        b0,b1,b2,b3,b4=self.fold(z,a2,a3,a6)
        c=H.add(b3,mm(b0,b4));d=H.add(b2,c)
        final_product=self.output.product(d,c,indices) if budget==0 else mm(d,c)
        result=H.add(b1,final_product)
        tape=[result]
        for step in range(budget):
            result=H.square_select(result,mm(result,result),diagnostic,step);tape.append(result)
        exp=H.valid_exp(result,diagnostic)
        F,g_exp=self.objective.evaluate(indices,exp,pi,sqrt_pi,live,kind)
        if not kind:return H.pack(F,diagnostic,None,x,indices,self.owner.nkappa,kind)
        g=H.mask_gradient(g_exp,diagnostic)
        for step in range(budget-1,-1,-1):
            square_g=H.mask_gradient(g,diagnostic,step,1);previous_t=tape[step].transpose(-1,-2)
            g=H.reverse_square(mm(square_g,previous_t),mm(previous_t,square_g),g,diagnostic,step)
        if budget==0:
            gb2=self.observed.product_left(g,c.transpose(-1,-2),indices)
            gb3=H.add(gb2,self.observed.product_right(g,d.transpose(-1,-2),indices))
        else:
            gb2=mm(g,c.transpose(-1,-2));gb3=H.add(gb2,mm(d.transpose(-1,-2),g))
        gb0=mm(gb3,b4.transpose(-1,-2));gb4=mm(b0.transpose(-1,-2),gb3)
        gz,ga2,ga3,ga6=self.fold_backward(gb0,g,gb2,gb3,gb4)
        a3t=a3.transpose(-1,-2)
        ga3=H.add3(ga3,mm(ga6,a3t),mm(a3t,ga6))
        zt=z.transpose(-1,-2)
        ga2=H.add(ga2,left(zt,ga3))
        raw=self.matrix.reverse.finish(gz,ga3,a2,ga2,z,diagnostic,s,a,pi,sqrt_pi,mr,rates,x,indices)
        return H.pack(F,diagnostic,raw,x,indices,self.owner.nkappa,kind)

    def manifest(self):
        result=super().manifest()
        result.update(schema='kaksforge.T18_observed_output_packed_structural_vjp_round6/3',
            observed_output=self.output.manifest(),observed_output_tape_contract=static_tape_contract(),
            all_forward_storage_full61=True,upstream_forward_values_full61=True,
            final_budget0_product_unobserved_cells_zero=True,
            budget_positive_full_final_forward_GEMM=True,
            all_reverse_statements_unchanged_from_packed_v2=True,
            mathematical_public_outputs_preserved_requires_GPU_gate=True)
        return result

def static_tape_contract():
    parent_contract=packed_static_tape_contract()
    parent=inspect.getsource(SharedObservedSelectedT18Runtime.forward_gradient)
    parent_sha=hashlib.sha256(parent.encode()).hexdigest()
    old='        c=H.add(b3,mm(b0,b4));d=H.add(b2,c);result=H.add(b1,mm(d,c))\n'
    new=('        c=H.add(b3,mm(b0,b4));d=H.add(b2,c)\n'
         '        final_product=self.output.product(d,c,indices) if budget==0 else mm(d,c)\n'
         '        result=H.add(b1,final_product)\n')
    actual=inspect.getsource(OutputObservedSelectedT18Runtime.forward_gradient)
    if parent_sha!=EXPECTED_PARENT_METHOD_SHA256 or parent.count(old)!=1 or actual!=parent.replace(old,new):
        raise RuntimeError('Observed-output tape changed statements outside final budget0 forward product')
    return dict(PASS=True,expected_parent_method_SHA256=EXPECTED_PARENT_METHOD_SHA256,
        parent_method_normalized_SHA256=parent_sha,new_method_normalized_SHA256=hashlib.sha256(actual.encode()).hexdigest(),
        immutable_packed_parent_contract=parent_contract,only_budget0_last_forward_product_replaced=True,
        all_other_statements_byte_identical=True,all_upstream_and_reverse_matrices_full61=True,
        budget_positive_full_original_final_product=True,kind0_and_kind1_same_observed_support=True,
        selected_output_full_ascending64_FP64_FMA=True,unobserved_product_cells_zero=True,
        budget0_tape_not_read_by_reverse=True,objective_reads_only_prepared_weight_ge_1e_minus6=True,
        valid_exp_is_cellwise_guard_without_norm_reduction=True,
        all14_models_full263_edges_and_mr=True,original_scaling_guards_and_optimizer=True,
        actual_GPU_public_parity_and_fullfit_still_required=True)

class PublicationGPUGradient(PackedProvider):
    def __init__(self,*args,observed_layout='original',force_full_output=False,**kwargs):
        if type(force_full_output) is not bool:raise ValueError('force_full_output must be explicit bool')
        self.output_runtime=None;self.output_installed_before_requests=False
        self.output_static_contract=static_tape_contract();build_observed_output()
        self.output_provider_source_before=sha(SOURCE)
        self.output_candidate=CANDIDATE if observed_layout=='original' else 'observed_output_shared_plus_structural587'
        super().__init__(*args,observed_layout=observed_layout,**kwargs)
        try:
            if self.requests or self.waves or self.gradient_buckets or self.api._graphs:
                raise RuntimeError('Observed-output install requires a fresh provider')
            self._enter()
            try:
                with self.api.cuda.device(self.device):
                    self.output_runtime=ObservedOutputRuntime.__new__(ObservedOutputRuntime)
                    self.output_runtime.__init__(self.api,self.device,self.observed_runtime,force_full=force_full_output)
                    self.native_runtime.output=self.output_runtime
                    self.native_runtime.__class__=OutputObservedSelectedT18Runtime
                    self.output_installed_before_requests=True
            finally:self._public_lock.release()
        except BaseException:self._close_resources(constructor_failure=True);raise

    def _close_resources(self,constructor_failure=False):
        output=getattr(self,'output_runtime',None)
        if getattr(self,'api',None) is not None and output is not None:
            self._drain()
            for key,bucket in list(self.gradient_buckets.items()):
                self._close_bucket(bucket);del self.gradient_buckets[key]
            while self._orphan_buckets:
                self._close_bucket(self._orphan_buckets[-1]);self._orphan_buckets.pop()
            module=getattr(output,'module',None)
            if module is not None and module.value:
                with self.api.cuda.device(self.device):output.close()
        super()._close_resources(constructor_failure=constructor_failure)

    def graph_manifest(self):
        result=super().graph_manifest()
        if sha(SOURCE)!=self.output_provider_source_before or not self.output_installed_before_requests:
            raise RuntimeError('Observed-output provider source/installation changed')
        result.update(schema=SCHEMA,reduced_candidate=self.output_candidate,
            observed_output_installed_before_requests=True,observed_output_runtime=self.output_runtime.manifest(),
            observed_output_tape_contract=self.output_static_contract,provider_source_sha256=self.output_provider_source_before,
            upstream_packed_provider_source_sha256=self.reduced_source_before,
            CUDA_timing_events_used=False,diagnostic_execution_only=False,
            optimizer_and_model_equations_changed=False,science_or_optimizer_changed=False,
            precision='FP64 full upstream/reverse; budget0 observed final-forward cells retain all64 original FMAs',
            required_science_gates='Observed-output finite/nonfinite/layout/forced-full primitives, HP28, partitions, complete1000, fresh CPU endpoints')
        return result

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--verify-cached',action='store_true');args=parser.parse_args()
    report=dict(PASS=True,schema=SCHEMA,candidate=CANDIDATE,engine=ENGINE,static_contract=static_tape_contract(),GPU_context_created=False)
    if args.verify_cached:
        ptx,metadata=build_observed_output();report.update(observed_output_PTX_sha256=metadata['ptx_sha256'],observed_output_PTX_bytes=len(ptx))
    print(json.dumps(report))

if __name__=='__main__':main()
