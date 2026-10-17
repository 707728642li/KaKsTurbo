"""Strict FP64 observed first-adjoint contractions plus selected final VJP.

All original forward matrices and upstream adjoints remain full61x61. Only the
first2 budget0 reverse products exploit the exact prepared objective support;
the inherited selected final VJP still contracts all263 edges and full mr. No
CUDA context/compilation, Torch, fitting, or diagnostics execute on import.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path

from gpu_observed_adjoint_round6 import ObservedAdjointRuntime, build_observed_adjoint
from publication_gpu_gradient import GPUGradientError
from publication_gpu_native_driver_round3 import sha
from publication_gpu_selected_round6 import (
    PublicationGPUGradient as SelectedProvider,
    SelectedTailT18Runtime,
    static_tape_contract as selected_static_tape_contract,
)

SOURCE=Path(__file__).resolve()
SCHEMA='kaksforge.observed_structural_vjp_gradient_round6/1'
ENGINE='CUDA Driver API; single context exact observed-adjoint plus structural587 VJP; cold-result reuse; no Torch import; no CUDA timing events'
CANDIDATE='observed_plus_structural587'
EXPECTED_PARENT_METHOD_SHA256='03b57f2cb5e86f5a8aecc2667c0088b7a47ff771571025ce174cfb0c90123722'

class ObservedSelectedT18Runtime(SelectedTailT18Runtime):
    def forward_gradient(self,indices,x,budget,kind):
        if kind not in (0,1) or not isinstance(budget,int) or not 0<=budget<=16:raise ValueError('Invalid native graph kind/budget')
        H=self.helpers;mm=self.gemm.product;left=self.sparse.product_left;right=self.sparse.product_right
        a,pi,sqrt_pi,live,s,mr,rates=self.matrix.forward(indices,x)
        z,diagnostic=H.norm_scale(a,budget)
        a2=left(z,z);a3=left(z,a2);a6=mm(a3,a3)
        b0,b1,b2,b3,b4=self.fold(z,a2,a3,a6)
        c=H.add(b3,mm(b0,b4));d=H.add(b2,c);result=H.add(b1,mm(d,c))
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
        result.update(schema='kaksforge.full_T18_observed_structural_vjp_round6/1',
            reverse_products='Budget0 first2 original products on observed-support CSR/CSC, full dense output; final1dense+2sparse products restricted to587 coordinates; all other products unchanged',
            observed_adjoint=self.observed.manifest(),observed_static_contract=static_tape_contract(),
            final_coordinate_restriction_only=False,observed_support_restriction_only_at_first_adjoint=True,
            upstream_dense_gradients_retained=True,optimizer_or_MA_changed=False)
        return result

def static_tape_contract():
    """Immutable parent hash plus exact one-block source replacement only."""
    parent_contract=selected_static_tape_contract()
    parent=inspect.getsource(SelectedTailT18Runtime.forward_gradient)
    parent_sha=hashlib.sha256(parent.encode()).hexdigest()
    old=('        gb2=mm(g,c.transpose(-1,-2));gb3=H.add(gb2,mm(d.transpose(-1,-2),g))\n')
    new=(
        '        if budget==0:\n'
        '            gb2=self.observed.product_left(g,c.transpose(-1,-2),indices)\n'
        '            gb3=H.add(gb2,self.observed.product_right(g,d.transpose(-1,-2),indices))\n'
        '        else:\n'
        '            gb2=mm(g,c.transpose(-1,-2));gb3=H.add(gb2,mm(d.transpose(-1,-2),g))\n')
    actual=inspect.getsource(ObservedSelectedT18Runtime.forward_gradient)
    if parent_sha!=EXPECTED_PARENT_METHOD_SHA256 or parent.count(old)!=1 or actual!=parent.replace(old,new):
        raise RuntimeError('Observed combined tape changed statements outside budget0 first2 reverse products')
    return dict(PASS=True,expected_parent_method_SHA256=EXPECTED_PARENT_METHOD_SHA256,
        parent_method_normalized_SHA256=parent_sha,new_method_normalized_SHA256=hashlib.sha256(actual.encode()).hexdigest(),
        parent_selected_contract=parent_contract,only_budget0_first2_reverse_products_replaced=True,
        all_other_statements_byte_identical=True,all_forward_matrices_full61=True,
        downstream_adjoints_full61=True,all14_models_full263_edges_and_mr=True,
        original_k_order_and_FP64=True,original_scaling_guards_and_optimizer=True,
        actual_GPU_parity_and_fullfit_still_required=True)

class PublicationGPUGradient(SelectedProvider):
    def __init__(self,*args,stream_count=1,force_full_tail=False,force_full_observed=False,**kwargs):
        if type(force_full_observed) is not bool:
            raise ValueError('force_full_observed must be explicit bool')
        self.observed_runtime=None
        self.observed_static_contract=static_tape_contract()
        build_observed_adjoint()  # Cached source/compiler/PTX verification before context creation.
        self.reduced_source_before=sha(SOURCE)
        self.reduced_candidate=CANDIDATE
        self.observed_installed_before_requests=False
        super().__init__(*args,stream_count=stream_count,force_full_tail=force_full_tail,**kwargs)
        try:
            if self.requests or self.waves or self.gradient_buckets or self.api._graphs:
                raise RuntimeError('Observed-adjoint install requires a fresh provider')
            self._enter()
            try:
                with self.api.cuda.device(self.device):
                    self.observed_runtime=ObservedAdjointRuntime.__new__(ObservedAdjointRuntime)
                    self.observed_runtime.__init__(self.api,self.device,self.prepared,force_full=force_full_observed)
                    self.native_runtime.observed=self.observed_runtime
                    self.native_runtime.__class__=ObservedSelectedT18Runtime
                    self.observed_installed_before_requests=True
            finally:
                self._public_lock.release()
        except BaseException:
            self._close_resources(constructor_failure=True)
            raise

    def _close_resources(self,constructor_failure=False):
        # Parent owns the lock, all lane events and pools. Known completion must
        # precede graph destruction, and every graph must be gone before this
        # additional module is unloaded. Parent closes its modules/context last.
        observed=getattr(self,'observed_runtime',None)
        if getattr(self,'api',None) is not None and observed is not None:
            self._drain()
            for key,bucket in list(self.gradient_buckets.items()):
                self._close_bucket(bucket)
                del self.gradient_buckets[key]
            while self._orphan_buckets:
                self._close_bucket(self._orphan_buckets[-1])
                self._orphan_buckets.pop()
            module=getattr(observed,'module',None)
            if module is not None and module.value:
                with self.api.cuda.device(self.device):
                    observed.close()
        super()._close_resources(constructor_failure=constructor_failure)

    def graph_manifest(self):
        result=super().graph_manifest()
        if sha(SOURCE)!=self.reduced_source_before or not self.observed_installed_before_requests:
            raise RuntimeError('Observed provider source/installation changed')
        result.update(schema=SCHEMA,reduced_candidate=self.reduced_candidate,
            observed_adjoint_installed_before_requests=True,observed_adjoint_runtime=self.observed_runtime.manifest(),
            reduced_tape_contract=self.observed_static_contract,provider_source_sha256=self.reduced_source_before,
            CUDA_timing_events_used=False,diagnostic_execution_only=False,execution_policy_only=False,
            optimizer_and_model_equations_changed=False,science_or_optimizer_changed=False,
            full_native_forward_gradient_inherited_unchanged=False,
            precision='FP64 original full forward/adjoints; explicit ascending FMA and exact support certificates',
            required_science_gates='HP28, observed finite/unsafe/forced-full primitives, partitions, complete1000, fresh CPU endpoints')
        return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-cached',action='store_true')
    args=parser.parse_args()
    report={'PASS':True,'schema':SCHEMA,'candidate':CANDIDATE,'engine':ENGINE,
            'static_contract':static_tape_contract(),'GPU_context_created':False}
    if args.verify_cached:
        ptx,metadata=build_observed_adjoint()
        report.update(observed_PTX_sha256=metadata['ptx_sha256'],observed_PTX_bytes=len(ptx))
    print(json.dumps(report))

if __name__=='__main__':
    main()
