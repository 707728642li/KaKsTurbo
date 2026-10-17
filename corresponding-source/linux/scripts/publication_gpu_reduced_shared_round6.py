"""Fresh shared observed-adjoint layout on the already-derived combined tape."""
from __future__ import annotations
import argparse
import hashlib
import inspect
import json
from pathlib import Path

from gpu_observed_adjoint_shared_round6 import ObservedSharedAdjointRuntime,build_observed_shared_adjoint
from publication_gpu_gradient import GPUGradientError
from publication_gpu_native_driver_round3 import sha
from publication_gpu_selected_round6 import PublicationGPUGradient as SelectedProvider
from publication_gpu_reduced_round6 import (PublicationGPUGradient as CombinedProvider,
    ObservedSelectedT18Runtime,static_tape_contract as combined_static_tape_contract)

SOURCE=Path(__file__).resolve()
SCHEMA='kaksforge.observed_packed_structural_vjp_gradient_round6/2'
ENGINE='CUDA Driver API; single context packed observed-adjoint plus structural587 VJP; cold-result reuse; no Torch import; no CUDA timing events'
CANDIDATE='observed_shared_plus_structural587'
EXPECTED_METHOD_SHA256='da9458e795a50437411cf222daaf3c1e0915554bd8a39e61837481f9efffc526'

class SharedObservedSelectedT18Runtime(ObservedSelectedT18Runtime):
    def manifest(self):
        result=super().manifest()
        result.update(schema='kaksforge.full_T18_observed_packed_structural_vjp_round6/2',
                      observed_shared_tape_contract=static_tape_contract())
        return result

def static_tape_contract():
    contract=combined_static_tape_contract()
    method=SharedObservedSelectedT18Runtime.forward_gradient
    identity=hashlib.sha256(inspect.getsource(method).encode()).hexdigest()
    if method is not ObservedSelectedT18Runtime.forward_gradient or identity!=EXPECTED_METHOD_SHA256:
        raise RuntimeError('Shared layout changed the combined mathematical tape')
    return dict(PASS=True,combined_parent_contract=contract,forward_gradient_identical_inherited_function=True,
        inherited_method_normalized_SHA256=identity,only_observed_product_layout_replaced=True,
        all14_models_full263_edges_and_mr=True,full_forward_and_upstream_matrices_preserved=True)

class PublicationGPUGradient(CombinedProvider):
    def __init__(self,*args,stream_count=1,force_full_tail=False,force_full_observed=False,observed_layout='shared',**kwargs):
        if type(force_full_observed) is not bool:raise ValueError('force_full_observed must be explicit bool')
        if observed_layout not in ('shared','original'):raise ValueError('observed layout must be shared or original')
        self.observed_layout=observed_layout
        self.observed_runtime=None;self.observed_static_contract=static_tape_contract()
        build_observed_shared_adjoint()
        self.reduced_source_before=sha(SOURCE)
        self.reduced_candidate=CANDIDATE if observed_layout=='shared' else 'observed_packed_plus_structural587'
        self.observed_installed_before_requests=False
        # Construct the unchanged selected provider directly; avoid constructing
        # an unused v1 observed module/upload before installing this new layout.
        SelectedProvider.__init__(self,*args,stream_count=stream_count,force_full_tail=force_full_tail,**kwargs)
        try:
            if self.requests or self.waves or self.gradient_buckets or self.api._graphs:
                raise RuntimeError('Shared observed-adjoint install requires a fresh provider')
            self._enter()
            try:
                with self.api.cuda.device(self.device):
                    self.observed_runtime=ObservedSharedAdjointRuntime.__new__(ObservedSharedAdjointRuntime)
                    self.observed_runtime.__init__(self.api,self.device,self.prepared,force_full=force_full_observed,layout=observed_layout)
                    self.native_runtime.observed=self.observed_runtime
                    self.native_runtime.__class__=SharedObservedSelectedT18Runtime
                    self.observed_installed_before_requests=True
            finally:self._public_lock.release()
        except BaseException:self._close_resources(constructor_failure=True);raise

    # The combined provider's verified close drains lanes, destroys every graph,
    # unloads this observed module, then frees the parent modules/context.
    def graph_manifest(self):
        result=SelectedProvider.graph_manifest(self)
        if sha(SOURCE)!=self.reduced_source_before or not self.observed_installed_before_requests:
            raise RuntimeError('Shared observed source/installation changed')
        result.update(schema=SCHEMA,reduced_candidate=self.reduced_candidate,
            observed_layout=self.observed_layout,
            observed_adjoint_installed_before_requests=True,observed_adjoint_runtime=self.observed_runtime.manifest(),
            reduced_tape_contract=self.observed_static_contract,provider_source_sha256=self.reduced_source_before,
            CUDA_timing_events_used=False,diagnostic_execution_only=False,execution_policy_only=False,
            optimizer_and_model_equations_changed=False,science_or_optimizer_changed=False,
            full_native_forward_gradient_inherited_unchanged=False,
            precision='FP64 inherited combined tape; exact packed support upload; explicit layout and finite/support certificate',
            required_science_gates='HP28, shared finite/unsafe/forced-full primitives, partitions, complete1000, fresh CPU endpoints')
        return result

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--verify-cached',action='store_true');args=parser.parse_args()
    report={'PASS':True,'schema':SCHEMA,'candidate':CANDIDATE,'engine':ENGINE,'static_contract':static_tape_contract(),'GPU_context_created':False}
    if args.verify_cached:
        ptx,metadata=build_observed_shared_adjoint();report.update(observed_PTX_sha256=metadata['ptx_sha256'],observed_PTX_bytes=len(ptx))
    print(json.dumps(report))

if __name__=='__main__':main()
