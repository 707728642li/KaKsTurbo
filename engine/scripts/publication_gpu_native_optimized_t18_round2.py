"""Constructor-only native-intscale + coalesced GEMM + structural-zero Q path."""
from __future__ import annotations
from gpu_native_intscale_t18_round2 import NativeIntScaleFusedT18Runtime,NativeIntScaleT18Helpers
from gpu_matrix_structural_zero_round2 import StructuralZeroForwardMatrixRuntime
from gpu_coalesced_gemm_round2 import CoalescedGemmRuntime
from gpu_sparse_t18_round2 import SparseRuntime
from gpu_t18_fusion_round2 import FusionRuntime
from gpu_native_objective_round2 import NativeObjectiveRuntime
from publication_gpu_gradient_v2 import PublicationGPUGradient as OriginalProvider


class NativeOptimizedT18Runtime(NativeIntScaleFusedT18Runtime):
    def __init__(self,owner,objective_runtime):
        self.owner,self.torch,self.device=owner,owner.torch,owner.device
        self.matrix=StructuralZeroForwardMatrixRuntime(owner);self.gemm=CoalescedGemmRuntime(self.torch,self.device,32)
        self.sparse=SparseRuntime(self.torch,self.device,self.gemm,owner.prepared.edges)
        self.coefficients=FusionRuntime(self.torch,self.device);self.helpers=NativeIntScaleT18Helpers(self.torch,self.device)
        self.objective=objective_runtime
    def manifest(self):
        result=super().manifest();result['schema']='kaksforge.native_full_T18_optimized_round2/1'
        result['native_intscale_full_tape_and_reverse_inherited_unchanged']=True
        result['constructor_only_combination']=['StructuralZeroForwardMatrixRuntime','CoalescedGemmRuntime','NativeIntScaleT18Helpers']
        return result


class NativeOptimizedT18Mixin:
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.native_t18=NativeOptimizedT18Runtime(self,NativeObjectiveRuntime(self))
    def _forward_gradient(self,indices,x,budget,kind):return self.native_t18.forward_gradient(indices,x,budget,kind)
    def graph_manifest(self):
        result=super().graph_manifest();result['native_full_T18_optimized']=self.native_t18.manifest();return result


class PublicationGPUGradient(NativeOptimizedT18Mixin,OriginalProvider):pass
