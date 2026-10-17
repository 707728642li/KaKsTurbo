"""Strict FP64 final coordinate VJP; full upstream T18 adjoints retained."""
from __future__ import annotations
import ctypes as C
import hashlib
import inspect
from pathlib import Path

from gpu_native_t18_round2 import NativeFusedT18Runtime
from publication_gpu_native_driver_round3 import NativeDriverT18Runtime, sha
from publication_gpu_multistream_coldreuse_round4 import PublicationGPUGradient as ColdProvider
from publication_gpu_gradient import GPUGradientError

SOURCE = Path(__file__).resolve()
SCHEMA = 'kaksforge.selected_structural_vjp_gradient_round6/1'


class SelectedTailT18Runtime(NativeDriverT18Runtime):
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
        result = super().manifest()
        result.update(schema='kaksforge.full_T18_selected_final_vjp_round6/1',
                      forward_products='5+budget; original full61 matrices and first2 structural sparse',
                      reverse_products='Original full upstream; final1dense+2sparse outputs restricted to587 coordinates',
                      full_native_forward_gradient_method_inherited_unchanged=False,
                      full_native_forward_gradient_inherited_unchanged=False,
                      upstream_dense_gradients_retained=True,
                      final_coordinate_restriction_only=True,
                      optimizer_or_MA_changed=False,
                      selected_tail_source_contract=static_tape_contract())
        return result


def static_tape_contract():
    old = inspect.getsource(NativeFusedT18Runtime.forward_gradient)
    old_tail = (
        '        gz,ga2=H.add_pair(gz,mm(ga3,a2.transpose(-1,-2)),ga2,left(zt,ga3))\n'
        '        g_a=H.add3_scale(gz,right(ga2,zt),left(zt,ga2),diagnostic)\n'
        '        raw=self.matrix.reverse.backward(g_a,s,a,pi,sqrt_pi,mr,rates,x,indices)\n')
    new_tail = (
        '        ga2=H.add(ga2,left(zt,ga3))\n'
        '        raw=self.matrix.reverse.finish(gz,ga3,a2,ga2,z,diagnostic,s,a,pi,sqrt_pi,mr,rates,x,indices)\n')
    actual = inspect.getsource(SelectedTailT18Runtime.forward_gradient)
    if old.count(old_tail) != 1 or actual != old.replace(old_tail, new_tail):
        raise RuntimeError('Selected VJP changed statements outside final coordinate contraction')
    return dict(PASS=True, forward_and_upstream_statements_unchanged=True,
                old_method_normalized_SHA256=hashlib.sha256(old.encode()).hexdigest(),
                new_method_normalized_SHA256=hashlib.sha256(actual.encode()).hexdigest(),
                full_upstream_adjoints=True, strict_positive_mask_and_full_mr_unchanged=True,
                all14_models_and_all263_edges=True, fast_math=False,
                scope='Final observable parameter VJP coordinates only; actual finite/overflow fallback and original gates required')


class PublicationGPUGradient(ColdProvider):
    def __init__(self, *args, stream_count=1, force_full_tail=False, **kwargs):
        from gpu_selected_vjp_round6 import build_selected_vjp, SelectedVJPRuntime
        self.selected_static_contract = static_tape_contract()
        build_selected_vjp()  # Verify cached source/compiler/PTX before context creation.
        self.selected_source_before = sha(SOURCE)
        self.selected_vjp_candidate = 'structural587'
        self.selected_vjp_installed_before_requests = False
        super().__init__(*args, stream_count=stream_count, **kwargs)
        try:
            if self.requests or self.waves or self.gradient_buckets or self.api._graphs:
                raise RuntimeError('Selected VJP install requires a fresh provider')
            self._enter()
            try:
                with self.api.cuda.device(self.device):
                    old = self.native_runtime.matrix.reverse
                    new = SelectedVJPRuntime(self, old, force_full=force_full_tail)
                    code = old.driver.cuModuleUnload(old.module)
                    if code:
                        new._check(new.driver.cuModuleUnload(new.module))
                        new.module = C.c_void_p()
                        raise RuntimeError('Old reverse module unload failed: ' + str(code))
                    old.module = C.c_void_p()
                    self.native_runtime.matrix.reverse = new
                    self.native_runtime.__class__ = SelectedTailT18Runtime
                    self.selected_vjp_installed_before_requests = True
            finally:
                self._public_lock.release()
        except BaseException:
            self._close_resources(constructor_failure=True)
            raise

    def graph_manifest(self):
        result = super().graph_manifest()
        if sha(SOURCE) != self.selected_source_before or not self.selected_vjp_installed_before_requests:
            raise RuntimeError('Selected provider source/installation changed')
        selected = self.native_runtime.matrix.reverse.manifest()
        result.update(schema=SCHEMA, selected_vjp_candidate=self.selected_vjp_candidate,
                      selected_vjp_installed_before_requests=True,
                      selected_vjp_runtime=selected,
                      selected_tape_contract=self.selected_static_contract,
                      provider_source_sha256=self.selected_source_before,
                      CUDA_timing_events_used=False, diagnostic_execution_only=False,
                      execution_policy_only=False,
                      optimizer_and_model_equations_changed=False,
                      science_or_optimizer_changed=False,
                      full_native_forward_gradient_inherited_unchanged=False,
                      sparse_constructor_only_change='Unused AD decorators omitted; selected final coordinate products now explicit',
                      precision='Full FP64 original tape and new final VJP; explicit FMA dot order, no fastmath/TF32',
                      required_science_gates='HP28, finite/nonsymmetric/unsafe full-tail, partitions, complete1000, fresh CPU endpoints')
        return result
