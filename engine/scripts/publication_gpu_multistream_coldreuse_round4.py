"""Execution-only cold-tape reuse and optional stable-key lanes, on round4 math.

The parent's first real dry tape is fully synchronized before graph capture.
CUDA Driver capture records kernel launches instead of executing them, and its
exact allocation pool preserves every result/workspace pointer. A fresh bucket
therefore may copy its already completed dry result instead of launching the
same graph once more. This is an execution hypothesis until actual FP64 byte
parity, HP28 and full-fit gates pass; nothing here changes numerical equations.
"""
from __future__ import annotations

import ctypes as C
from pathlib import Path
import time

from publication_gpu_multistream_round4 import PublicationGPUGradient as LaneProvider
from publication_gpu_native_driver_round3 import sha
from publication_gpu_gradient import GPUGradientError
from publication_gpu_gradient_v2 import GraphWorkspaceBudgetError

SOURCE = Path(__file__).resolve()
CAPTURE_DOC = 'https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__STREAM.html'


class PublicationGPUGradient(LaneProvider):
    def __init__(self, *args, schedule='round_robin', cold_reuse=True, **kwargs):
        if schedule not in ('round_robin', 'stable_keys'):
            raise ValueError('schedule must be round_robin or stable_keys')
        if type(cold_reuse) is not bool:
            raise ValueError('cold_reuse must be an explicit bool')
        self.schedule, self.cold_reuse = schedule, cold_reuse
        self._stable_key_lanes = {}
        self._stable_lane_weight = []
        self.cold_result_reuses = 0
        self.cold_graph_result_reuses = 0
        self.cold_graphless_result_reuses = 0
        self.coldreuse_source_before = sha(SOURCE)
        super().__init__(*args, **kwargs)
        self._stable_lane_weight = [0] * self.stream_count

    def _lane_for(self, ordinal, size, budget, kind):
        if self.schedule == 'round_robin':
            return ordinal % self.stream_count
        key = (size, budget, kind)
        lane = self._stable_key_lanes.get(key)
        if lane is None:
            # A deterministic first-seen greedy placement avoids a hash that
            # accidentally assigns every expensive kind1 tape to one lane.
            # The weight affects scheduling only, never a numerical decision.
            lane = min(range(self.stream_count), key=lambda index: (self._stable_lane_weight[index], index))
            self._stable_key_lanes[key] = lane
            self._stable_lane_weight[lane] += size * (budget + 1) * (1 if kind == 0 else 4)
        return lane

    def _new_gradient_bucket(self, size, budget, kind, input_row, lane=0):
        bucket = super()._new_gradient_bucket(size, budget, kind, input_row, lane)
        # Parent has completed the actual dry tape (and any requested warmups),
        # then optionally captured the exact same preallocated tape without
        # launching the graph. No other bucket shares these writable arrays.
        bucket['completed_cold_result_ready'] = True
        return bucket

    def _result_for(self, bucket, budget, kind, is_new):
        """Return an actual completed cold result or enqueue normal warm work."""
        if is_new and self.cold_reuse:
            if not bucket.get('completed_cold_result_ready'):
                raise RuntimeError('Fresh bucket has no completed dry tape to reuse')
            result = bucket['result']
            if result is None:
                raise RuntimeError('Fresh bucket lost its dry result')
            bucket['completed_cold_result_ready'] = False
            self.cold_result_reuses += 1
            if bucket['graph'] is None:
                self.cold_graphless_result_reuses += 1
            else:
                self.cold_graph_result_reuses += 1
            # No graph launch and no forward call is made in this branch.
            # Counters deliberately never claim it was a graph replay.
            return result
        bucket['completed_cold_result_ready'] = False
        if bucket['graph'] is not None:
            bucket['graph'].launch(stream=bucket['stream'])
            self.gradient_graph_replays += 1
            return bucket['result']
        with self.api.capture_pool(bucket['pool']):
            return self._forward_gradient(bucket['indices'], bucket['x'], budget, kind)

    def _evaluate_wave(self, slots, parameters, kinds, count):
        self._ensure_group_outputs(count)
        started = time.perf_counter()
        self.native_host._check(self.native_host.dll.kaks_publication_validate(self.native_host.context,
            slots, parameters, count, self.native_host.error, len(self.native_host.error)))
        self._elapsed('validate', started)
        started = time.perf_counter()
        self.native_host._check(self.tight_host.group(self.native_host.context, slots, parameters, count,
            self.max_squarings, int(self.adaptive_squarings), self.group_order, self.group_levels,
            self.group_offsets, C.byref(self.native_host.maximum), self.native_host.error, len(self.native_host.error)))
        self.staging.check(self.staging.dll.drv_split(self.group_order, count, kinds, self.group_offsets,
            self.split_order, self.split_offsets), 'stable kind split')
        self._elapsed('group', started)
        limit = self.batch_tasks
        while True:
            started = time.perf_counter()
            chunks, padded = self._plan(count, limit)
            self._ensure_pinned(padded, count)
            self._elapsed('plan', started)
            started = time.perf_counter()
            for size, budget, kind, offset, real, input_row, output_row in chunks:
                self.staging.check(self.staging.dll.drv_pack(slots, parameters, self.split_order, count,
                    offset, real, size, self._pinned_slots.data_ptr() + input_row * 8,
                    self._pinned_x.data_ptr() + input_row * 56), 'request pack')
            self._elapsed('pack', started)
            try:
                started = time.perf_counter()
                for ordinal, (size, budget, kind, offset, real, input_row, output_row) in enumerate(chunks):
                    lane = self._lane_for(ordinal, size, budget, kind)
                    key = (lane, size, budget, kind)
                    stream = self.lanes.stream(lane)
                    bucket = self.gradient_buckets.get(key)
                    is_new = bucket is None
                    if is_new:
                        bucket = self._new_gradient_bucket(size, budget, kind, input_row, lane)
                    else:
                        self.gradient_buckets.move_to_end(key)
                    with self.api.cuda.stream(stream):
                        if bucket['lane'] != lane or bucket['stream'] is not stream:
                            raise RuntimeError('Graph workspace migrated across CUDA lanes')
                        if not is_new:
                            self._h2d(bucket, size, input_row)
                        result = self._result_for(bucket, budget, kind, is_new)
                        self.api.copy_d2h(self._pinned_result, result, nbytes=real * 11 * 8,
                            host_offset=output_row * 11 * 8, stream=stream)
                        self.lanes.note_submission(lane)
                    self._pending.append((bucket, result))
                    self.grouped_padded_rows += size
                    self.grouped_graph_calls += 1
                    self.objective_transfer_bytes += real * 11 * 8
                self._elapsed('enqueue', started)
                started = time.perf_counter()
                self._drain()
                self._elapsed('complete', started)
                self._free_memory()
                break
            except GraphWorkspaceBudgetError as error:
                self._drain()
                if limit == 1:
                    raise
                limit = min(limit // 2, max(1, error.suggested_rows))
                limit = 1 << (limit.bit_length() - 1)
                self.chunk_reductions += 1
            except BaseException:
                self._drain()
                self._quarantined = True
                raise
        started = time.perf_counter()
        for size, budget, kind, offset, real, input_row, output_row in chunks:
            self.staging.check(self.staging.dll.drv_scatter(self.split_order, count, offset, real,
                self._pinned_result.data_ptr() + output_row * 11 * 8,
                self.output_F, self.output_g, self.output_diagnostics, self.output_codes), 'result scatter')
        self._elapsed('scatter', started)
        self.requests += count
        self.waves += 1
        for budget in range(self.max_squarings + 1):
            self.grouped_row_histogram[budget] += self.group_offsets[budget + 1] - self.group_offsets[budget]
            for kind in (0, 1):
                self.kind_requests[kind] += self.split_offsets[budget * 2 + kind + 1] - self.split_offsets[budget * 2 + kind]
        f = (C.c_double * count).from_buffer(self.output_F)
        g = (C.c_double * (count * 7)).from_buffer(self.output_g)
        self.last_diagnostics = (C.c_double * (count * 4)).from_buffer(self.output_diagnostics)
        self.last_codes = (C.c_uint32 * count).from_buffer(self.output_codes)
        bad = sum(self.last_codes)
        self.failed_requests += bad
        if bad:
            raise GPUGradientError('CUDA lane FP64 rows failed unchanged finite/scaling guard; no clipping or truncation',
                objectives=list(f), gradient_log=[list(g[row * 7:row * 7 + 7]) for row in range(count)],
                diagnostics=[list(self.last_diagnostics[row * 4:row * 4 + 4]) for row in range(count)], codes=list(self.last_codes))
        return f, g

    def graph_manifest(self):
        result = super().graph_manifest()
        if sha(SOURCE) != self.coldreuse_source_before:
            raise RuntimeError('Cold reuse lane source changed')
        result.update({
            'schema': 'kaksforge.same_context_multistream_coldreuse_gradient_round4/1',
            'round4_parent_provider_source_sha256': self.round4_source_before,
            'provider_source_sha256': self.coldreuse_source_before,
            'provider_source_unchanged': True,
            'schedule': self.schedule,
            'cold_reuse': self.cold_reuse,
            'cold_result_reuses': self.cold_result_reuses,
            'cold_graph_result_reuses': self.cold_graph_result_reuses,
            'cold_graphless_result_reuses': self.cold_graphless_result_reuses,
            'cold_result_reuse_scope': 'Fresh bucket only: original-input actual dry result fully synchronized before nonexecuting capture; no duplicate initial replay/forward',
            'cold_result_proof_required': 'Same-input complete FP64 output byte gate and HP28/full-fit gates; capture semantics alone is not a numerical gate',
            'NVIDIA_stream_capture_semantics_source': CAPTURE_DOC,
            'NVIDIA_stream_capture_semantics': 'Operations pushed into a capturing stream are recorded into a graph rather than executed',
            'stable_key_mapping': [[*key, lane] for key, lane in self._stable_key_lanes.items()],
            'stable_lane_weight': list(self._stable_lane_weight),
            'stable_weight_scope': 'Deterministic scheduling heuristic size*(budget+1)*(1 for kind0 else4); does not affect arithmetic',
            'graph_replay_counter_scope': 'Actual graph.launch calls only; completed dry result copies are counted separately',
            'execution_policy_only': True,
            'optimizer_and_model_equations_changed': False,
        })
        return result
