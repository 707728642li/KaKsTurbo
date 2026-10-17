"""Development provider: row-invariant FP64 matrix preparation + same-context lanes.

The unchanged round3 provider supplies metadata validation, bounds, grouping,
constant uploads, native host staging, output semantics, and explicit cleanup.
Only the already independently gated row-invariant matrix preparation and
subgroup execution scheduling are changed. Input pairs and optimizer are never
partitioned: every whole request wave returns its complete original ordering.
"""
from __future__ import annotations

import ctypes as C
from pathlib import Path
import time

from cuda_multistream_round4 import MultiStreamLanes
from publication_gpu_native_driver_round3 import PublicationGPUGradient as Round3Provider, RESERVE, sha
from publication_gpu_gradient import GPUGradientError
from publication_gpu_gradient_v2 import GraphWorkspaceBudgetError


SOURCE = Path(__file__).resolve()


class PublicationGPUGradient(Round3Provider):
    def __init__(self, *args, stream_count=1, matrix_installer=None, **kwargs):
        if type(stream_count) is not int or stream_count not in (1, 2, 4):
            raise ValueError('stream_count must be 1, 2, or 4')
        self.stream_count = stream_count
        self.lanes = None
        self.round4_source_before = sha(SOURCE)
        # The inherited source_before intentionally identifies round3: its
        # manifest still checks every inherited source, compiler and DLL.
        super().__init__(*args, **kwargs)
        try:
            if matrix_installer is None:
                from gpu_matrix_invariant_round4 import install_invariant_matrix
                matrix_installer = install_invariant_matrix
            if not callable(matrix_installer):
                raise TypeError('A gated row-invariant matrix installer is required')
            matrix_installer(self)
            self.matrix_installer_identity = {
                'module': matrix_installer.__module__,
                'name': getattr(matrix_installer, '__qualname__', matrix_installer.__name__),
            }
            self.lanes = MultiStreamLanes(self.api, stream_count)
            self._free_memory()
        except BaseException:
            self._close_resources(constructor_failure=True)
            raise

    def _drain(self):
        try:
            if self.lanes is None:
                self.api.synchronize(timeout=10)
            else:
                self.lanes.finish_wave(timeout=10)
        except BaseException:
            self._quarantined = True
            raise
        self._pending.clear()

    def _new_gradient_bucket(self, size, budget, kind, input_row, lane=0):
        """Record and capture one lane-private tape/workspace on its stream."""
        estimate = size * 61 * 61 * 8 * (256 + 4 * budget) + 64 * 1024**2
        while len(self.gradient_buckets) >= self.max_cached_buckets:
            self._evict_oldest()
        while estimate + RESERVE > self._free_memory(enforce_reserve=False)[0]:
            if not self._evict_oldest():
                raise GraphWorkspaceBudgetError('Lane pool plus2GiB reserve exceeds free memory', max(1, size // 2))
        stream = self.lanes.stream(lane)
        bucket = {'graph': None, 'pool': None, 'indices': None, 'x': None,
                  'result': None, 'lane': lane, 'stream': stream}
        capture_started = time.perf_counter()
        try:
            with self.api.cuda.stream(stream):
                bucket['indices'] = self.api.empty((size,), dtype=self.api.int64, device=self.device)
                bucket['x'] = self.api.empty((size, 7), dtype=self.api.float64, device=self.device)
                self._h2d(bucket, size, input_row)
                self.api.begin_record()
                try:
                    bucket['result'] = self._forward_gradient(bucket['indices'], bucket['x'], budget, kind)
                finally:
                    bucket['pool'] = self.api.end_record()
                self._drain()
                for _ in range(self.graph_warmups - 1):
                    with self.api.capture_pool(bucket['pool']):
                        bucket['result'] = self._forward_gradient(bucket['indices'], bucket['x'], budget, kind)
                    self._drain()
                if self.use_graphs:
                    with self.api.capture_pool(bucket['pool']):
                        self.api.capture_begin(stream=stream)
                        try:
                            bucket['result'] = self._forward_gradient(bucket['indices'], bucket['x'], budget, kind)
                            bucket['graph'] = self.api.capture_end()
                        except BaseException:
                            self.api.abort_capture()
                            raise
                    self.gradient_graph_captures += 1
            self._free_memory()
        except BaseException:
            try:
                self._drain()
                self._close_bucket(bucket)
            except BaseException:
                self._orphan_buckets.append(bucket)
                raise
            raise
        self.gradient_buckets[(lane, size, budget, kind)] = bucket
        self.cache_high_water = max(self.cache_high_water, len(self.gradient_buckets))
        self._elapsed('capture', capture_started)
        return bucket

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
                    lane = ordinal % self.stream_count
                    key = (lane, size, budget, kind)
                    stream = self.lanes.stream(lane)
                    bucket = self.gradient_buckets.get(key)
                    existing_bucket = bucket is not None
                    if bucket is None:
                        bucket = self._new_gradient_bucket(size, budget, kind, input_row, lane)
                    else:
                        self.gradient_buckets.move_to_end(key)
                    with self.api.cuda.stream(stream):
                        # Never overwrite lane-private GPU inputs ahead of
                        # earlier work: all copies/graphs/output copies for
                        # this bucket are enqueued on the same FIFO stream.
                        if bucket['lane'] != lane or bucket['stream'] is not stream:
                            raise RuntimeError('Graph workspace migrated across CUDA lanes')
                        if existing_bucket:
                            self._h2d(bucket, size, input_row)
                        if bucket['graph'] is not None:
                            bucket['graph'].launch(stream=stream)
                            self.gradient_graph_replays += 1
                            result = bucket['result']
                        else:
                            with self.api.capture_pool(bucket['pool']):
                                result = self._forward_gradient(bucket['indices'], bucket['x'], budget, kind)
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
        if sha(SOURCE) != self.round4_source_before:
            raise RuntimeError('Round4 lane provider source changed')
        result.update({
            'schema': 'kaksforge.same_context_multistream_gradient_round4/1',
            'round3_inherited_provider_source_sha256': self.source_before,
            'provider_source_sha256': self.round4_source_before,
            'provider_source_unchanged': True,
            'stream_count': self.stream_count,
            'lanes': self.lanes.manifest(),
            'lane_graph_counts': [sum(key[0] == lane for key in self.gradient_buckets)
                                  for lane in range(self.stream_count)],
            'gradient_bucket_key_schema': ['lane', 'size', 'budget', 'kind'],
            'wave_completion': 'Shared original request wave; lane-private workspace and same-lane H2D/graph/D2H; all streams drained before original-ordinal scatter',
            'constants_and_modules': 'One context; one upload of readonly constants and one shared loaded native runtime',
            'pinned_arena': 'One whole-wave pinned arena with disjoint per-subgroup source/result byte ranges; never rewritten before all lanes finish',
            'matrix_installer': self.matrix_installer_identity,
            'matrix_preparation_change': 'Gated row-invariant FP64 mr reduction: fixed width32 and shift0; model equations unchanged',
            'full_native_forward_gradient_inherited_unchanged': True,
            'scientific_scope': 'Same all14 FP64 T18 models/native optimizer; no split input or optimizer; numerical layout fix explicitly identified above',
            'requires_row_order_padding_partition_invariance_gate': True,
        })
        return result

    def _close_resources(self, constructor_failure=False):
        # Base cleanup drains every API-owned lane before closing graph/pool /
        # pinned allocations / modules; api.close owns marker and stream close.
        super()._close_resources(constructor_failure=constructor_failure)
        if self.lanes is not None:
            self.lanes.closed = True
