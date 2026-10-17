"""Owned nonblocking lanes on one already selected round3 CUDA context.

This module never creates a CUDA context or loads CUDA at import. The provider
owns the existing NativeCUDAArrayAPI, and it remains the sole resource owner.
Lane selection is deterministic; a bucket's GPU workspace stays on its lane.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import threading

from cuda_array_round3 import NativeCUDAArrayAPI


SOURCE = Path(__file__).resolve()


class MultiStreamLanes:
    """Borrow one API/context and own completion markers for 1, 2, or 4 lanes.

    Each event is recorded after every subgroup's final D2H, so a whole-wave
    wait proves that pinned results are readable. Workspace reuse is instead
    guaranteed by same-stream FIFO order, with the lane included in cache keys.
    All synchronization is bounded by the unchanged API's explicit deadline.
    """

    def __init__(self, api: NativeCUDAArrayAPI, stream_count=1):
        if not isinstance(api, NativeCUDAArrayAPI):
            raise TypeError('Lanes require the existing NativeCUDAArrayAPI')
        if type(stream_count) is not int or stream_count not in (1, 2, 4):
            raise ValueError('stream_count must be 1, 2, or 4')
        if api.closed or api._capturing or api._record is not None:
            raise RuntimeError('Cannot install lanes on a closed/recording API')
        self.api = api
        self.owner_thread = threading.get_ident()
        self.context_handle = api.context.value
        self.source_before = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
        self.streams = [api.default_stream]
        self.events = []
        self.closed = False
        self.submitted = [0] * stream_count
        self.recorded = [False] * stream_count
        # If construction fails, resources stay reachable in api._streams /
        # api._events and its explicit close cleans them after synchronization.
        for _ in range(stream_count - 1):
            self.streams.append(api.create_stream())
        for _ in range(stream_count):
            self.events.append(api.event(enable_timing=False))

    @property
    def stream_count(self):
        return len(self.streams)

    def _validate(self):
        if threading.get_ident() != self.owner_thread:
            raise RuntimeError('CUDA lanes require their creating thread')
        if self.closed or self.api.closed:
            raise RuntimeError('CUDA lanes are closed')
        if self.api.context.value != self.context_handle:
            raise RuntimeError('The sole owning CUDA context changed')

    def stream(self, lane):
        self._validate()
        if type(lane) is not int or not 0 <= lane < self.stream_count:
            raise ValueError('Invalid lane ordinal')
        return self.api._stream(self.streams[lane])

    def note_submission(self, lane):
        """Mark subgroup completion on the same lane, after its output copy."""
        stream = self.stream(lane)
        self.events[lane].record(stream)
        self.recorded[lane] = True
        self.submitted[lane] += 1

    def finish_wave(self, timeout=10.):
        self._validate()
        # The API waits all its owned streams against one shared deadline and
        # releases async-copy host references only after actual completion.
        self.api.synchronize(timeout=timeout)
        for lane, was_recorded in enumerate(self.recorded):
            if was_recorded and not self.events[lane].query():
                raise RuntimeError('Completed stream has an incomplete terminal marker')
        self.recorded[:] = [False] * self.stream_count

    def manifest(self):
        self._validate()
        source_after = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
        if source_after != self.source_before:
            raise RuntimeError('CUDA lane source changed while executing')
        return {
            'schema': 'kaksforge.same_context_CUDA_lanes_round4/1',
            'stream_count': self.stream_count,
            'single_API_object': True,
            'single_context_object': True,
            'context_handle': self.context_handle,
            'lane_stream_handles': [stream.cuda_stream for stream in self.streams],
            'nonblocking_stream_flags': 1,
            'physical_GPU_UUID': self.api.uuid,
            'subgroups_submitted_per_lane': list(self.submitted),
            'events_recorded_per_lane': list(self.recorded),
            'completion': 'Same-lane terminal markers after D2H; all owned streams drained before wave return',
            'resources_owner': 'One original NativeCUDAArrayAPI; no additional CUDA context',
            'source_sha256': self.source_before,
            'source_unchanged': True,
        }

    def close(self):
        if self.closed:
            return
        self.finish_wave()
        for event in self.events:
            event.close()
        # Streams belong to the API, whose close destroys them after all graphs
        # and allocation pools. Never destroy a stream underneath a live graph.
        self.closed = True
