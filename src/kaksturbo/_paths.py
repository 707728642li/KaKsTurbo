"""Filesystem paths at public boundaries, without global OS configuration.

Windows' extended-length namespace works even with LongPathsEnabled=0.
Normalize relative components before adding the prefix; propagate it through
child paths, checkpoints and the isolated engine instead of moving run data.
"""
from __future__ import annotations
import os
from pathlib import Path


def absolute_path(value) -> Path:
    path = Path(value).expanduser()
    if os.name != 'nt':
        return path.resolve()
    name = os.path.abspath(os.fspath(path)).replace('/', '\\')
    if name.startswith('\\\\.\\'):
        raise ValueError('Windows device paths are not input or output directories')
    if not name.startswith('\\\\?\\'):
        name = '\\\\?\\UNC\\' + name[2:] if name.startswith('\\\\') else '\\\\?\\' + name
    return Path(name).resolve()
