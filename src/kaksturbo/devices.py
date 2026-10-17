"""Device preflight only: never fit a model, allocate GPU memory, or fall back silently."""
from __future__ import annotations
import csv
import ctypes as C
import os
from pathlib import Path
import subprocess
from .errors import KaKsTurboError


class DeviceError(KaKsTurboError):
    code = 'device_unavailable'

    def __init__(self, reason, message, action):
        super().__init__(message, details={'reason': reason, 'action': action,
                                          'CPU_alternative': 'Use --backend cpu; no NVIDIA driver or CUDA Toolkit is needed.'})


def inventory():
    fields = 'index,uuid,name,memory.total,memory.used,utilization.gpu'
    try:
        query = subprocess.run(['nvidia-smi', '--query-gpu='+fields, '--format=csv,noheader,nounits'],
                               capture_output=True, text=True, timeout=20, check=True)
        processes = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=20, check=True)
    except FileNotFoundError as e:
        raise DeviceError('nvidia_smi_missing', 'NVIDIA device management tool nvidia-smi was not found.',
                          'If an NVIDIA GPU is installed, install its driver and put nvidia-smi on PATH. Otherwise use CPU.') from e
    except subprocess.TimeoutExpired as e:
        raise DeviceError('driver_timeout', 'NVIDIA driver query timed out.', 'Check nvidia-smi outside KaKsTurbo and retry when the driver responds.') from e
    except subprocess.CalledProcessError as e:
        detail = (e.stderr or e.stdout or '').strip()[-1000:]
        raise DeviceError('driver_query_failed', 'NVIDIA driver/device query failed: '+detail,
                          'Check that the GPU is visible to this OS/container and that its NVIDIA driver is working.') from e
    cards = []
    try:
        for row in csv.reader(query.stdout.splitlines()):
            if not row: continue
            p = [v.strip() for v in row]
            # A missing utilization measurement must not be mistaken for idle.
            util = None if p[5] in ('N/A','[N/A]','[Not Supported]') else int(p[5])
            cards.append(dict(index=int(p[0]),uuid=p[1],name=p[2],total_MiB=int(p[3]),
                              used_MiB=int(p[4]),utilization_percent=util))
    except (ValueError, IndexError) as e:
        raise DeviceError('invalid_inventory', 'GPU memory/identity information could not be read reliably.',
                          'Check nvidia-smi output; memory availability must be known before GPU execution.') from e
    if not cards:
        raise DeviceError('no_gpu', 'No NVIDIA GPU was reported.', 'Use CPU, or expose an NVIDIA GPU to this machine/container.')
    return dict(cards=cards,query_csv=query.stdout,process_snapshot=processes.stdout)


def choose(config, snapshot=None):
    def value(k,d): return config.get(k,d) if isinstance(config,dict) else getattr(config,k,d)
    snapshot = snapshot or inventory()
    all_cards = snapshot['cards']; cards = list(all_cards)
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible is not None:
        if visible.strip() in ('','-1'):
            raise DeviceError('gpu_hidden', 'CUDA_VISIBLE_DEVICES hides all GPUs.', 'Unset it or select a physical GPU UUID; otherwise use CPU.')
        permitted = []
        for item in visible.split(','):
            item = item.strip()
            matches = [c for c in all_cards if item == str(c['index']) or (item.startswith('GPU-') and c['uuid'].startswith(item))]
            if len(matches) != 1 or matches[0] in permitted:
                raise DeviceError('invalid_visibility', 'CUDA_VISIBLE_DEVICES contains an unknown, ambiguous, or duplicate GPU.', 'Use unique physical indices or GPU UUIDs from nvidia-smi.')
            permitted.append(matches[0])
        cards = permitted
    selector = value('device','auto')
    if selector != 'auto':
        matches = [c for c in cards if selector == 'cuda:'+str(c['index']) or (selector.startswith('GPU-') and c['uuid'].startswith(selector))]
        if len(matches) != 1:
            raise DeviceError('device_not_visible', 'The requested physical GPU is absent, hidden, or ambiguous: '+selector,
                              'Choose a visible physical index/UUID from nvidia-smi. Indices are physical, not remapped mask indices.')
        cards = matches
    eligible = [c for c in cards if c['total_MiB']-c['used_MiB'] >= 8192]
    if not eligible:
        raise DeviceError('insufficient_memory', 'No selected GPU has the required 8 GiB of free memory (including reserve).',
                          'Choose another GPU, retry after memory is released, or use CPU. KaKsTurbo will not stop other jobs.')
    if selector != 'auto': selected = eligible[0]
    else:
        idle = [c for c in eligible if c['utilization_percent']==0 and c['used_MiB']<=100]
        if idle: selected = next((c for c in idle if c['index']==1),idle[0])
        else:
            spare = [c for c in eligible if c['utilization_percent'] is not None and c['utilization_percent']<80]
            if not spare:
                raise DeviceError('gpu_busy_or_unknown', 'No GPU currently has measured compute headroom.',
                                  'Retry later or use CPU. An explicit --device selects a busy GPU only if its memory admission passes.')
            selected = min(spare,key=lambda c:(c['utilization_percent'],c['used_MiB']))
    return dict(selected,admission_free_MiB=selected['total_MiB']-selected['used_MiB'],
                admission_minimum_free_MiB=8192,runtime_reserve_bytes=2*1024**3),snapshot


def check_gpu_package(root):
    root=Path(root)
    library=root/'build/publication-graph-host-v1'/('torch_graph_host.dll' if os.name=='nt' else 'libtorch_graph_host.so')
    if not library.is_file():
        raise DeviceError('gpu_backend_not_packaged', 'This package does not include a qualified GPU backend for this platform.',
                          'Use --backend cpu. Installing CUDA alone cannot add the missing backend; install a qualified GPU package when available.')
    return library


def probe_driver(selected):
    """Initialize the driver, not a context; allocate no GPU memory. Inventory first."""
    try: driver = C.WinDLL('nvcuda.dll') if os.name=='nt' else C.CDLL('libcuda.so.1')
    except OSError as e:
        raise DeviceError('cuda_driver_missing','The CUDA driver library could not be loaded.',
                          'Install/expose the NVIDIA driver. A CUDA Toolkit installation alone is not sufficient.') from e
    def check(code, operation):
        if code:
            reason={100:'no_cuda_device',35:'driver_too_old',803:'driver_mismatch',804:'unsupported_compatibility',999:'driver_unknown_error'}.get(code,'cuda_driver_error')
            raise DeviceError(reason,f'{operation} failed (CUDA error {code}).', 'Check the host driver and GPU passthrough/container setup; CPU remains available.')
    check(driver.cuInit(0),'cuInit')
    version=C.c_int();check(driver.cuDriverGetVersion(C.byref(version)),'cuDriverGetVersion')
    if version.value<12090:
        raise DeviceError('driver_too_old',f'This GPU engine contains CUDA 12.9 / PTX 8.8 kernels; the driver reports {version.value}.',
                          'Update to a driver supporting CUDA 12.9 or newer, or use CPU. No Toolkit installation is required for precompiled kernels.')
    count=C.c_int();check(driver.cuDeviceGetCount(C.byref(count)),'cuDeviceGetCount')
    found=None
    for ordinal in range(count.value):
        device=C.c_int();check(driver.cuDeviceGet(C.byref(device),ordinal),'cuDeviceGet')
        uuid=(C.c_ubyte*16)();check(driver.cuDeviceGetUuid(C.byref(uuid),device),'cuDeviceGetUuid')
        h=bytes(uuid).hex();identity='GPU-'+h[:8]+'-'+h[8:12]+'-'+h[12:16]+'-'+h[16:20]+'-'+h[20:]
        if identity.lower()==selected['uuid'].lower(): found=device; break
    if found is None:
        raise DeviceError('driver_device_hidden','nvidia-smi sees the selected GPU, but the CUDA driver does not expose it.',
                          'Check CUDA_VISIBLE_DEVICES and container/VM device permissions; use a visible GPU UUID.')
    major=C.c_int();minor=C.c_int();check(driver.cuDeviceComputeCapability(C.byref(major),C.byref(minor),found),'cuDeviceComputeCapability')
    if (major.value,minor.value)<(8,6):
        raise DeviceError('unsupported_compute_capability',f'This compiled GPU package requires compute capability >=8.6; selected GPU is {major.value}.{minor.value}.',
                          'Use CPU or a separately validated build for this GPU architecture.')
    return dict(driver_version=version.value,compute_capability=f'{major.value}.{minor.value}',
                selected_uuid=selected['uuid'],driver_initialized=True,context_created=False,
                gpu_memory_allocated=False,numerical_fit_performed=False,kernel_execution_verified=False)
