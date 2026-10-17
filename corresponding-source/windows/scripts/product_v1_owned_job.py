"""Launch one owned Windows process tree before any child code can run.

CreateProcessW(CREATE_SUSPENDED) -> AssignProcessToJobObject -> ResumeThread.
The Job permits neither kind of breakaway and has KILL_ON_JOB_CLOSE. Only
handles for this new process/tree are ever used for termination. This is a
small file-stdio benchmark helper, not a complete subprocess.Popen replacement.
"""
from __future__ import annotations
import ctypes as C
import json
import math
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_SOURCES = {
    'create': 'https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-createprocessw',
    'suspended': 'https://learn.microsoft.com/en-us/windows/win32/procthread/process-creation-flags',
    'assign': 'https://learn.microsoft.com/en-us/windows/win32/api/jobapi2/nf-jobapi2-assignprocesstojobobject',
    'resume': 'https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-resumethread',
    'handles': 'https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-updateprocthreadattribute',
    'tree': 'https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects',
}
DWORD, WORD, HANDLE = C.c_uint32, C.c_uint16, C.c_void_p
CREATE_SUSPENDED = 0x4
CREATE_UNICODE_ENVIRONMENT = 0x400
CREATE_NO_WINDOW = 0x08000000
EXTENDED_STARTUPINFO_PRESENT = 0x80000
PROC_THREAD_ATTRIBUTE_HANDLE_LIST = 0x20002
STARTF_USESTDHANDLES = 0x100
WAIT_TIMEOUT, WAIT_FAILED, INFINITE = 258, 0xffffffff, 0xffffffff

class StartupInfo(C.Structure):
    _fields_ = [('cb', DWORD), ('reserved', C.c_wchar_p), ('desktop', C.c_wchar_p), ('title', C.c_wchar_p),
        ('x', DWORD), ('y', DWORD), ('x_size', DWORD), ('y_size', DWORD), ('x_chars', DWORD), ('y_chars', DWORD),
        ('fill', DWORD), ('flags', DWORD), ('show', WORD), ('reserved2_size', WORD), ('reserved2', C.c_void_p),
        ('stdin', HANDLE), ('stdout', HANDLE), ('stderr', HANDLE)]
class StartupInfoEx(C.Structure):
    _fields_ = [('startup', StartupInfo), ('attributes', C.c_void_p)]
class ProcessInformation(C.Structure):
    _fields_ = [('process', HANDLE), ('thread', HANDLE), ('pid', DWORD), ('tid', DWORD)]
class BasicLimits(C.Structure):
    _fields_ = [('process_user_time', C.c_int64), ('job_user_time', C.c_int64), ('flags', DWORD),
        ('min_working_set', C.c_size_t), ('max_working_set', C.c_size_t), ('active_limit', DWORD),
        ('affinity', C.c_size_t), ('priority', DWORD), ('scheduling', DWORD)]
class IOCounters(C.Structure):
    _fields_ = [(name, C.c_uint64) for name in ('read_ops', 'write_ops', 'other_ops', 'read_bytes', 'write_bytes', 'other_bytes')]
class ExtendedLimits(C.Structure):
    _fields_ = [('basic', BasicLimits), ('io', IOCounters), ('process_memory', C.c_size_t), ('job_memory', C.c_size_t),
        ('peak_process_memory', C.c_size_t), ('peak_job_memory', C.c_size_t)]
class Accounting(C.Structure):
    _fields_ = [(name, C.c_int64) for name in ('total_user_time', 'total_kernel_time', 'period_user_time', 'period_kernel_time')] + [
        (name, DWORD) for name in ('page_faults', 'total_processes', 'active_processes', 'terminated_processes')]

def _kernel():
    if os.name != 'nt':
        raise RuntimeError('OwnedJob v2 requires native Windows')
    k = C.WinDLL('kernel32', use_last_error=True)
    signatures = {
        'CreateJobObjectW': ([HANDLE, C.c_wchar_p], HANDLE),
        'SetInformationJobObject': ([HANDLE, C.c_int, HANDLE, DWORD], C.c_int),
        'QueryInformationJobObject': ([HANDLE, C.c_int, HANDLE, DWORD, C.POINTER(DWORD)], C.c_int),
        'AssignProcessToJobObject': ([HANDLE, HANDLE], C.c_int),
        'IsProcessInJob': ([HANDLE, HANDLE, C.POINTER(C.c_int)], C.c_int),
        'TerminateJobObject': ([HANDLE, DWORD], C.c_int),
        'CloseHandle': ([HANDLE], C.c_int),
        'GetCurrentProcess': ([], HANDLE),
        'DuplicateHandle': ([HANDLE, HANDLE, HANDLE, C.POINTER(HANDLE), DWORD, C.c_int, DWORD], C.c_int),
        'InitializeProcThreadAttributeList': ([HANDLE, DWORD, DWORD, C.POINTER(C.c_size_t)], C.c_int),
        'UpdateProcThreadAttribute': ([HANDLE, DWORD, C.c_size_t, HANDLE, C.c_size_t, HANDLE, HANDLE], C.c_int),
        'DeleteProcThreadAttributeList': ([HANDLE], None),
        'CreateProcessW': ([C.c_wchar_p, C.c_wchar_p, HANDLE, HANDLE, C.c_int, DWORD, HANDLE, C.c_wchar_p,
            C.POINTER(StartupInfoEx), C.POINTER(ProcessInformation)], C.c_int),
        'ResumeThread': ([HANDLE], DWORD),
        'TerminateProcess': ([HANDLE, DWORD], C.c_int),
        'WaitForSingleObject': ([HANDLE, DWORD], DWORD),
        'GetExitCodeProcess': ([HANDLE, C.POINTER(DWORD)], C.c_int),
    }
    for name, (args, result) in signatures.items():
        function = getattr(k, name); function.argtypes = args; function.restype = result
    return k

def _check(ok):
    if not ok:
        raise C.WinError(C.get_last_error())

def _error(error):
    return {'type': type(error).__name__, 'message': str(error), 'winerror': getattr(error, 'winerror', None)}

class OwnedJobInitializationError(RuntimeError):
    """Retain the newly created job when constructor cleanup needs a retry."""
    def __init__(self, original, cleanup_error, job):
        super().__init__(f'Job initialization failed ({original}); initial handle cleanup failed ({cleanup_error})')
        self.original = original; self.cleanup_error = cleanup_error; self.job = job

class OwnedProcess:
    """Popen-like pid/_handle/returncode/poll/wait; file stdio only.

    Keep this handle alive until wait/cleanup finishes. A numeric exit code259
    is correctly distinguished from a running process by waiting on its handle.
    """
    def __init__(self, kernel, information, args):
        self.k = kernel; self._handle = int(information.process); self._thread = int(information.thread)
        self.pid = int(information.pid); self.args = args; self.returncode = None; self.assigned = False

    @property
    def handle(self): return self._handle

    def poll(self):
        if self.returncode is not None: return self.returncode
        if self._handle is None: raise RuntimeError('Process handle is closed before exit was observed')
        status = self.k.WaitForSingleObject(self._handle, 0)
        if status == WAIT_TIMEOUT: return None
        if status == WAIT_FAILED: raise C.WinError(C.get_last_error())
        if status != 0: raise RuntimeError(f'Unexpected process wait status {status}')
        code = DWORD(); _check(self.k.GetExitCodeProcess(self._handle, C.byref(code)))
        self.returncode = code.value
        return self.returncode

    def wait(self, timeout=None):
        if self.returncode is not None: return self.returncode
        if timeout is not None and (not math.isfinite(timeout) or timeout < 0): raise ValueError('Invalid wait timeout')
        milliseconds = INFINITE if timeout is None else min(INFINITE - 1, math.ceil(timeout * 1000))
        status = self.k.WaitForSingleObject(self._handle, milliseconds)
        if status == WAIT_TIMEOUT: raise subprocess.TimeoutExpired(self.args, timeout)
        if status == WAIT_FAILED: raise C.WinError(C.get_last_error())
        if status != 0: raise RuntimeError(f'Unexpected process wait status {status}')
        return self.poll()

    def terminate(self, exitcode=124):
        if self.poll() is None: _check(self.k.TerminateProcess(self._handle, exitcode))

    def _close_thread(self):
        if self._thread is not None:
            _check(self.k.CloseHandle(self._thread)); self._thread = None

    def close(self):
        try: self._close_thread()
        finally:
            if self._handle is not None:
                _check(self.k.CloseHandle(self._handle)); self._handle = None

class OwnedJob:
    """Exactly one directly launched process and its inherited descendants.

    No external PID/handle assignment API is provided. No breakaway limits or
    flags are set. This object is intentionally single owner/non-concurrent.
    Explicit close is required; the OS kills this tree if the owner exits.
    """
    def __init__(self):
        self.k = _kernel(); self.handle = None; self.process = None; self.events = []
        self.handle = self.k.CreateJobObjectW(None, None)
        _check(self.handle)
        limits = ExtendedLimits(); limits.basic.flags = 0x2000 # KILL_ON_JOB_CLOSE only
        try: _check(self.k.SetInformationJobObject(self.handle, 9, C.byref(limits), C.sizeof(limits)))
        except BaseException as original:
            try: self.close()
            except BaseException as cleanup_error:
                raise OwnedJobInitializationError(original, cleanup_error, self) from original
            raise

    def active_processes(self):
        info = Accounting()
        _check(self.k.QueryInformationJobObject(self.handle, 1, C.byref(info), C.sizeof(info), None))
        return {'active_processes': info.active_processes, 'total_processes': info.total_processes,
            'terminated_processes': info.terminated_processes}

    def wait_empty(self, timeout=5.):
        deadline = time.monotonic() + timeout
        while True:
            state = self.active_processes()
            if not state['active_processes']: return state
            if time.monotonic() >= deadline: raise TimeoutError('Owned job still has active processes')
            time.sleep(min(.01, max(0., deadline - time.monotonic())))

    def terminate(self, exitcode=124):
        _check(self.k.TerminateJobObject(self.handle, exitcode))
        self.events.append({'event': 'terminate_owned_job', 'exitcode': exitcode})

    def spawn(self, args, *, cwd, env=None, stdout=None, stderr=None, _inject_assignment_failure=False):
        """Return only after the suspended child belongs to this job and runs.

        args[0] must be an absolute executable path; no shell is used. stdout/
        stderr are open file objects or None (NUL). Duplicate inheritable stdio
        handles are scoped to one explicit STARTUPINFOEX handle list. No caller
        handle inheritance flags are changed. Injection is fixture-only and
        makes the actual Assign API fail with an invalid *job* handle.
        """
        if self.handle is None or self.process is not None: raise RuntimeError('Use one launch per live OwnedJob')
        if isinstance(args, (str, bytes)) or not args: raise ValueError('Use an executable/argument sequence')
        args = [os.fspath(value) for value in args]
        if any('\0' in value for value in args): raise ValueError('NUL in command')
        if not Path(args[0]).is_absolute(): raise ValueError('Executable path must be absolute')
        cwd = str(Path(cwd).resolve())
        environment = None
        if env is not None:
            pairs = []
            for key, value in sorted(env.items(), key=lambda item: item[0].casefold()):
                if not key or '=' in key or '\0' in key or '\0' in value: raise ValueError('Invalid environment')
                pairs.append(f'{key}={value}')
            environment = C.create_unicode_buffer('\0'.join(pairs) + '\0\0')
        import msvcrt
        k = self.k; current = k.GetCurrentProcess(); handles = []; files = []; attributes = None
        child = None; assigned = False
        try:
            for supplied, mode in ((None, 'rb'), (stdout, 'wb'), (stderr, 'wb')):
                if supplied is None: supplied = open(os.devnull, mode); files.append(supplied)
                supplied.flush()
                duplicate = HANDLE()
                _check(k.DuplicateHandle(current, msvcrt.get_osfhandle(supplied.fileno()), current,
                    C.byref(duplicate), 0, True, 2)) # DUPLICATE_SAME_ACCESS
                handles.append(duplicate.value)
            size = C.c_size_t()
            k.InitializeProcThreadAttributeList(None, 1, 0, C.byref(size))
            if size.value == 0: raise C.WinError(C.get_last_error())
            storage = C.create_string_buffer(size.value)
            _check(k.InitializeProcThreadAttributeList(storage, 1, 0, C.byref(size)))
            attributes = storage
            inherited = (HANDLE * 3)(*handles)
            _check(k.UpdateProcThreadAttribute(storage, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
                inherited, C.sizeof(inherited), None, None))
            startup = StartupInfoEx(); startup.startup.cb = C.sizeof(startup); startup.attributes = C.cast(storage, HANDLE)
            startup.startup.flags = STARTF_USESTDHANDLES
            startup.startup.stdin, startup.startup.stdout, startup.startup.stderr = handles
            info = ProcessInformation(); command = C.create_unicode_buffer(subprocess.list2cmdline(args))
            flags = CREATE_SUSPENDED | CREATE_NO_WINDOW | CREATE_UNICODE_ENVIRONMENT | EXTENDED_STARTUPINFO_PRESENT
            _check(k.CreateProcessW(args[0], command, None, None, True, flags, environment, cwd, C.byref(startup), C.byref(info)))
            child = self.process = OwnedProcess(k, info, args)
            self.events.append({'event': 'created_suspended', 'pid': child.pid})
            _check(k.AssignProcessToJobObject(None if _inject_assignment_failure else self.handle, child._handle))
            assigned = child.assigned = True; member = C.c_int()
            _check(k.IsProcessInJob(child._handle, self.handle, C.byref(member)))
            if not member.value: raise RuntimeError('Assignment succeeded without job membership')
            self.events.append({'event': 'assigned_before_resume', 'pid': child.pid, 'membership_verified': True})
            previous = k.ResumeThread(child._thread)
            if previous == 0xffffffff: raise C.WinError(C.get_last_error())
            if previous != 1: raise RuntimeError(f'Unexpected initial suspend count {previous}')
            self.events.append({'event': 'resumed', 'pid': child.pid, 'previous_suspend_count': previous})
            child._close_thread()
            return child
        except BaseException as error:
            self.events.append({'event': 'launch_error', 'error': _error(error)})
            if child is not None:
                try:
                    if assigned: self.terminate()
                    else: child.terminate()
                    child.wait(timeout=5.)
                    self.events.append({'event': 'failed_launch_reaped', 'pid': child.pid, 'exitcode': child.returncode})
                except BaseException as cleanup_error:
                    self.events.append({'event': 'failed_launch_cleanup_error', 'error': _error(cleanup_error)})
            raise
        finally:
            if attributes is not None: k.DeleteProcThreadAttributeList(attributes)
            release_errors = []
            for handle in handles:
                try: _check(k.CloseHandle(handle))
                except BaseException as error: release_errors.append(error)
            for stream in files:
                try: stream.close()
                except BaseException as error: release_errors.append(error)
            if release_errors: raise BaseExceptionGroup('Stdio handle cleanup failed', release_errors)

    def close(self):
        # Close job first: KILL_ON_JOB_CLOSE protects any live descendants.
        try:
            if self.handle is not None:
                _check(self.k.CloseHandle(self.handle)); self.handle = None
        finally:
            if self.process is not None: self.process.close()

    def __enter__(self): return self
    def __exit__(self, *_): self.close()

def _save(path, data):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)

def run_owned_command(args, *, output, cwd=ROOT, env=None, timeout=10., cleanup_timeout=5.,
        _inject_assignment_failure=False, _inject_error_after_launch=False):
    """Run with file logs and persist report on launch/wait/cleanup errors.

    Standard failures are returned as a failed report. This wrapper owns a new
    output directory inside this project and never accepts existing processes.
    Primary wall includes launch, child/descendant completion and owned cleanup;
    output-directory setup and report serialization are outside that interval.
    """
    output = Path(output).resolve()
    cwd = Path(cwd).resolve()
    if output.exists(): raise FileExistsError('Preserve prior run; use a fresh output directory')
    if not math.isfinite(timeout) or timeout <= 0 or not math.isfinite(cleanup_timeout) or cleanup_timeout <= 0:
        raise ValueError('Use finite positive deadlines')
    output.mkdir(parents=True)
    report = {'schema': 'kakscuda.owned_job/2', 'command': [os.fspath(item) for item in args], 'cwd': str(cwd),
        'status': 'starting', 'PASS': False, 'official_sources': OFFICIAL_SOURCES,
        'launch_policy': 'CreateProcessW CREATE_SUSPENDED -> Assign own Job -> verify membership -> ResumeThread',
        'stdio_policy': 'Explicit inherited duplicate handle list; caller handles are unchanged',
        'timeout_seconds': timeout, 'cleanup_timeout_seconds': cleanup_timeout, 'cleanup_errors': [],
        'assignment_failure_injected': _inject_assignment_failure, 'after_launch_error_injected': _inject_error_after_launch,
        'tree_after_cleanup': {'active_processes': None, 'state': 'not_yet_observed'}}
    _save(output / 'report.json', report)
    job = child = None; started = time.perf_counter(); report['started_unix'] = time.time()
    try:
        with (output / 'stdout.log').open('wb') as out, (output / 'stderr.log').open('wb') as err:
            job = OwnedJob()
            child = job.spawn(args, cwd=cwd, env=env, stdout=out, stderr=err,
                _inject_assignment_failure=_inject_assignment_failure)
            report['child_pid'] = child.pid
            if _inject_error_after_launch: raise RuntimeError('Fixture injected exception after owned launch')
            try: report['child_exitcode'] = child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                report['timeout'] = True; report['status'] = 'timeout'; job.terminate()
                report['child_exitcode'] = child.wait(timeout=cleanup_timeout)
            if not report.get('timeout'): report['status'] = 'completed'
            try: report['tree_after_wait'] = job.wait_empty(timeout=cleanup_timeout)
            except TimeoutError:
                report['descendants_lingered'] = True; report['status'] = 'descendant_cleanup_required'
    except BaseException as error:
        report['error'] = _error(error); report['status'] = 'error'
        if isinstance(error, OwnedJobInitializationError):
            job = error.job
            report['initialization_error'] = _error(error.original)
            report['initialization_cleanup_error'] = _error(error.cleanup_error)
    finally:
        if job is not None:
            child = job.process
            if child is not None: report['child_pid'] = child.pid
            try:
                before = job.active_processes(); report['tree_before_final_cleanup'] = before
                if before['active_processes']: job.terminate()
                # Assignment failure can leave a non-member suspended process;
                # use only its retained process handle, never a PID lookup.
                if child is not None and not child.assigned and child.poll() is None: child.terminate()
                if child is not None: report['child_exitcode'] = child.wait(timeout=cleanup_timeout)
                report['tree_after_cleanup'] = job.wait_empty(timeout=cleanup_timeout)
            except BaseException as error: report['cleanup_errors'].append(_error(error))
            try: job.close()
            except BaseException as error: report['cleanup_errors'].append(_error(error))
            report['events'] = job.events
            report['job_handle_closed'] = job.handle is None
            report['process_handle_closed'] = child is None or child._handle is None
            report['thread_handle_closed'] = child is None or child._thread is None
        else:
            report['tree_after_cleanup'] = {'active_processes': 0, 'total_processes': 0, 'terminated_processes': 0}
            report['events'] = []; report['job_handle_closed'] = True
            report['process_handle_closed'] = True; report['thread_handle_closed'] = True
        report['full_process_wall_seconds'] = time.perf_counter() - started; report['ended_unix'] = time.time()
        report['cleanup_PASS'] = (not report['cleanup_errors'] and report.get('tree_after_cleanup', {}).get('active_processes') == 0
            and report['job_handle_closed'] and report['process_handle_closed'] and report['thread_handle_closed'])
        report['PASS'] = report['status'] == 'completed' and report.get('child_exitcode') == 0 and report['cleanup_PASS']
        _save(output / 'report.json', report)
    return report
