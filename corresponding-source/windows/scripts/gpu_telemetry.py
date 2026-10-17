"""Concurrent read-only NVML telemetry through one owned nvidia-smi child."""
import csv
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import time


class GPUTelemetry:
    def __init__(self, output, device_uuid, interval_ms=500):
        self.path=Path(output)/'gpu_telemetry.csv'
        self.process=None
        self.stream=None
        self.interval_ms=interval_ms
        self.device_uuid=device_uuid
        self.started_unix=time.time()
        if interval_ms:
            if interval_ms<100:
                raise ValueError('Telemetry interval must be >=100 ms or 0 to disable')
            self.stream=self.path.open('w',encoding='utf-8',newline='')
            command=['nvidia-smi',f'--id={device_uuid}',
                     '--query-gpu=timestamp,index,uuid,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu,clocks.sm,clocks.mem,pstate',
                     '--format=csv',f'--loop-ms={interval_ms}']
            try:
                self.process=subprocess.Popen(command,stdout=self.stream,stderr=subprocess.STDOUT,
                                              creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            except BaseException:
                self.stream.close()
                raise

    def close(self):
        try:
            if self.process:
                # Only stop the telemetry child created by this object.
                if self.process.poll() is None:
                    self.process.terminate()
                try:self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
                self.process=None
        finally:
            if self.stream:
                self.stream.close()
                self.stream=None

    def summary(self):
        self.close()
        data=[]
        if self.path.exists():
            with self.path.open(encoding='utf-8') as stream:
                for row in csv.DictReader(stream):
                    data.append({k.strip():v.strip() if v else '' for k,v in row.items() if k})
        result={'enabled':bool(self.interval_ms),'device_uuid':self.device_uuid,'target_interval_ms':self.interval_ms,
                'sample_count':len(data),'started_unix':self.started_unix,
                'sampling_note':'NVML utilization reports a device-defined recent sampling window; this is not per-kernel occupancy. Sampling runs concurrently; startup/stop are outside the primary timer.'}
        for prefix in ['utilization.gpu','utilization.memory','memory.used','power.draw','temperature.gpu','clocks.current.sm']:
            numbers=[]
            for row in data:
                value=next((v for k,v in row.items() if k.startswith(prefix)),None)
                match=re.match(r'^([0-9.]+)',value or '')
                if match:
                    number=float(match.group(1))
                    if math.isfinite(number):numbers.append(number)
            if numbers:
                result[prefix]={'min':min(numbers),'mean':statistics.mean(numbers),'max':max(numbers)}
                if prefix=='utilization.gpu':
                    result[prefix]['fraction_below_80_percent']=sum(n<80 for n in numbers)/len(numbers)
        return result

    def __enter__(self):return self
    def __exit__(self,*_):self.close()


def telemetry_uuid(cuda_device):
    """Resolve visible ordinal, then require caller to verify Torch's UUID."""
    ordinal=int(cuda_device.split(':')[1])
    result=subprocess.run(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],
                          capture_output=True,text=True,check=True)
    inventory=[tuple(v.strip() for v in row) for row in csv.reader(result.stdout.splitlines())]
    visible=os.environ.get('CUDA_VISIBLE_DEVICES')
    selector=str(ordinal)
    if visible:
        selectors=[v.strip() for v in visible.split(',')]
        if ordinal>=len(selectors):raise ValueError('CUDA ordinal is outside CUDA_VISIBLE_DEVICES')
        selector=selectors[ordinal]
    matches=[uuid for index,uuid in inventory if selector==index or (selector.startswith('GPU-') and uuid.startswith(selector))]
    if len(matches)!=1:raise ValueError('Cannot resolve a unique telemetry GPU UUID')
    return matches[0]
