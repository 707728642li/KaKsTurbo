"""Availability failures are not numerical failures and never trigger silent fallback."""
import os,subprocess,unittest
from unittest.mock import patch
from kaksturbo.devices import DeviceError, choose, inventory, check_gpu_package

def card(i=0,free=22000,util=0):
    return dict(index=i,uuid=f'GPU-{i:08d}-0000-0000-0000-000000000000',name='test',total_MiB=24000,used_MiB=24000-free,utilization_percent=util)
class Devices(unittest.TestCase):
    def setUp(self):
        self.env=patch.dict(os.environ,{},clear=True);self.env.start()
    def tearDown(self):self.env.stop()
    def reason(self,expected,call):
        with self.assertRaises(DeviceError) as cm:call()
        self.assertEqual(cm.exception.details['reason'],expected)
        self.assertIn('--backend cpu',cm.exception.details['CPU_alternative'])
    def test_missing_tool(self):
        with patch('kaksturbo.devices.subprocess.run',side_effect=FileNotFoundError):self.reason('nvidia_smi_missing',inventory)
    def test_driver_failure(self):
        with patch('kaksturbo.devices.subprocess.run',side_effect=subprocess.CalledProcessError(1,['nvidia-smi'],stderr='driver mismatch')):self.reason('driver_query_failed',inventory)
    def test_no_device(self):
        with patch('kaksturbo.devices.subprocess.run',return_value=subprocess.CompletedProcess([],0,stdout='',stderr='')):self.reason('no_gpu',inventory)
    def test_timeout(self):
        with patch('kaksturbo.devices.subprocess.run',side_effect=subprocess.TimeoutExpired([],20)):self.reason('driver_timeout',inventory)
    def test_prefer_idle_one(self):
        s={'cards':[card(0,24000),card(1,24000)]};self.assertEqual(choose({},s)[0]['index'],1)
    def test_do_not_wait_for_one(self):
        s={'cards':[card(0,24000),card(1,1000,100)]};self.assertEqual(choose({},s)[0]['index'],0)
    def test_memory(self):self.reason('insufficient_memory',lambda:choose({}, {'cards':[card(0,8000)]}))
    def test_busy(self):self.reason('gpu_busy_or_unknown',lambda:choose({}, {'cards':[card(0,9000,100)]}))
    def test_unknown_util(self):self.reason('gpu_busy_or_unknown',lambda:choose({}, {'cards':[card(0,9000,None)]}))
    def test_visibility_empty(self):
        os.environ['CUDA_VISIBLE_DEVICES']='';self.reason('gpu_hidden',lambda:choose({}, {'cards':[card()]}))
    def test_visibility_minus_one(self):
        os.environ['CUDA_VISIBLE_DEVICES']='-1';self.reason('gpu_hidden',lambda:choose({}, {'cards':[card()]}))
    def test_wrong_device(self):self.reason('device_not_visible',lambda:choose({'device':'cuda:3'},{'cards':[card()]}))
    def test_visibility_physical(self):
        os.environ['CUDA_VISIBLE_DEVICES']='1';self.assertEqual(choose({'device':'cuda:1'},{'cards':[card(0),card(1)]})[0]['index'],1)
    def test_visibility_duplicate(self):
        os.environ['CUDA_VISIBLE_DEVICES']='0,0';self.reason('invalid_visibility',lambda:choose({}, {'cards':[card()]}))
    def test_explicit_busy_no_overallocate(self):
        c=choose({'device':'cuda:0'},{'cards':[card(0,10000,99)]})[0];self.assertEqual(c['admission_minimum_free_MiB'],8192)
    def test_backend_absent(self):
        with patch('pathlib.Path.is_file',return_value=False):self.reason('gpu_backend_not_packaged',lambda:check_gpu_package('fake'))
if __name__=='__main__':unittest.main()
