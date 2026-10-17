from pathlib import Path
import unittest,sys,tempfile,json
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from kaksturbo import api,RunConfig,MODEL_ORDER
from kaksturbo.cli import build_parser
from kaksturbo.preparation import write_input_examples
import kaksturbo,kaksforge
class Contracts(unittest.TestCase):
 def setUp(self):
  self.temp_root=Path(__file__).resolve().parents[1]/'.test-tmp';self.temp_root.mkdir(exist_ok=True)
 def test_version(self):self.assertEqual(kaksturbo.__version__,"1.1.0rc10")
 def test_legacy_alias(self):self.assertIs(kaksturbo.fit_file,kaksforge.fit_file)
 def test_small_vm_threads(self):
  with patch('kaksturbo.models.os.sched_getaffinity',return_value={0,1},create=True):
   self.assertEqual(RunConfig().threads,2)
   self.assertEqual(build_parser().parse_args(['run','--input','a','--output','b']).threads,2)
   self.assertEqual(RunConfig(threads=1).threads,1)
 def test_contract(self):
  self.assertEqual(len(MODEL_ORDER),14);self.assertEqual(RunConfig().max_iterations,500)
  with self.assertRaises(Exception):RunConfig(max_iterations=100)
 def test_no_html_default(self):self.assertFalse(build_parser().parse_args(["run","--input","a","--output","b"]).report)
 def test_input_roundtrip(self):
  with tempfile.TemporaryDirectory(dir=self.temp_root) as t:
   p=Path(t)/"examples";write_input_examples(p)
   v=api.validate(p/"pairs.axt");self.assertTrue(v["valid"]);self.assertEqual(v["pairs"],2)
 def test_bad_manifest_rejected(self):
  from kaksturbo.runtime import validate_manifest,EngineIntegrityError
  with tempfile.TemporaryDirectory(dir=self.temp_root) as t:
   p=Path(t);(p/"engine-assets.json").write_text('{}')
   with self.assertRaises(EngineIntegrityError):validate_manifest(p)
if __name__=="__main__":unittest.main()

