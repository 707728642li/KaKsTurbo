"""Path IO regressions; no numerical fitting and no OS policy changes."""
from pathlib import Path
import os
import tempfile
import unittest
from kaksturbo._paths import absolute_path
from kaksturbo.checkpoint import atomic_json, load_json, contained
from kaksturbo.errors import IntegrityError
from kaksturbo.preparation import write_input_examples
from kaksturbo import api


@unittest.skipUnless(os.name == 'nt', 'Windows filesystem contract')
class WindowsPaths(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.root = absolute_path(self.temp.name)
        self.deep = self.root / ('deep '+ 'a'*90) / ('目录 with spaces '+ 'b'*90)
        self.deep.mkdir(parents=True)

    def tearDown(self):
        # Python's TemporaryDirectory cleanup also needs the extended root.
        import shutil
        shutil.rmtree(self.root)
        self.temp.cleanup()

    def ordinary(self,p):return str(p).removeprefix('\\\\?\\')

    def test_prefix_relative_and_already_extended(self):
        p = absolute_path(self.ordinary(self.deep/'..'/'other'))
        self.assertEqual(p,self.deep.parent/'other')
        self.assertEqual(absolute_path(p),p)

    def test_atomic_checkpoint_at_long_path(self):
        p=self.deep/'checkpoint.json'
        atomic_json(p,{'complete':True,'names':['forward','reverse','NA']})
        self.assertEqual(load_json(p)['names'],['forward','reverse','NA'])

    def test_input_examples_validate_long_paths(self):
        out=self.deep/'prepared'
        write_input_examples(self.ordinary(out))
        result=api.validate(self.ordinary(out/'pairs.axt'))
        self.assertTrue(result['valid'])
        self.assertEqual(result['pairs'],2)

    def test_output_containment_retained(self):
        with self.assertRaises(IntegrityError):contained(self.deep,'../outside.json')

    def test_windows_device_namespace_rejected(self):
        with self.assertRaises(ValueError):absolute_path(r'\\.\PhysicalDrive0')

if __name__=='__main__':unittest.main()
