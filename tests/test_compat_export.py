"""Export integration/fault tests using synthetic saved records, never fitting."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from kaksturbo.checkpoint import file_sha256
from kaksturbo.compat_export import LEGACY_FIELDS, REPRODUCTION_FIELDS, ROBUST_OMISSIONS, export_kaks_calculator
from kaksturbo.errors import ConfigurationError, IntegrityError
from kaksturbo.models import MODEL_ORDER
from kaksturbo.results import CORE_FIELDS, format_row


class Export(unittest.TestCase):
    def setUp(self):
        # Test execution chooses a project-owned current directory on each OS.
        self.temp = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.root = Path(self.temp.name)
        self.source = self.root / 'source'; self.source.mkdir()
        self.output = self.root / 'export'
        self.fixture()

    def tearDown(self):
        self.temp.cleanup()

    def fixture(self, mode='robust', ids=('forward', 'reverse', 'NA'), na=False):
        self.rows = []
        native = []
        for i, pair in enumerate(ids):
            row = dict.fromkeys(CORE_FIELDS)
            row.update(pair_id=pair, method='MA', mode=mode, backend='cpu', Ka=0.1, Ks=0.2,
                       Ka_Ks=0.7, aligned_codons=100, eligible_codons=90, observed_SNP=4,
                       status='available', lnL=-123.45678901234567, iteration_capped_models=[], boundary_models=[])
            if i == len(ids)-1 or na:
                row.update(Ka=None, Ks=None, Ka_Ks=None, lnL=None, status='unidentifiable', reason='zero_observed_substitutions')
            self.rows.append(row)
            base = {'Sequence': pair, 'Method': 'MA', 'Ka': 'NA' if row['Ka'] is None else '.1',
                    'Ks': 'NA' if row['Ks'] is None else '.2', 'Ka/Ks': 'NA' if row['Ka_Ks'] is None else '.7',
                    'S-sites': '90.125', 'N-sites': '179.875'}
            if mode == 'reproduce':
                old = dict.fromkeys(LEGACY_FIELDS, 'NA');old.update({k:v for k,v in base.items() if k in old})
                old.update({'P-Value(Fisher)': '.000123', 'Length': '270', 'GC(1:2:3)': '0.5(0.4:0.5:0.6)'})
                base = old
            native.append(base)
        with (self.source/'result.tsv').open('w',newline='') as f:
            w=csv.DictWriter(f,CORE_FIELDS,delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(format_row(x) for x in self.rows)
        with (self.source/'native_result.tsv').open('w',newline='') as f:
            w=csv.DictWriter(f,list(native[0]),delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(native)
        self.write_json()
        self.manifest={'schema':'kaksturbo.run_manifest/2','complete':True,'version':'1.1.0rc7','mode':mode,
            'method':'MA','model_count':14,'models':list(MODEL_ORDER),'count':len(ids),'backends_used':['cpu']}
        self.seal()

    def write_json(self):
        (self.source/'results.jsonl').write_text(''.join(json.dumps({'global_pair_index':i,'pair_id':x['pair_id'],'record':x})+'\n' for i,x in enumerate(self.rows)))

    def seal(self):
        self.manifest['artifacts_sha256']={n:file_sha256(self.source/n) for n in ('result.tsv','native_result.tsv','results.jsonl')}
        (self.source/'run_manifest.json').write_text(json.dumps(self.manifest))

    def run_export(self):
        return export_kaks_calculator(self.source,self.output)

    def read(self):
        with (self.output/'result.kaks').open() as f:return list(csv.DictReader(f,delimiter='\t',quoting=csv.QUOTE_NONE))

    def test_layout_values_precision_and_no_ratio_recalculation(self):
        before={p.name:file_sha256(p) for p in self.source.iterdir()}
        self.run_export();r=self.read()
        self.assertEqual(list(r[0]),list(LEGACY_FIELDS));self.assertEqual(len(r),3)
        self.assertEqual(float(r[0]['Ka/Ks']),.7);self.assertNotEqual(float(r[0]['Ka/Ks']),.1/.2)
        self.assertEqual(r[0]['Length'],'270');self.assertEqual(r[0]['S-Sites'],'90.125')
        self.assertEqual(float(r[0]['ML-Score']),self.rows[0]['lnL'])
        self.assertEqual(r[-1]['Sequence'],'NA');self.assertEqual(r[-1]['Ka/Ks'],'NA')
        self.assertEqual(before,{p.name:file_sha256(p) for p in self.source.iterdir()})

    def test_every_missing_statistic_has_reason(self):
        self.run_export();r=self.read()
        with (self.output/'missing_fields.tsv').open() as f:why=list(csv.DictReader(f,delimiter='\t'))
        missing={(i,k) for i,x in enumerate(r) for k in LEGACY_FIELDS[2:] if x[k]=='NA'}
        self.assertEqual(missing,{(int(x['pair_index']),x['field']) for x in why})
        self.assertTrue(all(r[0][k]=='NA' for k in ROBUST_OMISSIONS))

    def test_reproduction_native_cells_preserved(self):
        self.fixture('reproduce');self.run_export()
        self.assertEqual((self.output/'result.kaks').read_text(),(self.source/'native_result.tsv').read_text())

    def test_special_identifiers_not_rewritten(self):
        self.fixture(ids=('001', 'name with spaces # "quote"', 'NA'))
        self.run_export();self.assertEqual([r['Sequence'] for r in self.read()],[r['pair_id'] for r in self.rows])

    def test_actual_historical_header_variant_keeps_all_cells(self):
        self.fixture('reproduce')
        p=self.source/'native_result.tsv'
        cells=p.read_text().splitlines()
        p.write_text('\t'.join(REPRODUCTION_FIELDS)+'\n'+'\n'.join(cells[1:])+'\n')
        self.seal();self.run_export()
        self.assertEqual((self.output/'result.kaks').read_text().splitlines()[1:],cells[1:])
        info=json.loads((self.output/'compatibility.json').read_text())
        self.assertEqual(info['mapping']['output_to_native_header_aliases']['Divergence-Distance'],'Divergence-Time')

    def test_archived_nonconsumed_details_do_not_block_core_export(self):
        self.manifest['artifacts_sha256']['models.jsonl']='0'*64
        (self.source/'run_manifest.json').write_text(json.dumps(self.manifest))
        self.run_export()
        info=json.loads((self.output/'compatibility.json').read_text())
        self.assertEqual(info['unverified_nonconsumed_artifacts'],['models.jsonl'])
        self.assertNotIn('models.jsonl',info['verified_source_artifacts_sha256'])

    def test_changed_final_file_rejected(self):
        with (self.source/'result.tsv').open('a') as f:f.write('tamper\n')
        with self.assertRaises(IntegrityError):self.run_export()
        self.assertFalse(self.output.exists())

    def test_incomplete_run_rejected(self):
        self.manifest['complete']=False;self.seal()
        with self.assertRaises(IntegrityError):self.run_export()

    def test_json_cross_representation_mismatch_rejected(self):
        self.rows[0]['Ka']=.3;self.write_json();self.seal()
        with self.assertRaises(IntegrityError):self.run_export()
        self.assertFalse(self.output.exists())

    def test_count_mismatch_rejected(self):
        self.manifest['count']=4;self.seal()
        with self.assertRaises(IntegrityError):self.run_export()
        self.assertFalse(self.output.exists())

    def test_missing_native_row_rejected(self):
        p=self.source/'native_result.tsv';p.write_text('\n'.join(p.read_text().splitlines()[:-1])+'\n');self.seal()
        with self.assertRaises(IntegrityError):self.run_export()

    def test_wrong_model_coverage_rejected(self):
        self.manifest['model_count']=13;self.seal()
        with self.assertRaises(IntegrityError):self.run_export()

    def test_missing_registered_table_rejected(self):
        self.manifest['artifacts_sha256'].pop('results.jsonl')
        (self.source/'run_manifest.json').write_text(json.dumps(self.manifest))
        with self.assertRaises(IntegrityError):self.run_export()

    def test_existing_output_preserved(self):
        self.output.mkdir();p=self.output/'keep';p.write_text('original')
        with self.assertRaises(ConfigurationError):self.run_export()
        self.assertEqual(p.read_text(),'original')

    def test_cannot_export_into_original_run(self):
        with self.assertRaises(ConfigurationError):export_kaks_calculator(self.source,self.source/'export')

    def test_no_work_directory_or_original_input_required(self):
        self.assertFalse((self.source/'.work').exists());self.assertTrue(self.run_export()['complete'])


if __name__=='__main__':unittest.main()
