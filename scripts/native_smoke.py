"""Bounded package qualification, not a benchmark or scientific validation study."""
from pathlib import Path
import argparse,csv,hashlib,itertools,json,os,platform,subprocess,sys,time,traceback

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--installed',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 target=Path(a.installed).resolve();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
 os.environ['TMPDIR']=os.environ['TEMP']=os.environ['TMP']=str(out/'tmp');(out/'tmp').mkdir()
 # Ignore inherited engine overrides; exercise the wheel's bundled assets.
 for key in ('KAKSTURBO_RUNTIME_ROOT','KAKSFORGE_RUNTIME_ROOT','PYTHONPATH'):os.environ.pop(key,None)
 boot=out/'cli.py';boot.write_text('import sys\nsys.path.insert(0,'+repr(str(target))+')\nfrom kaksturbo.cli import main\nraise SystemExit(main())\n')
 evidence={'schema':'kaksturbo.local_package_smoke/1','platform':platform.platform(),'python':sys.version,'scope':'fresh target on existing host; CPU only; not a clean OS or benchmark','checks':[]}
 def run(name,args,expected=0,machine=False):
  cmd=[sys.executable,'-I','-S','-B',str(boot),*map(str,args)];t=time.perf_counter()
  r=subprocess.run(cmd,cwd=out,capture_output=True,text=True,encoding='utf8',timeout=180)
  (out/(name+'.stdout.txt')).write_text(r.stdout,encoding='utf8');(out/(name+'.stderr.txt')).write_text(r.stderr,encoding='utf8')
  check={'name':name,'argv':cmd,'exit_code':r.returncode,'expected':expected,'elapsed_seconds':time.perf_counter()-t,'PASS':r.returncode==expected};evidence['checks'].append(check)
  assert check['PASS'],f'{name}: exit {r.returncode}: {r.stderr[-2000:]}'
  return json.loads(r.stdout) if machine else r.stdout
 try:
  assert 'KaKsTurbo 1.1.0rc6' in run('version',['--version'])
  assert 'resume' in run('help',['--help'])
  run('examples',['examples','--output',out/'examples','--json'],machine=True)
  v=run('validate',['validate','--input',out/'examples/pairs.axt','--json'],machine=True);assert v['valid'] and v['pairs']==2
  run('prepare',['prepare','--input',out/'examples/aligned.fasta','--pairs',out/'examples/pairs.tsv','--output',out/'prepared','--json'],machine=True)
  d=run('doctor',['doctor','--backend','cpu','--json'],machine=True);assert d['PASS']
  run('invalid_cli',['run','--input','missing','--output',out/'invalid','--threads','0','--json'],expected=2)
  bad=json.loads((out/'invalid_cli.stderr.txt').read_text(encoding='utf8'));assert bad.get('error')
  codons=[''.join(x) for x in itertools.product('ACGT',repeat=3) if ''.join(x) not in ('TAA','TAG','TGA')]
  left=codons*2;right=left.copy()
  for i in range(3,len(right),13):right[i]=codons[(codons.index(right[i])+5)%len(codons)]
  s1,s2=''.join(left),''.join(right);fixture=out/'synthetic.axt'
  fixture.write_text(f'forward\n{s1}\n{s2}\n\nreverse\n{s2}\n{s1}\n\nidentical\n{s1}\n{s1}\n',encoding='ascii')
  evidence['input_sha256']=digest(fixture)
  result=out/'result';r=run('run',['run','--input',fixture,'--output',result,'--backend','cpu','--threads','2','--chunk-pairs','3','--max-seconds','120','--json'],machine=True)
  assert r['complete'] and r['count']==3
  rows=list(csv.DictReader((result/'result.tsv').open(encoding='utf8'),delimiter='\t'))
  models=[json.loads(x) for x in (result/'models.jsonl').read_text().splitlines()]
  assert [x['pair_id'] for x in rows]==['forward','reverse','identical'] and len(models)==42
  assert rows[-1]['Ka_Ks']=='NA','Identical-pair NA must remain explicit'
  assert not (result/'report.html').exists()
  checkpoint=json.loads((result/'checkpoint.json').read_text())
  before={n:digest(result/n) for n in checkpoint['final_artifacts_sha256']}
  run('status',['status',result,'--json'],machine=True)
  resumed=run('resume',['resume',result,'--json'],machine=True);assert resumed['complete']
  assert before=={n:digest(result/n) for n in before}
  evidence.update(PASS=True,rows=rows,model_slots=len(models),quality=json.loads((result/'quality_summary.json').read_text()),final_sha256=before,resume_preserved_outputs=True)
 except Exception as exc:
  evidence.update(PASS=False,error=str(exc),traceback=traceback.format_exc())
 finally:
  (out/'proof.json').write_text(json.dumps(evidence,indent=2,ensure_ascii=False),encoding='utf8')
 print(json.dumps({'PASS':evidence['PASS'],'proof':str(out/'proof.json'),'error':evidence.get('error')}))
 return 0 if evidence['PASS'] else 1
if __name__=='__main__':raise SystemExit(main())
