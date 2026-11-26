"""Generate pinned, private Kaggle Task 3 training notebook."""
import argparse,json,pathlib,re
p=argparse.ArgumentParser()
p.add_argument('--owner',required=True)
p.add_argument('--run-id',required=True)
p.add_argument('--sha',required=True)
p.add_argument('--source-lock',type=pathlib.Path,required=True)
p.add_argument('--method',choices=('off','on'),required=True)
p.add_argument('--seed',type=int,choices=(6304,6305),required=True)
p.add_argument('--output',type=pathlib.Path,required=True)
p.add_argument('--smoke',action='store_true')
a=p.parse_args()
if not re.fullmatch('[0-9a-f]{40}',a.sha):raise ValueError('Missing pinned code SHA')
if not re.fullmatch('[a-z0-9-]{3,30}',a.owner):raise ValueError('Invalid username')
prefix='atml-pa4-t3-smoke' if a.smoke else 'atml-pa4-t3'
slug=f'{prefix}-{a.method}-{a.seed}-{a.run_id}'
if len(slug)>50:raise ValueError('Kaggle slug too long')
a.output.mkdir(parents=True,exist_ok=True)
lock=json.loads(a.source_lock.read_text())
required={'gsm8k','model_draft','model_target','model_base','c4','wikitext','hellaswag'}
if not required.issubset(lock['revisions']):raise ValueError('Invalid pinned revision lock')
cfg={'sha':a.sha,'method':a.method,'seed':a.seed,'lock':lock,'smoke':a.smoke}
source="""
import json,pathlib,subprocess,sys,time,traceback,os
cfg=json.loads(CONFIG)
out=pathlib.Path('/kaggle/working/task3_results')
out.mkdir(parents=True,exist_ok=True)
manifest={'config':{k:cfg[k] for k in ['sha','method','seed']},
          'commands':[],'started_unix':time.time()}
def run(args,cwd=None):
    print('RUN',args,flush=True)
    manifest['commands'].append({'args':args,'cwd':str(cwd)})
    subprocess.run(args,cwd=cwd,check=True)
try:
    assert os.environ.get('CUDA_VISIBLE_DEVICES','none')!='', 'No GPU'
    project=pathlib.Path('/kaggle/working/atml-pa4')
    staff=pathlib.Path('/kaggle/working/atml-pa4-datasets')
    run(['git','clone','--depth','1','https://github.com/mahnoorwaqas/atml-pa4.git',str(project)])
    run(['git','fetch','--depth','1','origin',cfg['sha']],project)
    run(['git','checkout',cfg['sha']],project)
    run(['git','clone','--depth','1','https://github.com/AbDu11aHHH/atml-pa4-datasets.git',str(staff)])
    run([sys.executable,'scripts/validate_assets.py'],staff)
    run([sys.executable,'scripts/audit_staff.py',str(staff),str(out/'staff_checksum_audit.json')],project)
    run([sys.executable,'-m','pip','install','--quiet','transformers==4.57.1',
         'huggingface_hub==0.36.2','datasets','peft==0.17.1','accelerate',
         'safetensors','sentencepiece','scipy'],project)
    (staff/'data/local_source_revisions.json').write_text(json.dumps(cfg['lock'],indent=2)+'\\n')
    run([sys.executable,'scripts/setup_data.py','--public'],staff)
    run([sys.executable,'scripts/check_setup.py'],staff)
    (out/'source_revisions.json').write_text(json.dumps(cfg['lock'],indent=2)+'\\n')
    run([sys.executable,'task3/train.py','--staff',str(staff),
         '--out',str(out/'training'),'--method',cfg['method'],
         '--seed',str(cfg['seed'])]+(['--smoke'] if cfg['smoke'] else []),project)
    completed=json.loads((out/'training/completed.json').read_text())
    if completed.get('finished')!=True:raise RuntimeError('No finished marker')
    manifest['result']='success'
except BaseException as e:
    manifest.update(result='failed',error=repr(e),traceback=traceback.format_exc())
    raise
finally:
    manifest['finished_unix']=time.time()
    (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
"""
nb={'cells':[{'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],
    'source':('CONFIG = '+repr(json.dumps(cfg))+'\n'+source).splitlines(keepends=True)}],
    'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},
    'nbformat':4,'nbformat_minor':5}
metadata={'id':f'{a.owner}/{slug}','title':slug.replace('-', ' '),
    'code_file':'kernel.ipynb','language':'python','kernel_type':'notebook',
    'is_private':True,'enable_gpu':True,'enable_internet':True,
    'dataset_sources':[],'competition_sources':[],'kernel_sources':[]}
(a.output/'kernel.ipynb').write_text(json.dumps(nb,indent=2)+'\n')
(a.output/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
print(metadata['id'])
