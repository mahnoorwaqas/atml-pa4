"""Create private Task 3 evaluation notebook using a completed training kernel as data source."""
import argparse,json,pathlib,re
p=argparse.ArgumentParser()
p.add_argument('--owner',required=True)
p.add_argument('--run-id',required=True)
p.add_argument('--sha',required=True)
p.add_argument('--source-lock',type=pathlib.Path,required=True)
p.add_argument('--condition',required=True)
p.add_argument('--seed',type=int,required=True)
p.add_argument('--training-run-id',default=None)
p.add_argument('--output',type=pathlib.Path,required=True)
a=p.parse_args()
if not re.fullmatch('[a-z0-9-]{3,30}',a.owner):raise ValueError('Invalid Kaggle owner')
if not re.fullmatch('[0-9a-f]{40}',a.sha):raise ValueError('Bad source revision')
if a.condition!='original' and a.condition not in ('off','on'):
    raise ValueError('Bad condition')
if a.condition!='original' and (not a.training_run_id or a.seed not in (6304,6305)):
    raise ValueError('Missing completed training kernel')
lock=json.loads(a.source_lock.read_text())
slug=f'atml-pa4-t3-eval-{a.condition}-{a.seed}-{a.run_id}'
if len(slug)>50:raise ValueError('Kaggle slug too long')
training_kernel=(f'{a.owner}/atml-pa4-t3-{a.condition}-{a.seed}-{a.training_run_id}'
                 if a.condition!='original' else None)
cfg={'sha':a.sha,'lock':lock,'condition':a.condition,'seed':a.seed,'training_kernel':training_kernel}
code="""
import json,pathlib,subprocess,sys,traceback,time
cfg=json.loads(CONFIG)
out=pathlib.Path('/kaggle/working/task3_evaluation')
out.mkdir(parents=True,exist_ok=True)
manifest={'condition':cfg['condition'],'seed':cfg['seed'],'commands':[]}
def run(args,cwd=None):
    print('RUN',args,flush=True)
    manifest['commands'].append(args)
    subprocess.run(args,cwd=cwd,check=True)
try:
    project=pathlib.Path('/kaggle/working/atml-pa4')
    staff=pathlib.Path('/kaggle/working/atml-pa4-datasets')
    run(['git','clone','--depth','1','https://github.com/mahnoorwaqas/atml-pa4.git',str(project)])
    run(['git','fetch','--depth','1','origin',cfg['sha']],project)
    run(['git','checkout',cfg['sha']],project)
    run(['git','clone','--depth','1','https://github.com/AbDu11aHHH/atml-pa4-datasets.git',str(staff)])
    run([sys.executable,'-m','pip','install','--quiet','transformers==4.57.1',
         'huggingface_hub==0.36.2','datasets','accelerate',
         'safetensors','sentencepiece'],project)
    (staff/'data/local_source_revisions.json').write_text(json.dumps(cfg['lock'],indent=2)+'\\n')
    run([sys.executable,'scripts/setup_data.py','--public'],staff)
    run([sys.executable,'scripts/check_setup.py'],staff)
    draft='Qwen/Qwen2.5-0.5B-Instruct'
    if cfg['condition']!='original':
        dirs=[p.parent for p in pathlib.Path('/kaggle/input').rglob('config.json')
              if p.parent.name=='merged' and 'task3_results' in str(p)]
        if len(dirs)!=1:raise RuntimeError('Expected exactly one attached merged checkpoint: '+str(dirs))
        draft=str(dirs[0])
    run([sys.executable,'-m','task3.evaluate','--staff',str(staff),
       '--out',str(out/'metrics'),'--draft',draft,
       '--condition',cfg['condition'],'--seed',str(cfg['seed'])],project)
    if json.loads((out/'metrics/completed.json').read_text())['finished']!=True:
        raise RuntimeError('No completed evaluation')
    manifest['result']='success'
except BaseException as e:
    manifest.update(result='failed',error=repr(e),traceback=traceback.format_exc())
    raise
finally:
    (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
"""
a.output.mkdir(parents=True,exist_ok=True)
meta={'id':a.owner+'/'+slug,'title':slug.replace('-',' '),
      'code_file':'kernel.ipynb','language':'python','kernel_type':'notebook',
      'is_private':True,'enable_gpu':True,'enable_internet':True,
      'dataset_sources':[],'competition_sources':[],
      'kernel_sources':([training_kernel] if training_kernel else [])}
notebook={'cells':[{'cell_type':'code','metadata':{},'outputs':[],
                    'execution_count':None,
                    'source':('CONFIG = '+repr(json.dumps(cfg))+'\n'+code).splitlines(keepends=True)}],
          'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3',
                                   'language':'python'}},
          'nbformat':4,'nbformat_minor':5}
(a.output/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
(a.output/'kernel.ipynb').write_text(json.dumps(notebook,indent=2)+'\n')
print(meta['id'])
