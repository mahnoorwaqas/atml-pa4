"""Generate private Kaggle notebook for PA4 Task 3 shared/own prefix-bank analyses."""
import argparse,json,pathlib,re
p=argparse.ArgumentParser()
p.add_argument('--owner',required=True)
p.add_argument('--run-id',required=True)
p.add_argument('--sha',required=True)
p.add_argument('--source-lock',type=pathlib.Path,required=True)
p.add_argument('--output',type=pathlib.Path,required=True)
p.add_argument('--smoke',action='store_true')
a=p.parse_args()
assert re.fullmatch('[0-9a-f]{40}',a.sha)
assert re.fullmatch('[a-z0-9-]{3,30}',a.owner)
lock=json.loads(a.source_lock.read_text())
slug=f'atml-t3-banks-{"smoke" if a.smoke else "full"}-{a.run_id}'
if len(slug)>50:raise RuntimeError('Kaggle slug too long')
training={
 'off_6304':'atml-pa4-t3-off-6304-45773884864-1',
 'off_6305':'atml-pa4-t3-off-6305-45773884864-1',
 'on_6304':'atml-pa4-t3-on-6304-45785982970-1',
 'on_6305':'atml-pa4-t3-on-6305-45773884864-1'}
cfg={'sha':a.sha,'smoke':a.smoke,'lock':lock,'training':training}
source="""
import json,pathlib,subprocess,sys,traceback,time
cfg=json.loads(CONFIG)
out=pathlib.Path('/kaggle/working/task3_banks')
out.mkdir(parents=True,exist_ok=True)
manifest={'code_commit':cfg['sha'],'smoke':cfg['smoke'],'commands':[]}
def run(cmd,cwd=None):
    print('RUN',cmd,flush=True)
    manifest['commands'].append(cmd)
    subprocess.run(cmd,cwd=cwd,check=True)
try:
    project=pathlib.Path('/kaggle/working/atml-pa4')
    staff=pathlib.Path('/kaggle/working/atml-pa4-datasets')
    run(['git','clone','--depth','1','https://github.com/mahnoorwaqas/atml-pa4.git',str(project)])
    run(['git','fetch','--depth','1','origin',cfg['sha']],project)
    run(['git','checkout',cfg['sha']],project)
    run(['git','clone','--depth','1','https://github.com/AbDu11aHHH/atml-pa4-datasets.git',str(staff)])
    run([sys.executable,'-m','pip','install','--quiet','transformers==4.57.1',
         'huggingface_hub==0.36.2','datasets','accelerate','peft==0.17.1',
         'safetensors','sentencepiece'],project)
    (staff/'data/local_source_revisions.json').write_text(json.dumps(cfg['lock'],indent=2)+'\\n')
    run([sys.executable,'scripts/setup_data.py','--public'],staff)
    merged={}
    root=pathlib.Path('/kaggle/input/notebooks')
    for key,slug in cfg['training'].items():
        candidates=[f.parent for f in root.rglob('config.json')
                    if f.parent.name=='merged' and
                    slug in str(f) and 'task3_results' in str(f)]
        if len(candidates)!=1:
            raise RuntimeError(f'Expected exactly one merged kernel attachment {key}: {candidates}')
        merged[key]=str(candidates[0])
    args=[sys.executable,'-m','task3.bank_compare','--staff',str(staff),
        '--out',str(out/'metrics')]
    for key,path in merged.items():args+=['--'+key.replace('_','-'),path]
    if cfg['smoke']:args+=['--smoke']
    run(args,project)
    done=json.loads((out/'metrics/completed.json').read_text())
    if not done['finished']:raise RuntimeError('Banks did not finish')
    manifest['result']='success'
except BaseException as exc:
    manifest.update(result='failed',error=repr(exc),traceback=traceback.format_exc())
    raise
finally:
    (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
"""
a.output.mkdir(parents=True,exist_ok=True)
nb={'cells':[{'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],
              'source':('CONFIG = '+repr(json.dumps(cfg))+'\n'+source).splitlines(keepends=True)}],
    'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}},
    'nbformat':4,'nbformat_minor':5}
meta={'id':a.owner+'/'+slug,'title':slug.replace('-',' '),
     'code_file':'kernel.ipynb','language':'python','kernel_type':'notebook',
     'is_private':True,'enable_gpu':True,'enable_internet':True,
     'dataset_sources':[],'competition_sources':[],
     'kernel_sources':[a.owner+'/'+x for x in training.values()]}
(a.output/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
(a.output/'kernel.ipynb').write_text(json.dumps(nb,indent=2)+'\n')
print(meta['id'])
