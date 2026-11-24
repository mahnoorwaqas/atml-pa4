#!/usr/bin/env python3
"""Create private run-specific Kaggle kernel with an explicit Task 1 condition."""
import argparse,json,pathlib,re
p=argparse.ArgumentParser()
p.add_argument("--owner",required=True);p.add_argument("--run-id",required=True)
p.add_argument("--experiment",required=True);p.add_argument("--sha",required=True)
p.add_argument("--output",type=pathlib.Path,required=True)
a=p.parse_args()
if not re.fullmatch(r"[0-9a-f]{40}",a.sha):raise ValueError("Full SHA required")
if not re.fullmatch(r"[a-z0-9_-]+",a.owner):raise ValueError("Invalid username")
slug="atml-pa4-"+a.run_id
if not re.fullmatch(r"[a-z0-9-]{1,50}",slug):raise ValueError("Invalid slug")
conditions={"baseline":("baseline",0,0),"mag25":("magnitude",.25,0),
"mag50":("magnitude",.5,0),"mag75":("magnitude",.75,0),
"wanda25":("wanda",.25,0),"wanda50":("wanda",.5,0),"wanda75":("wanda",.75,0),
"adaptive50":("nonuniform",.5,0),"adaptive75":("nonuniform",.75,0),
"structured50":("structured",.5,0),
"int8":("ptq",0,8),"int4":("ptq",0,4),"int3":("ptq",0,3),
"awq8":("awq",0,8),"awq4":("awq",0,4),"awq3":("awq",0,3),
"mixed4":("mixed",0,4),"mixed3":("mixed",0,3)}
if a.experiment not in ("dataset-check",*conditions):raise ValueError(a.experiment)
a.output.mkdir(parents=True,exist_ok=True)
meta={"id":a.owner+"/"+slug,"title":"ATML PA4 "+a.experiment+" "+a.run_id,
"code_file":"kernel.ipynb","language":"python","kernel_type":"notebook",
"is_private":True,"enable_gpu":a.experiment!="dataset-check",
"enable_internet":True,"dataset_sources":[],"competition_sources":[],"kernel_sources":[]}
config={"experiment":a.experiment,"sha":a.sha,"condition":conditions.get(a.experiment)}
code="""
import json,pathlib,subprocess,sys,time,traceback,shutil
cfg=json.loads(CONFIG)
out=pathlib.Path('/kaggle/working/pa4_outputs')
out.mkdir(parents=True,exist_ok=True)
status={'config':cfg,'commands':[]}
t0=time.monotonic()
def run(argv,cwd=None):
    status['commands'].append({'argv':argv,'cwd':str(cwd) if cwd else None})
    print('RUN',argv,flush=True)
    subprocess.run(argv,cwd=cwd,check=True)
try:
    staff=pathlib.Path('/kaggle/working/atml-pa4-datasets')
    run(['git','clone','--depth','1','https://github.com/AbDu11aHHH/atml-pa4-datasets.git',str(staff)])
    run([sys.executable,'scripts/validate_assets.py'],staff)
    run([sys.executable,'scripts/verify_checksums.py'],staff)
    if cfg['experiment']=='dataset-check':
        run([sys.executable,'-m','unittest','discover','-s','tests','-v'],staff)
    else:
        project=pathlib.Path('/kaggle/working/atml-pa4')
        run(['git','clone','https://github.com/mahnoorwaqas/atml-pa4.git',str(project)])
        run(['git','checkout',cfg['sha']],project)
        run([sys.executable,'-m','pip','install','--quiet',
             'transformers==4.57.1','huggingface_hub==0.36.2',
             'datasets','accelerate','safetensors','sentencepiece'])
        run([sys.executable,'scripts/setup_data.py','--public'],staff)
        run([sys.executable,'scripts/check_setup.py'],staff)
        shutil.copyfile(staff/'data/local_source_revisions.json',out/'source_revisions.json')
        method,sparsity,bits=cfg['condition']
        path=out/('task1_'+cfg['experiment'])
        args=[sys.executable,'task1/run.py','--data',str(staff),'--output',str(path),
              '--method',method]
        if sparsity:args.extend(['--sparsity',str(sparsity)])
        if bits:args.extend(['--bits',str(bits)])
        if cfg['experiment'] not in ('baseline','mag50','wanda50','adaptive50','int4','int3','awq4','awq3','mixed4','mixed3'):
            args.append('--diagnostic-only')
        run(args,project)
    status['result']='success'
except BaseException as exc:
    status.update(result='failed',error=repr(exc),traceback=traceback.format_exc())
    raise
finally:
    status['elapsed_seconds']=time.monotonic()-t0
    (out/'run_manifest.json').write_text(json.dumps(status,indent=2))
"""
src="CONFIG = "+repr(json.dumps(config))+"\n"+code
nb={"cells":[{"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],
"source":src.splitlines(keepends=True)}],
"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
"language_info":{"name":"python"}},"nbformat":4,"nbformat_minor":5}
(a.output/"kernel-metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
(a.output/"kernel.ipynb").write_text(json.dumps(nb,indent=2)+"\n")
