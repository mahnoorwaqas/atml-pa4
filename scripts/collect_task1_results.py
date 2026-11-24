"""Collect one explicitly indexed completed result per Task 1 condition from private HF Hub."""
import argparse,json,pathlib,re
from collections import defaultdict
from huggingface_hub import HfApi,hf_hub_download

p=argparse.ArgumentParser()
p.add_argument('--output',required=True,type=pathlib.Path)
p.add_argument('--allow-partial',action='store_true')
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
api=HfApi()
repo='mahnoorwaqas/atml-pa4-results'
files=api.list_repo_files(repo_id=repo,repo_type='dataset')
regexp=re.compile(r'^task1/([^/]+)/github-run-(\d+)-attempt-(\d+)/completed.json$')
available=defaultdict(list)
for f in files:
    match=regexp.match(f)
    if match:available[match.group(1)].append((int(match.group(2)),int(match.group(3))))
expected=('baseline','mag25','mag50','mag75','wanda25','wanda50','wanda75',
          'adaptive50','adaptive75','structured50',
          'int8','int4','int3','awq8','awq4','awq3','mixed4','mixed3')
missing=sorted(set(expected)-set(available))
if missing and not a.allow_partial:raise RuntimeError('Missing completed conditions '+str(missing))
if 'baseline' not in available:raise RuntimeError('Need completed baseline')
selected={}
for experiment in expected:
    if experiment not in available:continue
    run,attempt=max(available[experiment])
    prefix=f'task1/{experiment}/github-run-{run}-attempt-{attempt}'
    names=[x for x in files if x.startswith(prefix+'/')]
    folder=a.output/f'task1_{experiment}'
    folder.mkdir(parents=True,exist_ok=True)
    for name in names:
        if not name.endswith('.json'):continue
        source=hf_hub_download(repo_id=repo,repo_type='dataset',filename=name)
        (folder/pathlib.PurePosixPath(name).name).write_bytes(pathlib.Path(source).read_bytes())
    if not (folder/'completed.json').is_file():
        raise RuntimeError('Incomplete archive '+prefix)
    selected[experiment]={'run':run,'attempt':attempt,'path':prefix}
(a.output/'collection_manifest.json').write_text(
    json.dumps({'selected':selected,'missing':missing,
                'other_runs':{k:v for k,v in available.items()}},indent=2)+'\n')
print('Selected',len(selected),'conditions from private Hugging Face repo; missing:',missing)
