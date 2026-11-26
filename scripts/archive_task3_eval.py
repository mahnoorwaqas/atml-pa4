"""Private archive of completed Task 3 per-prompt speed and quality evaluation."""
import argparse,json,os,pathlib
from huggingface_hub import HfApi
p=argparse.ArgumentParser()
p.add_argument('--download-dir',type=pathlib.Path,required=True)
p.add_argument('--condition',required=True)
p.add_argument('--seed',type=int,required=True)
p.add_argument('--github-run-id',required=True)
a=p.parse_args()
folder=a.download_dir/'task3_evaluation'
metrics=folder/'metrics'
if not (metrics/'completed.json').exists():raise RuntimeError('Missing Task 3 eval completion')
marker=json.loads((metrics/'completed.json').read_text())
if not marker['finished'] or marker['primary_cases']!=480 or marker['quality_cases']!=120:
    raise RuntimeError('Incorrect/incomplete Task 3 evaluation')
api=HfApi(token=os.environ['HF_TOKEN'])
repo='mahnoorwaqas/atml-pa4-results'
if not api.repo_info(repo,repo_type='dataset').private:
    raise RuntimeError('Refusing a public evaluation archive')
base=f'task3/evaluations/{a.condition}_{a.seed}/github-run-{a.github_run_id}'
api.upload_folder(repo_id=repo,repo_type='dataset',folder_path=str(metrics),
                  path_in_repo=base+'/metrics')
api.upload_file(path_or_fileobj=str(folder/'run_manifest.json'),repo_id=repo,
                repo_type='dataset',path_in_repo=base+'/run_manifest.json')
print('Private Task 3 evaluation archive',base)
