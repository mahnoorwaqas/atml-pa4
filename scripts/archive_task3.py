"""Archive completed Task 3 training runs to two verified private HF repositories."""
import argparse,json,os,pathlib
from huggingface_hub import HfApi
p=argparse.ArgumentParser()
p.add_argument('--download-dir',type=pathlib.Path,required=True)
p.add_argument('--method',choices=('off','on'),required=True)
p.add_argument('--seed',type=int,choices=(6304,6305),required=True)
p.add_argument('--github-run-id',required=True)
a=p.parse_args()
root=a.download_dir/'task3_results'
train=root/'training'
marker=train/'completed.json'
if not marker.is_file():raise RuntimeError('Task 3 completion marker missing')
completed=json.loads(marker.read_text())
if not completed.get('finished') or completed.get('method')!=a.method or completed.get('seed')!=a.seed:
    raise RuntimeError('Incorrect/failed Task 3 checkpoint')
status=json.loads((train/'training_status.json').read_text())
assert status['scored']=={'math':30000,'short':10000},'Unmatched teacher token quotas'
for required in [train/'adapter'/'adapter_config.json',train/'merged'/'config.json',
                 train/'rollouts.jsonl',root/'source_revisions.json',
                 root/'staff_checksum_audit.json',root/'run_manifest.json']:
    if not required.is_file():raise RuntimeError(f'Missing Task 3 artifact {required}')
api=HfApi(token=os.environ['HF_TOKEN'])
model_repo='mahnoorwaqas/atml-pa4-checkpoints'
result_repo='mahnoorwaqas/atml-pa4-results'
for repo,kind in [(model_repo,'model'),(result_repo,'dataset')]:
    if not api.repo_info(repo,repo_type=kind).private:
        raise RuntimeError('Refusing upload to a public repository: '+repo)
base=f'task3/{a.method}_{a.seed}/github-run-{a.github_run_id}'
api.upload_folder(repo_id=model_repo,repo_type='model',
                  folder_path=str(train/'adapter'),path_in_repo=base+'/adapter')
api.upload_folder(repo_id=model_repo,repo_type='model',
                  folder_path=str(train/'merged'),path_in_repo=base+'/merged')
for f in ['completed.json','training_status.json','rollouts.jsonl',
          'frozen_teacher_pool_manifest.json']:
    path=train/f
    if path.exists():
        api.upload_file(path_or_fileobj=str(path),repo_id=result_repo,
                        repo_type='dataset',path_in_repo=base+'/'+f)
for f in ['run_manifest.json','source_revisions.json','staff_checksum_audit.json']:
    api.upload_file(path_or_fileobj=str(root/f),repo_id=result_repo,
                    repo_type='dataset',path_in_repo=base+'/'+f)
print('Private Task 3 archive complete:',a.method,a.seed,base,flush=True)
