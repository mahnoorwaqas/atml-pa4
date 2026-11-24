"""Download the successful baseline's exact public source-revision lock.

HF_TOKEN is used only in GitHub Actions. No credential is embedded into Kaggle.
"""
import argparse,json,pathlib
from huggingface_hub import hf_hub_download
p=argparse.ArgumentParser()
p.add_argument('--run-id',required=True)
p.add_argument('--attempt',default='1')
p.add_argument('--output',required=True,type=pathlib.Path)
a=p.parse_args()
if not a.run_id.isdigit() or not a.attempt.isdigit():raise ValueError('invalid run ID')
file=f'task1/baseline/github-run-{a.run_id}-attempt-{a.attempt}/source_revisions.json'
download=hf_hub_download(repo_id='mahnoorwaqas/atml-pa4-results',
                         repo_type='dataset',filename=file)
data=json.loads(pathlib.Path(download).read_text())
expected={'c4','wikitext','hellaswag','gsm8k','model_base','model_target','model_draft'}
if set(data['revisions'])!=expected or any(len(v)!=40 for v in data['revisions'].values()):
    raise RuntimeError('Bad baseline source-revision lock')
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(data,indent=2)+'\n')
print('Reused locked baseline source revisions',a.run_id, a.attempt)
