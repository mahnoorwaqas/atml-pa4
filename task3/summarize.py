"""Task 3 paired evaluation synthesis from private Hugging Face archives.
No missing condition is silently dropped or inferred.
"""
from __future__ import annotations
import argparse,collections,json,math,os,pathlib,statistics
import numpy as np
from huggingface_hub import hf_hub_download,HfApi

CONDITIONS=[('original',6304),('off',6304),('off',6305),('on',6304),('on',6305)]
TRAINING={'off_6304':'45773884864','off_6305':'45773884864',
          'on_6304':'45785982970','on_6305':'45773884864'}

def load_jsonl(filename):
    with open(filename,encoding='utf8') as f:
        return [json.loads(row) for row in f if row.strip()]

def bootstrap_ci(values,reps=2000,seed=6304):
    vals=np.array(values,dtype=float)
    if not len(vals):raise ValueError('empty estimates')
    rng=np.random.default_rng(seed)
    ix=rng.integers(0,len(vals),size=(reps,len(vals)))
    boot=vals[ix].mean(axis=1)
    return [float(np.quantile(boot,.025)),float(np.quantile(boot,.975))]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--evaluation-run-id',required=True)
    p.add_argument('--out',type=pathlib.Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    token=os.environ['HF_TOKEN']
    api=HfApi(token=token)
    results='mahnoorwaqas/atml-pa4-results'
    assert api.repo_info(results,repo_type='dataset').private
    all_data={}
    all_training={}
    for condition,seed in CONDITIONS:
        key=f'{condition}_{seed}'
        prefix=f'task3/evaluations/{key}/github-run-{a.evaluation_run_id}/metrics/'
        def fetch(name):
            return hf_hub_download(repo_id=results,repo_type='dataset',
                           filename=prefix+name,token=token)
        completed=json.load(open(fetch('completed.json')))
        if not completed.get('finished') or completed.get('primary_cases')!=480 or completed.get('quality_cases')!=120:
            raise RuntimeError('Incomplete eval '+key)
        records=load_jsonl(fetch('execution.jsonl'))
        quality=load_jsonl(fetch('standalone_quality.jsonl'))
        fixed=load_jsonl(fetch('math_fixed64.jsonl'))
        if len(records)!=480 or len(quality)!=120 or len(fixed)!=48:
            raise RuntimeError('Wrong record count '+key)
        by_pair={(r['id'],r['gamma'],r['repetition']):r for r in records}
        if len(by_pair)!=480:raise RuntimeError('Duplicate eval keys '+key)
        all_data[key]={'execution':records,'quality':quality,'fixed':fixed}
        if condition!='original':
            train_prefix=f'task3/{key}/github-run-{TRAINING[key]}/'
            stat=json.load(open(hf_hub_download(repo_id=results,repo_type='dataset',
                    filename=train_prefix+'training_status.json',token=token)))
            assert stat['scored']=={'math':30000,'short':10000}
            all_training[key]=stat
    summary={'evaluation_run_id':a.evaluation_run_id,'conditions':{},'training':{},
             'definitions':{
                'bootstrap':'2000 prompt-level paired bootstrap resamples, fixed seed 6304',
                'target_reference':'Adjacent same-prompt target-only run; sampling output need not be identical',
                'speedup':'target-only output tokens per decode second / speculative output tokens per decode second',
                'source_note':'All estimates computed from recorded private Kaggle outputs; no invented measurements'}}
    lines=['# Task 3: exact speculative sampling evaluation','',
           'All five conditions contain 480 primary EOS-aware observations, 120 separate answer-quality observations, and 48 fixed-64 mathematics cases.','',
           '| Condition | Gamma | Mean per-prompt throughput ratio | 95% paired CI | Acceptance (examined) | Target passes/token |',
           '|---|---:|---:|---|---:|---:|']
    for condition,seed in CONDITIONS:
        key=f'{condition}_{seed}'
        data=all_data[key]
        grouped=collections.defaultdict(list)
        for rec in data['execution']:
            grouped[rec['gamma']].append(rec)
        item={'gamma':{},'quality':{}}
        for gamma,recs in sorted(grouped.items()):
            per_prompt=collections.defaultdict(list)
            domains=collections.defaultdict(list)
            for row in recs:
                per_prompt[row['id']].append(row)
                domains[row['domain']].append(row)
            if len(per_prompt)!=120 or any(len(x)!=2 for x in per_prompt.values()):
                raise RuntimeError('Expected two repeats of each prompt '+key)
            def mean_ratio(x):return float(np.mean([r['paired_throughput_ratio'] for r in x]))
            ratios=[mean_ratio(x) for x in per_prompt.values()]
            proposed=sum(x['speculative']['proposed'] for x in recs)
            examined=sum(x['speculative']['examined'] for x in recs)
            accepted=sum(x['speculative']['accepted'] for x in recs)
            out_tokens=sum(len(x['speculative_output_ids']) for x in recs)
            targetpasses=sum(x['speculative']['target_passes'] for x in recs)
            target_baseline=sum(x['target_reference']['decode_seconds'] for x in recs)
            speculative_total=sum(x['speculative']['decode_seconds'] for x in recs)
            metrics={
               'paired_ratio_mean':float(np.mean(ratios)),
               'paired_ratio_prompt_bootstrap95':bootstrap_ci(ratios,seed=6304+gamma),
               'accepted_over_examined':accepted/max(examined,1),
               'accepted_over_proposed':accepted/max(proposed,1),
               'target_verifications_per_output_token':targetpasses/max(out_tokens,1),
               'total_speculative_output_tokens':out_tokens,
               'target_baseline_decode_seconds_sum':target_baseline,
               'speculative_decode_seconds_sum':speculative_total,
               'domains':{d:{
                  'n':len(x),'mean_paired_ratio':mean_ratio(x),
                  'accepted_over_examined':sum(z['speculative']['accepted'] for z in x)/
                      max(sum(z['speculative']['examined'] for z in x),1)
               }for d,x in domains.items()}}
            item['gamma'][str(gamma)]=metrics
            lo,hi=metrics['paired_ratio_prompt_bootstrap95']
            lines.append(f'| {key} | {gamma} | {metrics["paired_ratio_mean"]:.3f}× | [{lo:.3f}, {hi:.3f}] | {metrics["accepted_over_examined"]:.3f} | {metrics["target_verifications_per_output_token"]:.3f} |')
        for d in sorted({r['domain'] for r in data['quality']}):
            vals=[r for r in data['quality'] if r['domain']==d]
            item['quality'][d]={'n':len(vals),'accuracy':sum(bool(r['exact_match_or_numeric']) for r in vals)/len(vals),
                 'truncated_fraction':sum(r['termination']=='cap' for r in vals)/len(vals)}
        summary['conditions'][key]=item
        if key in all_training:
            train=all_training[key]
            summary['training'][key]={
                'teacher_scored_tokens':train['scored'],
                'updates':train['updates'],
                'skipped_updates':train['skipped_updates'],
                'total_wall_seconds':train['train_wall_seconds'],
                'teacher_rollout_seconds':train['teacher_rollout_seconds'],
                'student_rollout_seconds':train['student_rollout_seconds'],
                'scoring_backward_seconds':train['teacher_scoring_training_seconds'],
                'adapter_merge_seconds':train['merge_seconds'],
                'merged_logit_max_abs_error':train['merged_logit_max_abs_error']}
    lines+=['','## Standalone student accuracy (separate answer-quality cap)','',
            '| Condition | Math accuracy | Familiar accuracy | Shifted accuracy |',
            '|---|---:|---:|---:|']
    for key,item in summary['conditions'].items():
        q=item['quality']
        lines.append(f'| {key} | {q["math"]["accuracy"]:.3f} | {q["familiar_extraction"]["accuracy"]:.3f} | {q["heldout_extraction"]["accuracy"]:.3f} |')
    lines+=['','## Interpretation guardrails','',
      '- The experiment uses finite-sample throughput estimates; performance superiority is not guaranteed.',
      '- The above evaluation does not yet constitute the required fixed-bank forward/reverse-KL and TV evidence.',
      '- Packed prompt counts, generation EOS and checkpoint path are verified before reporting.',
      '- Training recovery and evaluator errors are documented separately.','']
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (a.out/'summary.md').write_text('\n'.join(lines))
    print('Summarized all 5 Task 3 conditions with 2000 paired bootstrap reps')
if __name__=='__main__':main()
