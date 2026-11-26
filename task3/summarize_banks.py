"""Aggregate exact common-prefix evidence into balanced, paired Task 3 reports."""
from __future__ import annotations
import argparse,collections,json,os,pathlib
import numpy as np
from huggingface_hub import HfApi,hf_hub_download

VALUES=('forward_kl','reverse_kl','tv','expected_acceptance','greedy_agreement')
DRAFTS=('original_6304','off_6304','off_6305','on_6304','on_6305')
BANKS=('teacher','original_draft','original_spec_g4',
       'policy_off_6304','policy_off_6305','policy_on_6304','policy_on_6305')

def stats(rows):
    if not rows:return None
    return {'n':len(rows),**{k:float(np.mean([float(x[k]) for x in rows])) for k in VALUES}}

def ci(rows,seed=6304):
    if len(rows)<2:return [float(np.mean(rows))]*2
    r=np.random.default_rng(seed)
    a=np.array(rows,float)
    idx=r.integers(0,len(a),size=(2000,len(a)))
    return list(map(float,np.quantile(a[idx].mean(axis=1),[.025,.975])))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--bank-run-id',required=True)
    p.add_argument('--out',type=pathlib.Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    api=HfApi(token=os.environ['HF_TOKEN'])
    repo='mahnoorwaqas/atml-pa4-results'
    assert api.repo_info(repo,repo_type='dataset').private
    prefix=f'task3/prefix-banks/github-run-{a.bank_run_id}/'
    def get(name):
        return hf_hub_download(repo_id=repo,repo_type='dataset',
                  filename=prefix+name,token=os.environ['HF_TOKEN'])
    completed=json.load(open(get('completed.json')))
    assert completed['finished'] and not completed['smoke'] and completed['prompt_count']==120
    assert all(completed['bank_counts'][b]==120 for b in BANKS)
    aggregate=collections.defaultdict(list)
    by_prompt=collections.defaultdict(list)
    aligned=collections.defaultdict(dict)
    total=0
    for line in open(get('prefix_metrics.jsonl'),encoding='utf8'):
        row=json.loads(line)
        key=(row['draft_condition'],row['bank'],row['domain'],row['position_bin'])
        aggregate[key].append(row)
        by_prompt[(row['draft_condition'],row['bank'],row['prompt_id'])].append(row)
        aligned[(row['bank'],row['prompt_id'],row['position'])][row['draft_condition']]=row
        total+=1
    assert len(completed['metric_rows'])==5
    assert len(set(completed['metric_rows'].values()))==1
    assert sum(completed['metric_rows'].values())==total
    out={'bank_run_id':a.bank_run_id,'prefix_metric_rows':total,
         'by_domain_position':{},'model_bank_summary':{},
         'teacher_speculative_consistency':{},'representative_disagreements':[]}
    for k,rows in aggregate.items():
        out['by_domain_position']['|'.join(k)]=stats(rows)
    lines=['# Task 3 fixed prefix evidence','',
           'All five drafts are evaluated on identical saved prefix token IDs within each bank; banks themselves are independent sampled distributions.',
           '',
           '| Draft | Bank | N prefixes | Forward KL | Reverse KL | TV | Expected acceptance | Greedy agreement |',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    for model in DRAFTS:
        for bank in BANKS:
            rows=[r for (m,b,_,_),rr in aggregate.items() if m==model and b==bank for r in rr]
            t=stats(rows);assert t is not None
            out['model_bank_summary'][model+'|'+bank]=t
            lines.append(f'| {model} | {bank} | {t["n"]} | {t["forward_kl"]:.4f} | {t["reverse_kl"]:.4f} | {t["tv"]:.4f} | {t["expected_acceptance"]:.4f} | {t["greedy_agreement"]:.3f} |')
    lines+=['','## Teacher versus original-speculative prefix-bank consistency','',
            'The banks are independent samples from the target-prefix law, not identical token sequences. This compares prompt-matched averages, not individual-prefix pairs.','',
            '| Draft | Prompt-paired TV mean difference (spec - teacher) | 95% bootstrap CI | N prompts |',
            '|---|---:|---|---:|']
    for model in DRAFTS:
        differences=[]
        prompts={p for m,b,p in by_prompt if m==model and b=='teacher'}
        for prompt in prompts:
            teacher=by_prompt.get((model,'teacher',prompt),[])
            spec=by_prompt.get((model,'original_spec_g4',prompt),[])
            if teacher and spec:
                differences.append(float(np.mean([r['tv'] for r in spec])-
                                         np.mean([r['tv'] for r in teacher])))
        c=ci(differences)
        val={'prompt_count':len(differences),'tv_difference_mean':float(np.mean(differences)),
             'prompt_bootstrap95':c}
        out['teacher_speculative_consistency'][model]=val
        lines.append(f'| {model} | {val["tv_difference_mean"]:+.4f} | [{c[0]:+.4f}, {c[1]:+.4f}] | {len(differences)} |')
    differences=[]
    for (bank,prompt,position),item in aligned.items():
        if 'off_6304' not in item or 'on_6304' not in item:continue
        differences.append({
          'bank':bank,'prompt_id':prompt,'position':position,
          'off_6304_tv':item['off_6304']['tv'],
          'on_6304_tv':item['on_6304']['tv'],
          'delta_on_minus_off':item['on_6304']['tv']-item['off_6304']['tv']})
    out['representative_disagreements']=sorted(
        differences,key=lambda x:abs(x['delta_on_minus_off']),reverse=True)[:20]
    lines+=['','## Important limitations','',
        '- The teacher/speculative consistency statistic checks TV-derived expectations across matched prompts, not a full high-dimensional two-sample distribution test.',
        '- Stronger distributional equivalence evidence comes separately from the 100,000-trial exact Markov χ² test.',
        '- Shared-prefix comparisons do not guarantee faster GPU execution.',
        '- Common-prefix TV is not interchangeable with empirical accepted/examined fractions measured on speculative trajectories.',
        '']
    (a.out/'prefix_summary.json').write_text(json.dumps(out,indent=2)+'\n')
    (a.out/'prefix_summary.md').write_text('\n'.join(lines))
    print('Prefix bank report complete',total,'records',flush=True)
if __name__=='__main__':main()
