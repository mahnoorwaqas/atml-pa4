"""Aggregate all Task 1 runs WITHOUT choosing hyperparameters on held-out data.

Examples:
  python task1/summarize.py --root ./all_results --out ./task1_summary
Results are read from recursive task1_<condition> directories.
"""
import argparse,csv,json,math,pathlib
import numpy as np

PAIRWISE=[("adaptive50","wanda50"),("adaptive75","wanda75"),
          ("mixed4","awq4"),("mixed3","awq3")]
def load(path):return json.loads(path.read_text())
def bootstrap_ppl(a,b,rep=2000,seed=6304):
    if len(a)!=len(b) or not a:raise ValueError('paired windows count mismatch')
    for i,(x,y) in enumerate(zip(a,b)):
        if x["id"]!=y["id"] or x["tokens"]!=y["tokens"]:
            raise ValueError(f'window {i} has mismatched contexts')
    n=np.array([x["tokens"] for x in a],dtype=np.int64)
    x=np.array([x["nll_sum"] for x in a])
    y=np.array([r["nll_sum"] for r in b])
    rng=np.random.default_rng(seed)
    diffs=[]
    for _ in range(rep):
        ix=rng.integers(0,len(a),len(a))
        den=n[ix].sum()
        diffs.append(math.exp(x[ix].sum()/den)-math.exp(y[ix].sum()/den))
    direct=math.exp(x.sum()/n.sum())-math.exp(y.sum()/n.sum())
    return dict(delta_ppl=direct,ci95=np.quantile(diffs,[.025,.975]).tolist(),
                repetitions=rep,seed=seed,windows=len(a),tokens=int(n.sum()))
def bootstrap_hs(a,b,metric,rep=2000,seed=6304):
    index_a={int(x["order"]):x for x in a};index_b={int(x["order"]):x for x in b}
    if set(index_a)!=set(index_b):raise ValueError('HellaSwag IDs not matched')
    keys=sorted(index_a)
    if not keys:raise ValueError('No paired cases')
    dif=np.array([index_a[k][metric]-index_b[k][metric] for k in keys],dtype=float)
    rng=np.random.default_rng(seed)
    ix=rng.integers(0,len(keys),(rep,len(keys)))
    means=dif[ix].mean(axis=1)
    return dict(delta_accuracy=float(dif.mean()),ci95=np.quantile(means,[.025,.975]).tolist(),
                examples=len(keys),repetitions=rep,seed=seed)
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=pathlib.Path,required=True)
    p.add_argument("--out",type=pathlib.Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    groups={}
    for p in sorted(args.root.rglob('completed.json')):
        if load(p).get('finished') is True and p.parent.name.startswith('task1_'):
            key=p.parent.name.removeprefix('task1_')
            groups.setdefault(key,[]).append(p.parent)
    # Never silently choose between multiple runs of the same method.
    ambiguous={k:[str(x) for x in v] for k,v in groups.items() if len(v)!=1}
    if ambiguous:raise RuntimeError(f'Ambiguous duplicate run names: {ambiguous}')
    runs={k:v[0] for k,v in groups.items()}
    if 'baseline' not in runs:raise RuntimeError('Missing FP16 baseline')
    baseline_lock=load(runs['baseline']/'source_revisions.json')
    baseline_ids=load(runs['baseline']/'data_selection.json')
    rows=[];paired={}
    for name,dir in sorted(runs.items()):
        # Gzip metadata includes creation-time bytes: binary .gz hashes need
        # not match even if the *decompressed* locked windows are identical.
        if load(dir/'source_revisions.json')!=baseline_lock:
            raise RuntimeError('Upstream source-revision mismatch: '+name)
        if load(dir/'data_selection.json')!=baseline_ids:
            raise RuntimeError('Staff calibration/diagnostic row IDs mismatch: '+name)
        wiki=load(dir/'wiki_summary.json');c4=load(dir/'c4_summary.json')
        hs=load(dir/'hellaswag_summary.json');comp=load(dir/'compression.json')
        costs=load(dir/'costs.json')
        timings=load(dir/'timing.json')
        rows.append(dict(condition=name,
                 calibration_seconds=costs['calibration_seconds'],
                 compression_seconds=costs['compression_seconds'],
                 prefill512_ms_median=float(np.median(timings['512']['prefill_ms'])),
                 prefill2048_ms_median=float(np.median(timings['2048']['prefill_ms'])),
                 decode64_ms_median=float(np.median(timings['decode_64_tokens']['latency_ms'])),
                 prefill512_peak_allocated_bytes=timings['512']['peak_allocated_bytes'],
                 wiki_nll=wiki['mean_nll'],wiki_ppl=wiki['ppl'],
                 wiki_delta_ppl=wiki['ppl']-load(runs['baseline']/'wiki_summary.json')['ppl'],
                 c4_nll=c4['mean_nll'],c4_ppl=c4['ppl'],
                 c4_delta_ppl=c4['ppl']-load(runs['baseline']/'c4_summary.json')['ppl'],
                 hs_n=hs['n'],hs_acc=hs['accuracy'],hs_acc_norm=hs['acc_norm'],
                 eligible_sparsity=comp.get('eligible_sparsity'),
                 total_packed_bytes=comp.get('packed_total_bytes'),
                 dense_fp16_bytes=comp.get('dense_fp16_storage_bytes')))
    for left,right in PAIRWISE:
        if left not in runs or right not in runs:continue
        x,y=runs[left],runs[right]
        paired[left+'__minus__'+right]={
            'wikitext':bootstrap_ppl(load(x/'wiki_windows.json'),load(y/'wiki_windows.json')),
            'c4':bootstrap_ppl(load(x/'c4_windows.json'),load(y/'c4_windows.json')),
            'hellaswag_raw':bootstrap_hs(load(x/'hellaswag_examples.json'),
                                          load(y/'hellaswag_examples.json'),'acc'),
            'hellaswag_char_norm':bootstrap_hs(load(x/'hellaswag_examples.json'),
                                                load(y/'hellaswag_examples.json'),'acc_norm')}
    with (args.out/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (args.out/'paired_bootstrap.json').write_text(json.dumps(paired,indent=2)+'\n')
    (args.out/'all_results.json').write_text(json.dumps(rows,indent=2)+'\n')
    # Sensitivity maps are retained as machine-readable per-projection data.
    for name,path in runs.items():
        data=load(path/'compression.json')
        if 'per_projection' in data:
            (args.out/f'{name}_projection_map.json').write_text(
                json.dumps(data['per_projection'],indent=2)+'\n')
    print(json.dumps({'conditions':list(runs),'paired_comparisons':list(paired),'output':str(args.out)}))
if __name__=='__main__':main()
