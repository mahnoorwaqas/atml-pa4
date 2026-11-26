"""Frozen shared-prefix banks and student-own rollout bank comparisons.
GPU-only: evaluates full-vocabulary forward/reverse KL, TV, agreement, expected
one-token acceptance on identical bank prefix IDs for every draft.
"""
from __future__ import annotations
import argparse,collections,gzip,json,os,pathlib,random,time
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from task3.fast_sampler import target_cached,sample_cached
from task3.evaluate import TARGET,DRAFT,prompt_data,tokenize

DRAFT_KEYS=('original_6304','off_6304','off_6305','on_6304','on_6305')
BANKS=('teacher','original_draft','original_spec_g4',
       'policy_off_6304','policy_off_6305','policy_on_6304','policy_on_6305')

def make_model(name,revision=None):
    return AutoModelForCausalLM.from_pretrained(name,revision=revision,
        dtype=torch.float16,attn_implementation='sdpa').to('cuda').eval()

def write_lines(path,items):
    with path.open('w',encoding='utf8') as h:
        for obj in items:
            h.write(json.dumps(obj,ensure_ascii=False)+'\n')
            h.flush()

def choose_positions(length,seed):
    rng=random.Random(seed)
    selected=[]
    for low,high,quota in [(1,16,8),(17,32,8),(33,64,16),(65,192,16)]:
        choices=list(range(low,min(high,length)+1))
        selected+=sorted(rng.sample(choices,min(quota,len(choices))))
    return sorted(set(selected))

def metrics(tlogits,slogits):
    t=torch.log_softmax(tlogits.float(),dim=-1)
    s=torch.log_softmax(slogits.float(),dim=-1)
    p=t.exp();q=s.exp()
    fw=float((p*(t-s)).sum().item())
    rv=float((q*(s-t)).sum().item())
    tv=float(((p-q).abs().sum()*0.5).item())
    greedy=bool(torch.argmax(t)==torch.argmax(s))
    return {'forward_kl':max(0.,fw),'reverse_kl':max(0.,rv),
            'tv':tv,'expected_acceptance':1.-tv,'greedy_agreement':greedy}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--staff',type=pathlib.Path,required=True)
    p.add_argument('--out',type=pathlib.Path,required=True)
    p.add_argument('--off-6304',required=True)
    p.add_argument('--off-6305',required=True)
    p.add_argument('--on-6304',required=True)
    p.add_argument('--on-6305',required=True)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    revisions=json.loads((a.staff/'data/local_source_revisions.json').read_text())['revisions']
    tokenizer=AutoTokenizer.from_pretrained(TARGET,revision=revisions['model_target'])
    draft_tokenizer=AutoTokenizer.from_pretrained(DRAFT,revision=revisions['model_draft'])
    tests=prompt_data(a.staff)
    for x in tests[:10]:
        if not torch.equal(tokenize(tokenizer,x['prompt']),
                           tokenize(draft_tokenizer,x['prompt'])):
            raise RuntimeError('Target and draft chat templates/token IDs disagree')
    if a.smoke:tests=[tests[i] for i in (0,60,90)]
    files={'off_6304':a.off_6304,'off_6305':a.off_6305,
           'on_6304':a.on_6304,'on_6305':a.on_6305}
    models={'teacher':make_model(TARGET,revisions['model_target']),
            'original_6304':make_model(DRAFT,revisions['model_draft'])}
    for key,path in files.items():
        if not (pathlib.Path(path)/'config.json').is_file():
            raise RuntimeError('Missing independently merged model '+key+' '+path)
        models[key]=make_model(path)
    for key,model in models.items():
        if model.config.vocab_size!=models['teacher'].config.vocab_size:
            raise RuntimeError('Vocabulary size mismatch: '+key)
        print('Loaded frozen model',key,flush=True)
    teacher=models['teacher'];original=models['original_6304']
    bankfile=a.out/'frozen_banks.jsonl'
    metricfile=a.out/'prefix_metrics.jsonl'
    start=time.time();banks_count=collections.Counter();metrics_count=collections.Counter()
    with bankfile.open('w',encoding='utf8') as bf,metricfile.open('w',encoding='utf8') as mf:
        for idx,case in enumerate(tests):
            prompt=tokenize(tokenizer,case['prompt'])
            cap=8 if a.smoke else (192 if case['domain']=='math' else 32)
            for b,bank in enumerate(BANKS):
                if bank=='teacher':model=teacher
                elif bank=='original_draft':model=original
                elif bank.startswith('policy_'):model=models[bank.removeprefix('policy_')]
                else:model=None
                seed=6304+10007*idx+1000003*b
                rng=torch.Generator(device='cuda').manual_seed(seed)
                if bank=='original_spec_g4':
                    response,_=sample_cached(teacher,original,prompt,4,cap,
                                       tokenizer.eos_token_id,rng)
                else:
                    response,_=target_cached(model,prompt,cap,
                                       tokenizer.eos_token_id,rng)
                positions=choose_positions(len(response),seed+17)
                if a.smoke:positions=positions[:min(4,len(positions))]
                if not positions:raise RuntimeError('Empty saved prefix bank')
                frozen={'prompt_id':case['id'],'domain':case['domain'],'bank':bank,
                   'producing_checkpoint':(bank.removeprefix('policy_')
                      if bank.startswith('policy_') else bank),
                   'seed':seed,'response_token_ids':response,
                   'positions':positions,'source_length':len(response),
                   'termination_reason':'eos' if response[-1]==tokenizer.eos_token_id else 'cap'}
                bf.write(json.dumps(frozen)+'\n');bf.flush()
                banks_count[bank]+=1
                # One full prefilling pass per model per bank/response. All metrics
                # are paired on EXACT SAME frozen prefix token IDs.
                source=torch.tensor([response],device='cuda',dtype=torch.long)
                full=torch.cat((prompt,source),dim=1)
                indices=torch.tensor([prompt.shape[1]+pos-2 for pos in positions],
                                     device='cuda')
                with torch.inference_mode():
                    logits=teacher(input_ids=full,use_cache=False).logits[0]
                    tlog=logits.index_select(0,indices).float()
                del logits
                for key in DRAFT_KEYS:
                    with torch.inference_mode():
                        logits=models[key](input_ids=full,use_cache=False).logits[0]
                        slog=logits.index_select(0,indices).float()
                    del logits
                    for j,pos in enumerate(positions):
                        x=metrics(tlog[j],slog[j])
                        x.update(prompt_id=case['id'],domain=case['domain'],
                            bank=bank,producing_checkpoint=frozen['producing_checkpoint'],
                            position=pos,position_bin=('1-16' if pos<=16
                              else '17-32' if pos<=32 else '33-64' if pos<=64 else '65-192'),
                            draft_condition=key,seed=seed)
                        mf.write(json.dumps(x)+'\n')
                        metrics_count[key]+=1
                    mf.flush();del slog
                del tlog
            if (idx+1)%10==0:
                print('PREFIX BANK PROGRESS',idx+1,'/',len(tests),
                      'metrics',dict(metrics_count),flush=True)
    expected=len(tests)*len(BANKS)
    if sum(banks_count.values())!=expected:raise RuntimeError('Incorrect bank count')
    if any(metrics_count[k]!=metrics_count['original_6304'] for k in DRAFT_KEYS):
        raise RuntimeError('Missing draft comparisons')
    complete={'finished':True,'smoke':a.smoke,'prompt_count':len(tests),
        'bank_counts':dict(banks_count),'metric_rows':dict(metrics_count),
        'wall_seconds':time.time()-start,'source_revisions':revisions,
        'models':files,'metrics':('forward_kl','reverse_kl','tv',
                                 'expected_acceptance','greedy_agreement')}
    (a.out/'completed.json').write_text(json.dumps(complete,indent=2)+'\n')
    print('PREFIX BANK COMPARISON COMPLETE',json.dumps({
        'bank_count':expected,'metric_rows':metrics_count,
        'seconds':complete['wall_seconds']}),flush=True)

if __name__=='__main__':main()
