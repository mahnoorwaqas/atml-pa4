"""Held-out Task 3 target-only vs exact cached speculative decoding.
Fixed 120 prompts, gamma 2/4, two paired repetitions, 64-token EOS-aware cap.
"""
from __future__ import annotations
import argparse,gzip,json,math,pathlib,random,re,time
import numpy as np
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from task3.fast_sampler import sample_cached,target_cached
TARGET="Qwen/Qwen2.5-1.5B-Instruct"
DRAFT="Qwen/Qwen2.5-0.5B-Instruct"

def read(path):
    f=gzip.open if str(path).endswith('.gz') else open
    with f(path,'rt',encoding='utf8') as handle:
        return [json.loads(line) for line in handle if line.strip()]

def dump(file,x):
    file.parent.mkdir(parents=True,exist_ok=True)
    file.write_text(json.dumps(x,indent=2,default=str)+'\n')

def prompt_data(staff):
    math=read(staff/'data/public_locked/gsm8k_test_locked.jsonl.gz')
    short=read(staff/'data/short_form/short_test.jsonl')
    # Familiar/shifted templates are labeled by test split, not family ID.
    if len(math)!=60 or len(short)!=60:raise RuntimeError('Expected exactly 120 held-out prompts')
    out=[]
    for r in math:
        out.append(dict(id=f'GSM8K-{r["order"]:04d}',domain='math',
             prompt=r['example']['question'],answer=r['example'].get('answer','')))
    for r in short:
        if r.get('split') not in ('familiar','shifted'):
            raise RuntimeError(f"Invalid short-form test split: {r.get('split')!r}")
        out.append(dict(id=r['id'],
             domain=('familiar_extraction' if r['split']=='familiar'
                     else 'heldout_extraction'),
             prompt=r['input'],answer=r['answer'],
             family=r['family'],split=r['split']))
    if sum(x['domain']=='familiar_extraction' for x in out)!=30:
        raise RuntimeError('Expected 30 familiar short-format prompts')
    if sum(x['domain']=='heldout_extraction' for x in out)!=30:
        raise RuntimeError('Expected 30 held-out short-format prompts')
    return out

def tokenize(tok,text):
    return tok.apply_chat_template(
        [{'role':'user','content':text}],tokenize=True,
        add_generation_prompt=True,return_tensors='pt').to('cuda')

def load_model(name,revision=None):
    return AutoModelForCausalLM.from_pretrained(
        name,revision=revision,torch_dtype=torch.float16,
        attn_implementation='sdpa').to('cuda').eval()

def answer_value(response):
    cleaned=response.strip()
    marker=re.search(r'####\s*([^\n]+)',cleaned)
    if marker:cleaned=marker.group(1)
    choices=re.findall(r'[-+]?\d[\d,]*(?:\.\d+)?',cleaned)
    return choices[-1].replace(',','') if choices else None

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--staff',type=pathlib.Path,required=True)
    p.add_argument('--out',type=pathlib.Path,required=True)
    p.add_argument('--draft',default=DRAFT)
    p.add_argument('--condition',required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--smoke',action='store_true',help='Execute a small GPU test, not final evaluation')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(6304);random.seed(6304);np.random.seed(6304)
    lock=json.loads((args.staff/'data/local_source_revisions.json').read_text())['revisions']
    tok=AutoTokenizer.from_pretrained(TARGET,revision=lock['model_target'])
    draft_tok=AutoTokenizer.from_pretrained(DRAFT,revision=lock['model_draft'])
    examples=prompt_data(args.staff)
    if args.smoke:
        examples=[examples[0],examples[60],examples[90]]
    nreps=1 if args.smoke else 2
    cap=8 if args.smoke else 64
    for example in examples[:10]:
        if not torch.equal(tokenize(tok,example['prompt']),
                           tokenize(draft_tok,example['prompt'])):
            raise RuntimeError('Target/draft chat-template IDs differ')
    target=load_model(TARGET,lock['model_target'])
    draft=load_model(args.draft,lock['model_draft'] if args.draft==DRAFT else None)
    if target.config.vocab_size!=draft.config.vocab_size:
        raise RuntimeError('Target/draft vocabularies differ')
    prefill=tokenize(tok,examples[0]['prompt'])
    with torch.inference_mode():
        ref=target(prefill,use_cache=False).logits[:,-1,:].float()
        cached=target(prefill,use_cache=True).logits[:,-1,:].float()
        err=float((ref-cached).abs().max())
    if err>.05:raise RuntimeError(f'Split-prefill logits mismatch {err}')
    dump(args.out/'protocol.json',{'target':TARGET,'draft':args.draft,
        'seed':args.seed,'condition':args.condition,
        'source_revisions':lock,'split_prefill_max_logit_error':err,
        'target_temperature':1.,'target_top_p':1.,
        'max_new_tokens_primary':cap,'gamma':[2,4],'repetitions':nreps,
        'smoke':args.smoke,
        'gpu':torch.cuda.get_device_name(0)})
    file=args.out/'execution.jsonl'
    with file.open('w',encoding='utf8') as output:
        for idx,example in enumerate(examples):
            prompt=tokenize(tok,example['prompt'])
            for repetition in range(nreps):
                for gamma in (2,4):
                    base_seed=6304+idx*7919+repetition*1009+gamma*101
                    target_rng=torch.Generator(device='cuda').manual_seed(base_seed)
                    draft_rng=torch.Generator(device='cuda').manual_seed(base_seed)
                    torch.cuda.reset_peak_memory_stats()
                    target_tokens,tref=target_cached(target,prompt,cap,tok.eos_token_id,target_rng)
                    target_peak=torch.cuda.max_memory_allocated()
                    torch.cuda.reset_peak_memory_stats()
                    tokens,stats=sample_cached(target,draft,prompt,gamma,cap,
                                               tok.eos_token_id,draft_rng)
                    record={'id':example['id'],'domain':example['domain'],
                            'seed':args.seed,'condition':args.condition,
                            'repetition':repetition,'gamma':gamma,
                            'target_reference':tref,
                            'target_output_ids':target_tokens,
                            'speculative_output_ids':tokens,
                            'speculative':stats,
                            'target_peak_bytes':target_peak,
                            'speculative_peak_bytes':torch.cuda.max_memory_allocated(),
                            'paired_throughput_ratio':stats['tokens_per_second']/
                                              max(tref['tokens_per_second'],1e-8)}
                    output.write(json.dumps(record)+'\n');output.flush()
            if (idx+1)%10==0:print('EVALUATION',idx+1,'of',len(examples),flush=True)
    qfile=args.out/'standalone_quality.jsonl'
    with qfile.open('w',encoding='utf8') as out:
        for idx,example in enumerate(examples):
            prompt=tokenize(tok,example['prompt'])
            limit=(16 if args.smoke else (512 if example['domain']=='math' else 32))
            rng=torch.Generator(device='cuda').manual_seed(args.seed+200003+idx)
            tokens,metrics=target_cached(draft,prompt,limit,tok.eos_token_id,rng)
            decoded=tok.decode(tokens,skip_special_tokens=True)
            if example['domain']=='math':
                expected=answer_value(example['answer'])
                observed=answer_value(decoded)
                correct=expected is not None and expected==observed
            else:
                expected=str(example['answer']).strip()
                observed=decoded.strip()
                correct=observed==expected
            out.write(json.dumps({'id':example['id'],'domain':example['domain'],
                'expected':expected,'response':decoded,'answer_extracted':observed,
                'exact_match_or_numeric':bool(correct),
                'termination':metrics['terminated'],'generated_tokens':len(tokens),
                'cap':limit})+'\n')
    # 24 fixed math examples, 64 tokens even across EOS.
    diag=args.out/'math_fixed64.jsonl'
    with diag.open('w',encoding='utf8') as out:
        for idx,ex in enumerate(examples[:0] if args.smoke else examples[:24]):
            prompt=tokenize(tok,ex['prompt'])
            for gamma in (2,4):
                base=6304+idx*7919+gamma
                t,_=target_cached(target,prompt,64,-1,
                   torch.Generator(device='cuda').manual_seed(base))
                seq,metrics=sample_cached(target,draft,prompt,gamma,64,-1,
                   torch.Generator(device='cuda').manual_seed(base))
                if len(t)!=64 or len(seq)!=64:raise RuntimeError('Invalid fixed-length timing')
                out.write(json.dumps({'id':ex['id'],'gamma':gamma,
                   'speculative_metrics':metrics})+'\n')
    dump(args.out/'completed.json',{'finished':True,'condition':args.condition,
                'seed':args.seed,'smoke':args.smoke,
                'primary_cases':len(examples)*nreps*2,
                'quality_cases':len(examples),'fixed64_cases':0 if args.smoke else 24*2})
    print('TASK3 HELD-OUT EVALUATION COMPLETE',args.condition,args.seed,flush=True)
if __name__=='__main__':main()
