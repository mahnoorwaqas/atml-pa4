"""KV-cached exact speculative sampler, using explicit cache crop/rollback.
Reference implementation task3/sampler.py is the distributional control.
"""
from __future__ import annotations
import time,torch
from task3.sampler import exact_step,sample
def sync():torch.cuda.synchronize()
def forward(model,ids,cache=None):
    with torch.inference_mode():
        result=model(input_ids=ids,past_key_values=cache,use_cache=True)
    return result.logits[0,-1].float(),result.past_key_values
def sample_cached(target,draft,prompt_ids,gamma,max_new_tokens,eos_id,generator):
    if prompt_ids.ndim!=2 or prompt_ids.shape[0]!=1:raise ValueError('batch=1 required')
    if not 1<=gamma<=16:raise ValueError('gamma')
    if max_new_tokens<1:raise ValueError('token cap')
    sync();t0=time.perf_counter()
    target_next,target_cache=forward(target,prompt_ids)
    draft_next,draft_cache=forward(draft,prompt_ids)
    sync();prefill_seconds=time.perf_counter()-t0
    tokens=[]
    stats=dict(proposed=0,examined=0,accepted=0,target_passes=0,
        draft_forwards=0,rounds=0,bonus_rounds=0,
        accepted_prefix=[],proposal_seconds=0.,verification_seconds=0.,
        sampling_seconds=0.,rollback_seconds=0.,prefill_seconds=prefill_seconds,
        terminated='cap')
    while len(tokens)<max_new_tokens:
        start_len=prompt_ids.shape[1]+len(tokens)
        proposals=[];qrows=[]
        k=min(gamma,max_new_tokens-len(tokens))
        sync();phase=time.perf_counter()
        for i in range(k):
            q=torch.softmax(draft_next.float(),dim=-1)
            nxt=sample(q,generator);qrows.append(q);proposals.append(nxt)
            if nxt==eos_id:break
            if i<k-1:
                draft_next,draft_cache=forward(draft,
                    torch.tensor([[nxt]],device=prompt_ids.device),draft_cache)
                stats['draft_forwards']+=1
        sync();stats['proposal_seconds']+=time.perf_counter()-phase
        sync();phase=time.perf_counter()
        with torch.inference_mode():
            verified=target(torch.tensor([proposals],device=prompt_ids.device),
                past_key_values=target_cache,use_cache=True)
        target_cache=verified.past_key_values
        trows=[torch.softmax(target_next.float(),dim=-1)]
        for j in range(len(proposals)):
            trows.append(torch.softmax(verified.logits[0,j].float(),dim=-1))
        stats['target_passes']+=1
        sync();stats['verification_seconds']+=time.perf_counter()-phase
        sync();phase=time.perf_counter()
        def uniform():return float(torch.rand((),generator=generator,device=prompt_ids.device))
        def draw(dist):return sample(dist,generator)
        chosen,accepted,examined,all_accepted=exact_step(trows,qrows,proposals,uniform,draw)
        stats['accepted']+=accepted
        stats['examined']+=examined;stats['proposed']+=len(proposals)
        stats['rounds']+=1;stats['accepted_prefix'].append(accepted)
        stats['bonus_rounds']+=int(all_accepted)
        sync();stats['sampling_seconds']+=time.perf_counter()-phase
        was_eos=False
        for token in chosen:
            if len(tokens)>=max_new_tokens:break
            tokens.append(token)
            if token==eos_id:
                was_eos=True
                stats['terminated']='eos'
                break
        if was_eos or len(tokens)==max_new_tokens:break
        sync();phase=time.perf_counter()
        if all_accepted:
            # Target has already verified all k draft tokens.
            bonus=chosen[-1]
            target_next,target_cache=forward(target,
                torch.tensor([[bonus]],device=prompt_ids.device),target_cache)
            # Draft cache holds k-1 proposals (last proposed token not fed).
            draft_next,draft_cache=forward(draft,
                torch.tensor([[proposals[-1],bonus]],device=prompt_ids.device),draft_cache)
            stats['draft_forwards']+=1
        else:
            # Roll back to the exactly accepted prefix, then append residual sample.
            residual=chosen[-1]
            expected=start_len+accepted
            if not hasattr(target_cache,'crop') or not hasattr(draft_cache,'crop'):
                raise RuntimeError('Installed Transformers cache has no crop()')
            target_cache.crop(expected)
            draft_cache.crop(expected)
            if target_cache.get_seq_length()!=expected or draft_cache.get_seq_length()!=expected:
                raise RuntimeError('Cache position inconsistent after rejection')
            target_next,target_cache=forward(target,
                torch.tensor([[residual]],device=prompt_ids.device),target_cache)
            draft_next,draft_cache=forward(draft,
                torch.tensor([[residual]],device=prompt_ids.device),draft_cache)
            stats['draft_forwards']+=1
        if target_cache.get_seq_length()!=prompt_ids.shape[1]+len(tokens):
            raise RuntimeError('Target cache length inconsistent')
        if draft_cache.get_seq_length()!=prompt_ids.shape[1]+len(tokens):
            raise RuntimeError('Draft cache length inconsistent')
        sync();stats['rollback_seconds']+=time.perf_counter()-phase
    stats['generated_tokens']=len(tokens)
    stats['accepted_over_examined']=stats['accepted']/max(stats['examined'],1)
    stats['accepted_over_proposed']=stats['accepted']/max(stats['proposed'],1)
    stats['target_passes_per_token']=stats['target_passes']/max(len(tokens),1)
    stats['decode_seconds']=(stats['proposal_seconds']+stats['verification_seconds']+
                             stats['sampling_seconds']+stats['rollback_seconds'])
    stats['tokens_per_second']=len(tokens)/max(stats['decode_seconds'],1e-9)
    return tokens,stats

def target_cached(model,prompt_ids,max_new_tokens,eos_id,generator):
    sync();t0=time.perf_counter()
    nxt,cache=forward(model,prompt_ids)
    sync();prefill=time.perf_counter()-t0
    tokens=[]
    sync();t0=time.perf_counter()
    while len(tokens)<max_new_tokens:
        tok=sample(torch.softmax(nxt,dim=-1),generator)
        tokens.append(tok)
        if tok==eos_id:break
        nxt,cache=forward(model,torch.tensor([[tok]],device=prompt_ids.device),cache)
    sync();elapsed=time.perf_counter()-t0
    return tokens,dict(prefill_seconds=prefill,decode_seconds=elapsed,
        tokens_per_second=len(tokens)/max(elapsed,1e-9),
        generated_tokens=len(tokens),terminated=('eos' if tokens[-1]==eos_id else 'cap'))
