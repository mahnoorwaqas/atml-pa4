"""Exact rejection-based speculative decoding, including positive residual and bonus.
Full-prefix forward passes are a correctness reference, not an optimized KV-cache kernel.
"""
from __future__ import annotations
from dataclasses import dataclass,field
import torch

def sample(probs,generator=None):
    return int(torch.multinomial(probs.float(),1,generator=generator).item())

def positive_residual(p,q):
    residual=(p.float()-q.float()).clamp_min(0)
    total=residual.sum()
    if total<=0: raise RuntimeError("Residual has zero mass after valid rejection")
    return residual/total

def exact_step(target_distributions,draft_distributions,proposed,random_uniform,draw):
    """Pure exact accept/reject step. Caller includes target bonus distribution last.
    Each target row corresponds to the exact prefix *prior to* one draft token.
    """
    n=len(proposed)
    if len(target_distributions)!=n+1 or len(draft_distributions)!=n:
        raise ValueError("Need n draft and n+1 target conditional distributions")
    for j,tok in enumerate(proposed):
        p=target_distributions[j].float()
        q=draft_distributions[j].float()
        prob=min(1.,float(p[tok]/q[tok])) if float(q[tok])>0 else 1.
        if random_uniform()<prob:
            continue
        replacement=draw(positive_residual(p,q))
        return list(proposed[:j])+[replacement],j,j+1,False
    bonus=draw(target_distributions[n])
    return list(proposed)+[bonus],n,n,True

def next_token_distribution(model,ids):
    with torch.inference_mode():
        logits=model(input_ids=ids,use_cache=False).logits[0,-1].float()
        return torch.softmax(logits,dim=-1)

@dataclass
class Counters:
    proposed:int=0
    examined:int=0
    accepted:int=0
    target_verifications:int=0
    draft_forwards:int=0
    rounds:int=0
    bonus_rounds:int=0
    rejected_rounds:int=0
    accepted_prefix_lengths:list=field(default_factory=list)

def speculative(model_target,model_draft,prompt_ids,gamma,max_new_tokens,eos_id,
                generator=None):
    """Exact sampler using one target verification forward per round.
    Runs without cache to provide an auditable correctness reference. Model
    inputs contain one batch and identical tokenizer vocabulary.
    """
    if gamma<1 or max_new_tokens<1:raise ValueError("Invalid generation settings")
    device=prompt_ids.device
    output=[];counts=Counters()
    while len(output)<max_new_tokens:
        remaining=max_new_tokens-len(output)
        k=min(gamma,remaining)
        proposal=[];qrows=[]
        for _ in range(k):
            ids=torch.cat((prompt_ids,torch.tensor([output+proposal],device=device)),dim=1) if output or proposal else prompt_ids
            q=next_token_distribution(model_draft,ids)
            counts.draft_forwards+=1
            tok=sample(q,generator)
            qrows.append(q);proposal.append(tok)
            if tok==eos_id:break
        prefix=torch.tensor([output+proposal],device=device)
        verify_ids=torch.cat((prompt_ids,prefix),dim=1)
        with torch.inference_mode():
            logits=model_target(input_ids=verify_ids,use_cache=False).logits[0].float()
        offset=prompt_ids.shape[1]-1
        prows=[torch.softmax(logits[offset+i],dim=-1) for i in range(len(proposal)+1)]
        def uni():return float(torch.rand((),device=device,generator=generator))
        def draw(p):return sample(p,generator)
        chosen,accepted,examined,bonus=exact_step(prows,qrows,proposal,uni,draw)
        counts.proposed+=len(proposal)
        counts.examined+=examined
        counts.accepted+=accepted
        counts.target_verifications+=1;counts.rounds+=1
        counts.accepted_prefix_lengths.append(accepted)
        counts.bonus_rounds+=int(bonus);counts.rejected_rounds+=int(not bonus)
        for tok in chosen:
            if len(output)>=max_new_tokens:break
            output.append(tok)
            if tok==eos_id:return output,counts
    return output,counts

def target_only(model,prompt_ids,max_new_tokens,eos_id,generator=None):
    out=[]
    for _ in range(max_new_tokens):
        ids=torch.cat((prompt_ids,torch.tensor([out],device=prompt_ids.device)),dim=1) if out else prompt_ids
        tok=sample(next_token_distribution(model,ids),generator)
        out.append(tok)
        if tok==eos_id:break
    return out
