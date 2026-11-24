"""Calibration-only Task 1 algorithms. Inference remains dense FP16; bytes are logical."""
from __future__ import annotations
import math
import torch

BITS=(2,3,4,8)
def payload_bytes(rows,cols,bits,group=128):
    return sum(rows*(math.ceil(min(group,cols-i)*bits/8)+4)
               for i in range(0,cols,group))
def quantize(w,bits,group=128):
    """Asymmetric affine; nearest-even; FP16 scale + signed INT16 zero-point per row-group.
    Each group is byte aligned, trailing partial group excludes padded values from statistics.
    """
    if bits not in BITS or w.ndim!=2 or group<1:raise ValueError('quantizer shape/bits')
    rows,cols=w.shape; chunks=[]
    for i in range(0,cols,group):
        a=w[:,i:i+group].float()
        mn=a.amin(1,keepdim=True);mx=a.amax(1,keepdim=True)
        s=((mx-mn)/(2**bits-1)).clamp_min(1e-12)
        z=torch.round(-mn/s).clamp(0,2**bits-1)
        q=torch.round(a/s+z).clamp(0,2**bits-1)
        chunks.append(((q-z)*s).to(w.dtype))
    return torch.cat(chunks,dim=1),payload_bytes(rows,cols,bits,group)
def mse(original,recon,feature_sq):
    x=feature_sq.to(original.device).float().clamp_min(0)
    return float((original.float().sub(recon.float()).square()*x[None,:]).sum(1).mean())
def awq(w,feature_sq,bits,grid=(0.,.25,.5,.75,1.)):
    """Diagonal-output-error AWQ-style search; explicit inverse scales counted as FP16."""
    a=feature_sq.to(w.device).float().clamp_min(1e-9).sqrt()
    a=(a/a.mean()).clamp_min(1e-6)
    best=None
    for alpha in grid:
        scale=a.pow(alpha).clamp(1e-3,1e3)
        full=w.float()*scale[None,:]
        preservation=float((full/scale[None,:]-w.float()).abs().max())
        if preservation>1e-4*max(1.,float(w.abs().max())):raise AssertionError('rescaling')
        quant,nb=quantize(full.to(w.dtype),bits)
        recon=(quant.float()/scale[None,:]).to(w.dtype)
        err=mse(w,recon,feature_sq)
        if best is None or err<best[0]:
            best=(err,recon,nb+2*w.shape[1],alpha,preservation)
    return best
def ranked_mask(w,score,zeros,structured=False):
    rows,cols=w.shape
    mask=torch.ones_like(w,dtype=torch.bool)
    if structured:
        if cols%4:raise ValueError('unaligned structured groups')
        ix=torch.argsort(score.reshape(rows,-1,4),dim=-1,stable=True)[...,:2]
        mask.reshape(rows,-1,4).scatter_(-1,ix,False)
    else:
        if not 0<=zeros<=w.numel():raise ValueError('invalid zero count')
        base,extra=divmod(zeros,rows)
        order=torch.argsort(score,dim=1,stable=True)
        if base:mask.scatter_(1,order[:,:base],False)
        if extra:mask[:extra].scatter_(1,order[:extra,base:base+1],False)
    assert int((~mask).sum())==(w.numel()//2 if structured else zeros)
    return mask
def bounded_allocation(spec,target,lo=.1,hi=.85):
    """spec[name]=(eligible weights, positive calibration error proxy)."""
    names=sorted(spec)
    low={n:math.ceil(lo*spec[n][0]) for n in names}
    high={n:math.floor(hi*spec[n][0]) for n in names}
    if not sum(low.values())<=target<=sum(high.values()):raise ValueError('infeasible bounds')
    allocation=low.copy()
    weights={n:spec[n][0]/max(float(spec[n][1]),1e-12) for n in names}
    left=target-sum(allocation.values())
    while left:
        active=[n for n in names if allocation[n]<high[n]]
        total=sum(weights[n] for n in active)
        proposed={n:min(high[n]-allocation[n],int(left*weights[n]/total)) for n in active}
        if not sum(proposed.values()):proposed[max(active,key=lambda n:(weights[n],n))]=1
        for name,num in proposed.items():allocation[name]+=num;left-=num
    assert sum(allocation.values())==target
    return allocation
def mixed_allocation(options,budget):
    """options[name][bits]=(calibration error, packed bytes including compensation)."""
    selected={n:2 for n in sorted(options)}
    used=sum(options[n][2][1] for n in selected)
    if used>budget:raise ValueError('INT2 metadata already exceeds budget')
    while True:
        upgrades=[]
        for n,old in selected.items():
            for b in BITS:
                if b<=old:continue
                cost=options[n][b][1]-options[n][old][1]
                improvement=options[n][old][0]-options[n][b][0]
                if cost>0 and improvement>0 and used+cost<=budget:
                    upgrades.append((improvement/cost,n,b,cost))
        if not upgrades:break
        _,name,b,cost=max(upgrades)
        selected[name]=b;used+=cost
    assert used<=budget
    return selected,used
def unique_param_count(model,eligible_suffixes):
    seen=set();eligible=0;other_bytes=0
    for n,p in model.named_parameters():
        key=(p.data_ptr(),p.numel(),str(p.dtype))
        if key in seen:continue
        seen.add(key)
        if n.endswith('.weight') and n.split('.')[-2] in eligible_suffixes:
            eligible+=p.numel()
        else:other_bytes+=p.numel()*p.element_size()
    return eligible,other_bytes
