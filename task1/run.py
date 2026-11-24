"""Task 1 runner (dense FP16 execution, estimated packed storage).
Staff public dataset generator remains authoritative; no test-driven allocation.
"""
import argparse, gzip, hashlib, json, math, os, pathlib, random, time
import numpy as np
import torch
import torch.nn.functional as F

ELIGIBLE = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
MODEL = "Qwen/Qwen2.5-1.5B"
def records(path):
    with gzip.open(path, "rt", encoding="utf8") as f:
        return [json.loads(line) for line in f if line.strip()]
def dump(path, obj):
    p=pathlib.Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    tmp=p.with_suffix(p.suffix+".tmp"); tmp.write_text(json.dumps(obj,indent=2,default=str))
    os.replace(tmp,p)
def digest(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1048576),b""):h.update(b)
    return h.hexdigest()
def modules(m):
    return {n:x for n,x in m.named_modules() if isinstance(x,torch.nn.Linear) and n.split(".")[-1] in ELIGIBLE}
def logprobs(m,ids):
    with torch.inference_mode():
        return m(input_ids=ids,use_cache=False).logits[0,:-1].float()
def perplexity(m,windows,device,outfile):
    n=0;loss=0.;rows=[]
    for i,w in enumerate(windows):
        x=torch.tensor(w["token_ids"],device=device,dtype=torch.long)[None,:]
        lo,hi=w["score_target_offsets"]
        assert 1<=lo<hi<=x.shape[1]
        logits=logprobs(m,x)
        target=x[0,1:]; scores=F.cross_entropy(logits,target,reduction="none")
        vals=scores[lo-1:hi-1]
        item={"id":i,"source_row":w.get("source_row"),"token_offset":w.get("token_offset"),
              "tokens":len(vals),"nll_sum":float(vals.sum())}
        rows.append(item);n+=len(vals);loss+=item["nll_sum"]
    assert n==50000,(n,len(rows))
    dump(outfile,rows)
    return {"tokens":n,"mean_nll":loss/n,"ppl":math.exp(loss/n),"windows":len(rows)}
def hs_prompt(ex):
    a=ex["ctx_a"].strip()
    b=ex["ctx_b"].strip()
    return a+" "+(b[:1].upper()+b[1:] if b else "")
def hellaswag(m,examples,tok,device,outfile,limit=2000):
    rows=[]
    for i,rec in enumerate(examples[:limit]):
        ex=rec["example"]; text=hs_prompt(ex)
        beginnings=tok.encode(text,add_special_tokens=False)
        if not beginnings: raise ValueError("Empty HellaSwag prompt")
        scores=[]
        for ending in ex["endings"]:
            # Tokenize prompt and its continuation together to respect boundary merges.
            combined=tok.encode(text+ending,add_special_tokens=False)
            plen=len(beginnings)
            if combined[:plen]!=beginnings:
                # Explicit separate tokenization convention is fixed for every model.
                cont=tok.encode(ending,add_special_tokens=False)
                combined=beginnings+cont
            else:cont=combined[plen:]
            if not cont:raise ValueError("Empty HellaSwag ending")
            ids=torch.tensor(combined,device=device)[None,:]
            if len(combined)>m.config.max_position_embeddings:
                raise ValueError("HellaSwag sample exceeds model context")
            with torch.inference_mode():
                logits=m(ids,use_cache=False).logits[0,plen-1:-1].float()
                val=F.log_softmax(logits,dim=-1).gather(1,torch.tensor(cont,device=device)[:,None]).sum()
            scores.append(float(val))
        normalized=[v/max(1,len(e)) for v,e in zip(scores,ex["endings"])]
        label=int(ex["label"])
        rows.append({"id":rec["source_row_index"],"order":rec["order"],"label":label,
                     "scores":scores,"acc":int(np.argmax(scores)==label),
                     "acc_norm":int(np.argmax(normalized)==label),
                     "normalized_scores":normalized})
    dump(outfile,rows)
    return {"n":len(rows),"accuracy":float(np.mean([r["acc"] for r in rows])),
            "acc_norm":float(np.mean([r["acc_norm"] for r in rows]))}
class Stats:
    def __init__(self,model):
        self.stats={};self.handles=[]
        for name,layer in modules(model).items():
            def f(module,args,name=name):
                x=args[0].detach().float().reshape(-1,module.in_features)
                sq=x.square().sum(0).cpu();ab=x.abs().sum(0).cpu()
                if name not in self.stats:self.stats[name]=[sq,ab,x.shape[0]]
                else:
                    self.stats[name][0]+=sq;self.stats[name][1]+=ab;self.stats[name][2]+=x.shape[0]
            self.handles.append(layer.register_forward_pre_hook(f))
    def close(self):
        for h in self.handles:h.remove()
    def norms(self):return {k:v[0].sqrt() for k,v in self.stats.items()}
    def means(self):return {k:v[1]/v[2] for k,v in self.stats.items()}
def quantize(w,bits,group_size=128):
    rows,cols=w.shape;groups=math.ceil(cols/group_size);pad=groups*group_size-cols
    a=F.pad(w.float(),(0,pad)).reshape(rows,groups,group_size)
    mn=a.amin(-1,keepdim=True);mx=a.amax(-1,keepdim=True)
    scale=((mx-mn)/((1<<bits)-1)).clamp_min(1e-12)
    q=torch.round((a-mn)/scale).clamp(0,(1<<bits)-1)
    restored=(q*scale+mn).reshape(rows,-1)[:,:cols].to(w.dtype)
    bytes_=rows*groups*(math.ceil(group_size*bits/8)+4)
    return restored,bytes_
def awq(w,acts,bits,return_error=False):
    x=acts.to(w.device).float().clamp_min(1e-6);x=x/x.mean()
    best=None
    for alpha in (0.,.25,.5,.75,1.):
        scale=x.pow(alpha).clamp(1e-3,1e3)
        z,nb=quantize(w.float()*scale[None,:],bits)
        z=z.float()/scale[None,:]
        err=((w.float()-z).square()*x.square()[None,:]).mean().item()
        if best is None or err<best[0]:best=(err,z.to(w.dtype),nb+2*len(scale),alpha)
    return best if return_error else (best[1],best[2],best[3])
def apply(m,method,sparsity,bits,norms,means):
    mods=modules(m);infos={};num=sum(l.weight.numel() for l in mods.values())
    with torch.no_grad():
        if method=="nonuniform":
            sens={}
            for name,layer in mods.items():
                z=(layer.weight.detach().float().abs()*norms[name].to(layer.weight.device)[None,:]).flatten()
                k=max(1,int(z.numel()*.5));sens[name]=max(1e-12,float(torch.topk(z,k,largest=False).values.square().mean()))
            # bounded integer allocation to exactly match eligible-weight zeros
            target=round(num*sparsity);caps={};alloc={};priority={}
            for name,l in mods.items():
                n=l.weight.numel();lo=math.ceil(.10*n);hi=math.floor(.85*n)
                caps[name]=hi;alloc[name]=lo;priority[name]=n/sens[name]
            remaining=target-sum(alloc.values())
            if remaining<0:raise ValueError("Infeasible pruning budget")
            while remaining:
                active=[k for k in alloc if alloc[k]<caps[k]]
                if not active:raise ValueError("Infeasible pruning upper bound")
                total=sum(priority[k] for k in active)
                chosen={k:min(caps[k]-alloc[k],int(remaining*priority[k]/total)) for k in active}
                if not sum(chosen.values()):
                    key=max(active,key=lambda k:(priority[k],k));chosen[key]=1
                for k,v in chosen.items():alloc[k]+=v;remaining-=v
            infos["allocation"]={k:{"zeros":alloc[k],"sparsity":alloc[k]/mods[k].weight.numel(),
                                     "sensitivity":sens[k]} for k in alloc}
        if method=="mixed":
            # Deterministic calibration-only bit-gain selection; Q2 INT3/INT4 packed budget.
            choices={};options={};budget=0;used=0
            for name,l in mods.items():
                options[name]={b:awq(l.weight,means[name],b,True) for b in (2,3,4,8)}
                budget+=options[name][bits][2]
                choices[name]=2;used+=options[name][2][2]
            if used>budget:raise ValueError("Mixed budget infeasible")
            while True:
                best=None
                for name in mods:
                    old=choices[name];new={2:3,3:4,4:8,8:None}[old]
                    if new is None:continue
                    before=options[name][old];after=options[name][new]
                    cost=after[2]-before[2];gain=max(0.,before[0]-after[0])
                    if used+cost<=budget and gain>0:
                        candidate=(gain/max(cost,1),name,new,cost)
                        if best is None or candidate>best:best=candidate
                if best is None:break
                _,name,new,cost=best;choices[name]=new;used+=cost
            infos["bit_assignment"]=choices;infos["eligible_budget_bytes"]=budget
        for name,l in mods.items():
            w=l.weight;rows,cols=w.shape
            if method in ("magnitude","wanda","structured","nonuniform"):
                score=w.detach().float().abs()
                if method!="magnitude":score*=norms[name].to(w.device)[None,:]
                if method=="structured":
                    if cols%4:raise ValueError("2:4 alignment")
                    small=torch.argsort(score.reshape(rows,-1,4),dim=-1,stable=True)[...,:2]
                    mask=torch.ones((rows,cols//4,4),device=w.device,dtype=torch.bool)
                    mask.scatter_(-1,small,False);mask=mask.reshape(rows,cols)
                elif method=="nonuniform":
                    z=infos["allocation"][name]["zeros"]
                    ids=torch.argsort(score.reshape(-1),stable=True)[:z]
                    mask=torch.ones(w.numel(),device=w.device,dtype=torch.bool)
                    mask[ids]=False;mask=mask.reshape_as(w)
                else:
                    count=round(cols*sparsity)
                    ix=torch.argsort(score,dim=-1,stable=True)[:,:count]
                    mask=torch.ones_like(w,dtype=torch.bool);mask.scatter_(-1,ix,False)
                w.mul_(mask)
            elif method=="ptq":
                z,nb=quantize(w,bits);w.copy_(z);infos[name]={"packed_bytes":nb,"bits":bits}
            elif method in ("awq","mixed"):
                b=infos["bit_assignment"][name] if method=="mixed" else bits
                z,nb,alpha=awq(w,means[name],b)
                w.copy_(z);infos[name]={"packed_bytes":nb,"bits":b,"alpha":alpha}
        zeros=sum(int((l.weight==0).sum().item()) for l in mods.values())
        effective=round(num*sparsity) if method=="nonuniform" else zeros
        total_bytes=2*(sum(p.numel() for p in m.parameters())-num)
        if method in ("ptq","awq","mixed"):total_bytes+=sum(info["packed_bytes"] for info in infos.values() if isinstance(info,dict) and "packed_bytes" in info)
        elif method=="baseline":total_bytes+=2*num
        else:total_bytes+=math.ceil(num/8)+2*(num-zeros)
    infos.update({"eligible_weights":num,"zero_weights":zeros,"target_zeros":effective,
                  "storage_estimate_bytes":total_bytes,"dense_fp16_execution":True})
    return infos
def bench(m,tok,device,out):
    result={}
    for size in (512,2048):
        ids=torch.randint(0,len(tok),(1,size),device=device)
        with torch.inference_mode():
            for _ in range(3):m(ids,use_cache=True)
            torch.cuda.synchronize()
            times=[];torch.cuda.reset_peak_memory_stats()
            for _ in range(5):
                torch.cuda.synchronize();t=time.perf_counter()
                m(ids,use_cache=True);torch.cuda.synchronize()
                times.append(1000*(time.perf_counter()-t))
        result[str(size)]={"prefill_ms":times,"peak_allocated_bytes":torch.cuda.max_memory_allocated()}
    dump(out,result)
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",required=True);p.add_argument("--output",required=True)
    p.add_argument("--method",required=True,choices=["baseline","magnitude","wanda","structured","nonuniform","ptq","awq","mixed"])
    p.add_argument("--sparsity",type=float,default=.5);p.add_argument("--bits",type=int,default=4)
    p.add_argument("--diagnostic-only",action="store_true")
    a=p.parse_args();torch.manual_seed(6304);random.seed(6304);np.random.seed(6304)
    from transformers import AutoModelForCausalLM,AutoTokenizer
    root=pathlib.Path(a.data)/"data/public_locked";out=pathlib.Path(a.output);out.mkdir(parents=True,exist_ok=True)
    paths={k:root/f for k,f in {
        "cal":"c4_calibration_locked.jsonl.gz","dev":"c4_development_locked.jsonl.gz",
        "wiki":"wikitext2_test_windows_locked.jsonl.gz",
        "c4":"c4_validation_windows_locked.jsonl.gz","hs":"hellaswag_val_locked.jsonl.gz"}.items()}
    for k,v in paths.items():
        if not v.is_file():raise FileNotFoundError(f"{k}: {v}")
    dump(out/"config.json",vars(a))
    dump(out/"inputs.json",{k:digest(v) for k,v in paths.items()})
    import transformers
    dump(out/"environment.json",{"torch":torch.__version__,"transformers":transformers.__version__,
         "cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(0),"seed":6304})
    dev=torch.device("cuda")
    tokenizer=AutoTokenizer.from_pretrained(MODEL,use_fast=True)
    model=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float16,
                                              attn_implementation="sdpa").to(dev).eval()
    dump(out/"checkpoint.json",{"name":MODEL,"revision":model.config._commit_hash,
                                 "tokenizer":tokenizer.name_or_path})
    with torch.inference_mode():
        stats=None
        if a.method in ("wanda","structured","nonuniform","awq","mixed"):
            stats=Stats(model)
            for window in records(paths["cal"]):
                ids=torch.tensor(window["token_ids"],device=dev)[None,:]
                model(ids,use_cache=False)
            stats.close()
        compression=apply(model,a.method,a.sparsity,a.bits,
                          stats.norms() if stats else {},
                          stats.means() if stats else {})
    dump(out/"compression.json",compression)
    for key in ("wiki","c4"):
        result=perplexity(model,records(paths[key]),dev,out/f"{key}_windows.json")
        dump(out/f"{key}_summary.json",result)
    hs=records(paths["hs"])
    result=hellaswag(model,hs,tokenizer,dev,out/"hellaswag_examples.json",
                     512 if a.diagnostic_only else 2000)
    dump(out/"hellaswag_summary.json",result)
    bench(model,tokenizer,dev,out/"timing.json")
    dump(out/"completed.json",{"method":a.method,"finished":True})
if __name__=="__main__": main()
