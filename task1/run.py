"""Task 1 runner (dense FP16 execution, estimated packed storage).
Staff public dataset generator remains authoritative; no test-driven allocation.
"""
import argparse, gzip, hashlib, json, math, os, pathlib, random, time, re
import numpy as np
import torch
import torch.nn.functional as F
import protocol as proto

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
def perplexity(m,windows,device,outfile,expected_tokens=50000):
    n=0;loss=0.;rows=[]
    for i,w in enumerate(windows):
        x=torch.tensor(w["token_ids"],device=device,dtype=torch.long)[None,:]
        lo,hi=w.get("score_target_offsets",[1,x.shape[1]])
        assert 1<=lo<hi<=x.shape[1]
        logits=logprobs(m,x)
        target=x[0,1:]; scores=F.cross_entropy(logits,target,reduction="none")
        vals=scores[lo-1:hi-1]
        item={"id":i,"source_row":w.get("source_row"),"token_offset":w.get("token_offset"),
              "tokens":len(vals),"nll_sum":float(vals.sum())}
        rows.append(item);n+=len(vals);loss+=item["nll_sum"]
    if expected_tokens is not None and n!=expected_tokens:
        raise AssertionError(f"Unexpected token count {n} vs {expected_tokens}")
    dump(outfile,rows)
    return {"tokens":n,"mean_nll":loss/n,"ppl":math.exp(loss/n),"windows":len(rows)}
def hs_preprocess(text):
    text=text.strip().replace(" [title]",". ")
    return re.sub(r"\[.*?\]", "", text)
def hs_prompt(ex):
    ctx=ex["ctx_a"]+" "+ex["ctx_b"].capitalize()
    return hs_preprocess(ex["activity_label"]+": "+ctx)
def hellaswag(m,examples,tok,device,outfile,limit=2000):
    rows=[]
    for i,rec in enumerate(examples[:limit]):
        ex=rec["example"]; text=hs_prompt(ex)
        beginnings=tok.encode(text,add_special_tokens=False)
        if not beginnings: raise ValueError("Empty HellaSwag prompt")
        scores=[]
        choices=[hs_preprocess(e) for e in ex["endings"]]
        for ending in choices:
            # Harness uses a leading space (default target_delimiter).
            combined=tok.encode(text+" "+ending,add_special_tokens=False)
            plen=len(beginnings)
            if combined[:plen]!=beginnings:
                # Explicit separate tokenization convention is fixed for every model.
                cont=tok.encode(" "+ending,add_special_tokens=False)
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
        normalized=[v/max(1,len(e)) for v,e in zip(scores,choices)]
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
        self.stats={};self.samples={};self.handles=[]
        for name,layer in modules(model).items():
            def f(module,args,name=name):
                x=args[0].detach().float().reshape(-1,module.in_features)
                if name not in self.samples:
                    self.samples[name]=x[:64].to(torch.float16).cpu()
                sq=x.square().sum(0).cpu();ab=x.abs().sum(0).cpu()
                if name not in self.stats:self.stats[name]=[sq,ab,x.shape[0]]
                else:
                    self.stats[name][0]+=sq;self.stats[name][1]+=ab;self.stats[name][2]+=x.shape[0]
            self.handles.append(layer.register_forward_pre_hook(f))
    def close(self):
        for h in self.handles:h.remove()
    def norms(self):return {k:v[0].sqrt() for k,v in self.stats.items()}
    def means(self):return {k:v[1]/v[2] for k,v in self.stats.items()}
    def squares(self):return {k:v[0]/v[2] for k,v in self.stats.items()}
def apply(m,method,sparsity,bits,norms,feature_sq,samples):
    """Apply one condition, using calibration-only, matched-budget choices."""
    mods=modules(m)
    required=getattr(m.config,"num_hidden_layers",None)
    if required is not None and len(mods)!=required*7:
        raise RuntimeError(f"Eligible projection mismatch: {len(mods)} vs {required*7}")
    eligible=sum(layer.weight.numel() for layer in mods.values())
    unique,excluded=proto.unique_param_count(m,set(ELIGIBLE))
    if unique!=eligible:raise RuntimeError("Eligible/tied parameter storage mismatch")
    info={"representation":"dense_fp16_for_numerical_evaluation",
          "groupsize":128,"quantizer":"affine asymmetric per-row-group, nearest-even, scale FP16 + zero point INT16",
          "eligible_weights":eligible,"excluded_fp16_bytes":excluded,
          "calibration_proxy":"diagonal E[x_j^2] output MSE"}
    alloc={};bits_map={};packed={};sensitivity={}
    target=round(eligible*sparsity)
    if method=="nonuniform":
        for name,layer in mods.items():
            w=layer.weight.detach()
            s=w.float().abs()*norms[name].to(w.device).float()[None,:]
            probe=proto.ranked_mask(w,s,round(w.numel()*.20))
            error=proto.mse(w,w*probe,feature_sq[name])
            sensitivity[name]=max(error,1e-12)
        alloc=proto.bounded_allocation(
            {n:(l.weight.numel(),sensitivity[n]) for n,l in mods.items()},target)
        assert sum(alloc.values())==target
        info["allocation_zeros"]=alloc
        info["sensitivity_probe"]=sensitivity
    if method=="mixed":
        options={};budget=0
        for name,layer in mods.items():
            options[name]={}
            for bit in (2,3,4,8):
                result=proto.awq(layer.weight,feature_sq[name],bit)
                options[name][bit]=(result[0],result[2])
            budget+=options[name][bits][1]
        bits_map,used=proto.mixed_allocation(options,budget)
        info["eligible_q2_budget_bytes"]=budget
        info["eligible_q3_bytes"]=used
        info["bits_by_projection"]=bits_map
        info["calibration_bit_errors"]={k:{str(b):round(v[0],8) for b,v in d.items()} for k,d in options.items()}
    per_projection={}
    total_zero=0;compressed_eligible_bytes=0
    with torch.no_grad():
        for name,layer in mods.items():
            w=layer.weight
            initial=w.detach().clone()
            score=initial.float().abs()
            if method in ("wanda","nonuniform","structured"):
                score*=norms[name].to(w.device).float()[None,:]
            if method in ("magnitude","wanda","nonuniform","structured"):
                zeros=(alloc[name] if method=="nonuniform" else
                       w.numel()//2 if method=="structured" else
                       w.shape[0]*round(w.shape[1]*sparsity))
                mask=proto.ranked_mask(w,score,zeros,method=="structured")
                w.mul_(mask)
                compressed_eligible_bytes+=((w.numel()+7)//8)+2*(w.numel()-zeros)
            elif method=="ptq":
                recon,nb=proto.quantize(w,bits)
                w.copy_(recon)
                compressed_eligible_bytes+=nb
                packed[name]=nb
            elif method in ("awq","mixed"):
                b=bits_map[name] if method=="mixed" else bits
                err,recon,nb,alpha,fp_err=proto.awq(w,feature_sq[name],b)
                w.copy_(recon)
                compressed_eligible_bytes+=nb;packed[name]=nb
                per_projection.setdefault(name,{})["awq_alpha"]=alpha
                per_projection[name]["full_precision_preservation_maxerr"]=fp_err
                per_projection[name]["bits"]=b
            else:
                compressed_eligible_bytes+=2*w.numel()
            measured_zeros=int((w==0).sum())
            total_zero+=measured_zeros
            q=torch.quantile(score.float().reshape(-1),
                             torch.tensor([.10,.50,.90],device=score.device)).cpu().tolist()
            per_projection.setdefault(name,{}).update({
                "eligible_weights":w.numel(),"zero_weights":measured_zeros,
                "score_p10_p50_p90":q,
                "calibration_output_mse_proxy":proto.mse(initial,w,feature_sq[name]),
                "calibration_output_mse_sampled":float(F.linear(samples[name].to(w.device).float(),
                         initial.float()-w.float()).square().mean()),
                "sampled_calibration_rows":len(samples[name]),
                "l2_weight_delta":float((initial.float()-w.float()).square().sum().sqrt())})
            del initial
    if method in ("magnitude","wanda","nonuniform","structured"):
        if total_zero!=target:raise AssertionError(f"zero mismatch {total_zero} vs {target}")
    if method=="mixed" and compressed_eligible_bytes!=info["eligible_q3_bytes"]:
        raise AssertionError("mixed budget accounting inconsistent")
    if method=="mixed" and compressed_eligible_bytes>info["eligible_q2_budget_bytes"]:
        raise AssertionError("Q3 exceeds uniform Q2 packed budget")
    info.update({"per_projection":per_projection,"packed_bytes_by_projection":packed,
                 "eligible_zero_count":total_zero,
                 "eligible_sparsity":total_zero/eligible,
                 "packed_eligible_bytes":compressed_eligible_bytes,
                 "packed_total_bytes":compressed_eligible_bytes+excluded,
                 "dense_fp16_storage_bytes":2*eligible+excluded,
                 "dense_gpu_execution":True})
    return info
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
    # Decode latency: 64 greedy cached steps following a fixed 512-token prefill.
    tokens=torch.randint(0,len(tok),(1,512),device=device)
    with torch.inference_mode():
        for _ in range(3):
            prefill=m(tokens,use_cache=True)
            cache=prefill.past_key_values
            m(prefill.logits[:,-1:].argmax(-1),past_key_values=cache,use_cache=True)
        torch.cuda.synchronize()
        times=[]
        for _ in range(5):
            prefill=m(tokens,use_cache=True)
            cache=prefill.past_key_values
            next_id=prefill.logits[:,-1:].argmax(-1)
            torch.cuda.synchronize();t0=time.perf_counter()
            for _ in range(64):
                output=m(next_id,past_key_values=cache,use_cache=True)
                cache=output.past_key_values
                next_id=output.logits[:,-1:].argmax(-1)
            torch.cuda.synchronize()
            times.append(1000*(time.perf_counter()-t0))
    result["decode_64_tokens"]={"latency_ms":times,"repetitions":5}
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
    lock=json.loads((pathlib.Path(a.data)/"data/local_source_revisions.json").read_text())
    revision=lock["revisions"]["model_base"]
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=revision,use_fast=True)
    model=AutoModelForCausalLM.from_pretrained(MODEL,revision=revision,torch_dtype=torch.float16,
                                              attn_implementation="sdpa").to(dev).eval()
    dump(out/"checkpoint.json",{"name":MODEL,"revision":revision,
                                 "tokenizer":tokenizer.name_or_path})
    t_start=time.perf_counter()
    with torch.inference_mode():
        # The same 128 fixed windows are required for every diagnostic, including
        # baseline and magnitude/PTQ output-error comparisons.
        stats=Stats(model)
        for window in records(paths["cal"]):
            ids=torch.tensor(window["token_ids"],device=dev)[None,:]
            model(ids,use_cache=False)
        stats.close()
        t_calibration=time.perf_counter()
        calibration_norms=stats.norms() if stats else {}
        calibration_mse=stats.squares() if stats else {}
        compression=apply(model,a.method,a.sparsity,a.bits,
                          calibration_norms,calibration_mse,stats.samples)
    dump(out/"costs.json",{"calibration_seconds":t_calibration-t_start,
                           "compression_seconds":time.perf_counter()-t_calibration})
    dump(out/"compression.json",compression)
    dump(out/"protocol.json",{"seed":6304,"calibration_count":128,
         "development_count":32,"group_size":128,"attention_backend":"sdpa",
         "quantization":"asymmetric affine with nearest-even group rounding",
         "score_convention":"conditional log likelihood and characters for HellaSwag acc_norm"})
    development=perplexity(model,records(paths["dev"]),dev,
                           out/"dev_windows.json",expected_tokens=32*511)
    dump(out/"development_summary.json",development)
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
