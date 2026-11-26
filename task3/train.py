"""Task 3 soft forward-KL LoRA distillation: matched scored-token quotas.
One seed/method per Kaggle kernel. No held-out prompts enter training.
"""
from __future__ import annotations
import argparse,gzip,hashlib,json,math,os,pathlib,random,time
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer,AutoModelForCausalLM
from peft import LoraConfig, get_peft_model,PeftModel

TEACHER="Qwen/Qwen2.5-1.5B-Instruct"
DRAFT="Qwen/Qwen2.5-0.5B-Instruct"
QUOTAS={"math":30000,"short":10000}
CAPS={"math":192,"short":32}

def rows(path):
    op=gzip.open if str(path).endswith(".gz") else open
    with op(path,"rt",encoding="utf8") as f:
        return [json.loads(line) for line in f if line.strip()]

def write_json(file,obj):
    file.parent.mkdir(parents=True,exist_ok=True)
    file.write_text(json.dumps(obj,indent=2,default=str)+"\n")

def prompts(staff):
    locked=staff/"data/public_locked/gsm8k_train_locked.jsonl.gz"
    short=staff/"data/short_form/short_train.jsonl"
    math_rows=rows(locked)
    short_rows=rows(short)
    if len(math_rows)!=500 or len(short_rows)!=300:
        raise RuntimeError(f"Unexpected fixed training sets {len(math_rows)}, {len(short_rows)}")
    result={"math":[],"short":[]}
    for r in math_rows:
        result["math"].append({"id":f"GSM8K-{r['order']:04d}",
            "text":r["example"]["question"],"domain":"math"})
    for r in short_rows:
        result["short"].append({"id":r["id"],"text":r["input"],"domain":"short",
                                 "family":r["family"]})
    return result

def prompt_ids(tok,text):
    return tok.apply_chat_template([{"role":"user","content":text}],
                    tokenize=True,add_generation_prompt=True,return_tensors="pt").to("cuda")

def enforce_tokenizer_alignment(teacher_tok,draft_tok):
    examples=["Hello there","12 June 2024","Extract quantity: 40 grams",
              "Calculate 11 + 19","π and 🎯"]
    for example in examples:
        t=prompt_ids(teacher_tok,example).cpu()
        d=prompt_ids(draft_tok,example).cpu()
        if not torch.equal(t,d):raise RuntimeError("Draft and teacher chat template mismatch")
    if teacher_tok.eos_token_id!=draft_tok.eos_token_id:
        raise RuntimeError("Target/student EOS mismatch")

def rollout(model,ids,cap,eos_id,generator):
    with torch.inference_mode():
        sample=model.generate(ids,max_new_tokens=cap,do_sample=True,
             temperature=1.0,top_p=1.,top_k=0,
             pad_token_id=eos_id,eos_token_id=eos_id,
             generator=generator,use_cache=True)
    response=sample[0,ids.shape[1]:].tolist()
    termination="eos" if response and response[-1]==eos_id else "cap"
    if not response:raise RuntimeError("Empty rollout")
    return response,termination

def cyclic_indices(size,seed,category):
    rng=random.Random(seed+(0 if category=="math" else 731));positions=list(range(size))
    while True:
        rng.shuffle(positions)
        for index in positions:yield index

def schedule(seed,sets):
    iterators={d:cyclic_indices(len(sets[d]),seed,d) for d in QUOTAS}
    used=dict.fromkeys(QUOTAS,0)
    while any(used[d]<QUOTAS[d] for d in QUOTAS):
        # Choose most under-filled category relative to its fixed quota.
        active=[d for d in ("math","short") if used[d]<QUOTAS[d]]
        domain=min(active,key=lambda d:(used[d]/QUOTAS[d], d))
        source=sets[domain][next(iterators[domain])]
        used[domain]+=1
        yield domain,source

def make_model(name,revision):
    return AutoModelForCausalLM.from_pretrained(name,revision=revision,
           torch_dtype=torch.float16,attn_implementation="sdpa").to("cuda")

def forward_kl(teacher,student,ids,response,offset,n):
    # Logits at prompt length -1 + response token offset predict that token.
    full=torch.cat([ids,torch.tensor([response],device="cuda")],dim=1)
    begin=ids.shape[1]-1+offset
    end=begin+n
    with torch.inference_mode():
        t_logits=teacher(full,use_cache=False).logits[:,begin:end].float()
        t_prob=t_logits.softmax(dim=-1)
    with torch.autocast(device_type="cuda",dtype=torch.float16):
        s_logits=student(full,use_cache=False).logits[:,begin:end].float()
        kl=F.kl_div(F.log_softmax(s_logits,dim=-1),t_prob,reduction="none").sum(-1)
        loss=kl.sum()
    if not torch.isfinite(loss):raise FloatingPointError("Nonfinite full-vocabulary forward KL")
    return loss

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--staff",type=pathlib.Path,required=True)
    p.add_argument("--out",type=pathlib.Path,required=True)
    p.add_argument("--method",choices=("off","on"),required=True)
    p.add_argument("--seed",type=int,choices=(6304,6305),required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    torch.backends.cuda.matmul.allow_tf32=False
    if not torch.cuda.is_available():raise RuntimeError("Task 3 requires CUDA GPU")
    source=json.loads((a.staff/"data/local_source_revisions.json").read_text())["revisions"]
    sets=prompts(a.staff)
    tok=AutoTokenizer.from_pretrained(TEACHER,revision=source["model_target"])
    draft_tok=AutoTokenizer.from_pretrained(DRAFT,revision=source["model_draft"])
    enforce_tokenizer_alignment(tok,draft_tok)
    teacher=make_model(TEACHER,source["model_target"]).eval()
    teacher.requires_grad_(False)
    student=make_model(DRAFT,source["model_draft"])
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    lcfg=LoraConfig(r=8,lora_alpha=16,lora_dropout=0.,
             target_modules=["q_proj","k_proj","v_proj","o_proj"],bias="none",
             task_type="CAUSAL_LM")
    student=get_peft_model(student,lcfg)
    for name,parameter in student.named_parameters():
        if parameter.requires_grad:parameter.data=parameter.data.float()
    student.train()
    adapter_count=sum(x.numel() for x in student.parameters() if x.requires_grad)
    if adapter_count<=0 or any(x.requires_grad for x in teacher.parameters()):
        raise RuntimeError("Broken frozen-backbone/adapter configuration")
    opt=torch.optim.AdamW((x for x in student.parameters() if x.requires_grad),
                           lr=3e-4,weight_decay=.01)
    scaler=torch.amp.GradScaler("cuda")
    gen=torch.Generator(device="cuda").manual_seed(a.seed)
    tracker={"method":a.method,"seed":a.seed,
      "source_revisions":source,"adapter_parameters":adapter_count,
      "teacher_total_parameters":sum(p.numel() for p in teacher.parameters()),
      "student_total_parameters":sum(p.numel() for p in student.parameters()),
      "quota":QUOTAS,"updates":0,"skipped_updates":0,"scored":{"math":0,"short":0},
      "teacher_rollout_seconds":0.,"student_rollout_seconds":0.,
      "teacher_scoring_training_seconds":0.,"merge_seconds":0.,
      "train_start_unix":time.time(),"loss_records":[]}
    log=a.out/"rollouts.jsonl"
    # Off-policy: generate a frozen teacher pool before *any* optimizer updates.
    prepared=[]
    if a.method=="off":
        prepared_count={"math":0,"short":0}
        for domain,record in schedule(a.seed,sets):
            if prepared_count[domain]>=QUOTAS[domain]:continue
            ids=prompt_ids(tok,record["text"])
            t0=time.perf_counter()
            teacher.eval()
            response,termination=rollout(teacher,ids,CAPS[domain],tok.eos_token_id,gen)
            tracker["teacher_rollout_seconds"]+=time.perf_counter()-t0
            allowed=min(len(response),QUOTAS[domain]-prepared_count[domain])
            prepared.append((domain,record,response[:allowed],termination))
            prepared_count[domain]+=allowed
            if all(prepared_count[d]==QUOTAS[d] for d in QUOTAS):break
        write_json(a.out/"frozen_teacher_pool_manifest.json",{
            "seed":a.seed,"pool_presentations":len(prepared),
            "scored_tokens":prepared_count})
    else:prepared=None
    opt.zero_grad(set_to_none=True)
    tokens_in_update=0;total_loss=0.;presentation=0
    def update():
        nonlocal tokens_in_update,total_loss
        if not tokens_in_update:return
        # Each response segment contributed sum(KL)/256; correct final update normalization.
        if tokens_in_update!=256:
            factor=256/tokens_in_update
            for param in student.parameters():
                if param.grad is not None:param.grad.mul_(factor)
        scaler.unscale_(opt)
        norm=torch.nn.utils.clip_grad_norm_(
            [x for x in student.parameters() if x.requires_grad],1.0)
        old_scale=scaler.get_scale()
        scaler.step(opt);scaler.update()
        new_scale=scaler.get_scale()
        if new_scale<old_scale:tracker["skipped_updates"]+=1
        else:tracker["updates"]+=1
        tracker["loss_records"].append({"step":tracker["updates"],
            "teacher_scored_total":sum(tracker["scored"].values()),
            "teacher_scored_math":tracker["scored"]["math"],
            "teacher_scored_short":tracker["scored"]["short"],
            "forward_kl_mean":total_loss/tokens_in_update,
            "grad_norm_before_clip":float(norm),
            "scale":float(new_scale),"lr":3e-4})
        opt.zero_grad(set_to_none=True)
        tokens_in_update=0;total_loss=0.
    iterator=iter(prepared) if prepared is not None else schedule(a.seed,sets)
    with log.open("w",encoding="utf8") as fout:
        while any(tracker["scored"][d]<QUOTAS[d] for d in QUOTAS):
            if prepared is not None:
                domain,record,response,termination=next(iterator)
            else:
                domain,record=next(iterator)
                if tracker["scored"][domain]>=QUOTAS[domain]:continue
                ids=prompt_ids(tok,record["text"])
                student.eval()
                t0=time.perf_counter()
                response,termination=rollout(student,ids,CAPS[domain],tok.eos_token_id,gen)
                tracker["student_rollout_seconds"]+=time.perf_counter()-t0
                student.train()
            remaining=QUOTAS[domain]-tracker["scored"][domain]
            if remaining<=0:continue
            permitted=response[:remaining]
            truncated=len(permitted)<len(response)
            if not permitted:raise RuntimeError("No supervised tokens")
            ids=prompt_ids(tok,record["text"])
            entry={"source_id":record["id"],"domain":domain,"method":a.method,
                  "seed":a.seed,"rollout_seed":a.seed,"response_token_ids":permitted,
                  "teacher_scored_tokens":len(permitted),
                  "termination_reason":"quota" if truncated else termination,
                  "presentation":presentation,"training_steps_start":tracker["updates"]}
            fout.write(json.dumps(entry)+"\n");fout.flush()
            pos=0
            while pos<len(permitted):
                count=min(256-tokens_in_update,len(permitted)-pos)
                t0=time.perf_counter()
                loss_sum=forward_kl(teacher,student,ids,permitted,pos,count)
                scaler.scale(loss_sum/256).backward()
                tracker["teacher_scoring_training_seconds"]+=time.perf_counter()-t0
                total_loss+=float(loss_sum.detach().cpu())
                tokens_in_update+=count
                tracker["scored"][domain]+=count
                pos+=count
                if tokens_in_update==256:update()
            presentation+=1
            if presentation%20==0:
                print("PROGRESS",a.method,a.seed,tracker["scored"],
                      "optimizer updates",tracker["updates"],flush=True)
            if presentation%50==0:
                write_json(a.out/"training_status.json",tracker)
    update()
    if tracker["scored"]!=QUOTAS:raise RuntimeError("Token quota mismatch")
    student.save_pretrained(a.out/"adapter",safe_serialization=True)
    print("Adapter saved",a.out/"adapter",flush=True)
    # Merge adapters into a clean, independently instantiated student model.
    t0=time.perf_counter()
    del student,teacher,opt,scaler
    import gc;gc.collect();torch.cuda.empty_cache()
    fresh=make_model(DRAFT,source["model_draft"]).eval()
    wrapped=PeftModel.from_pretrained(fresh,a.out/"adapter",is_trainable=False).eval()
    fixed_ids=prompt_ids(tok,"What is 12 plus 19?")
    with torch.inference_mode():
        prior=wrapped(fixed_ids).logits[:,-1,:].float().cpu()
    merged=wrapped.merge_and_unload().eval()
    with torch.inference_mode():
        after=merged(fixed_ids).logits[:,-1,:].float().cpu()
    error=float((after-prior).abs().max())
    tracker["merged_logit_max_abs_error"]=error
    if error>.1:raise RuntimeError(f"LoRA merge validation failed: {error}")
    merged.save_pretrained(a.out/"merged",safe_serialization=True)
    tok.save_pretrained(a.out/"merged")
    tracker["merge_seconds"]=time.perf_counter()-t0
    tracker["peak_cuda_allocated_bytes"]=torch.cuda.max_memory_allocated()
    tracker["train_wall_seconds"]=time.time()-tracker["train_start_unix"]
    write_json(a.out/"training_status.json",tracker)
    write_json(a.out/"completed.json",{"finished":True,"method":a.method,"seed":a.seed,
                                      "teacher_scored_tokens":tracker["scored"]})
    print("TASK3 TRAINING COMPLETE",a.method,a.seed,tracker["scored"],flush=True)

if __name__=="__main__":main()
