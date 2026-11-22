import sys
import time
import random
import json
import copy
from pathlib import Path
import torch
import torch.nn.functional as F

import os
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab as L
from align import scene, scene_rows, task_cases, write as align_write
from ft_baseline import protect_pool
from ideas import label_grad
from ideas.label_grad import batch_forward
from ideas.transplant import FT
from lab import summarize

lb = L.Lab("qwen3-0.6b")
tok, model = lb.tok, lb.model
TASKS = ["energy", "industrial", "meter_query"]
LRS = {"energy": 1.5e-5, "industrial": 1.8e-5, "meter_query": 1.5e-5}
seal = {t: task_cases(t, "heldout4a", "heldout4b") for t in TASKS}

reps_map = {
    "qkv": lambda lay: lay.self_attn.q_proj,
    "o": lambda lay: lay.self_attn.o_proj,
    "gu": lambda lay: lay.mlp.gate_proj,
    "down": lambda lay: lay.mlp.down_proj,
}
mod_to_rname = {
    "self_attn.q_proj": "qkv",
    "self_attn.k_proj": "qkv",
    "self_attn.v_proj": "qkv",
    "self_attn.o_proj": "o",
    "mlp.gate_proj": "gu",
    "mlp.up_proj": "gu",
    "mlp.down_proj": "down",
}


def collect_input_cov_cpu(seqs_list, last_only=False, weight=1.0, init_cov=None):
    cov = {}
    hs = []
    for l, lay in enumerate(model.model.layers):
        for rname, getter in reps_map.items():
            def make_hk(key):
                def f(_, inp):
                    x = inp[0][0].detach().float()
                    x = x[-1:] if last_only else x
                    cov[key] = cov.get(key, 0) + weight * (x.T @ x)
                return f
            hs.append(getter(lay).register_forward_pre_hook(make_hk((l, rname))))
    with torch.no_grad():
        for ids in seqs_list:
            model(input_ids=ids.to(model.device)[None])
    for h in hs:
        h.remove()
    out = init_cov if init_cov is not None else {}
    for k, v in cov.items():
        v_cpu = v.cpu()
        out[k] = out[k] + v_cpu if k in out else v_cpu
    cov.clear()
    torch.cuda.empty_cache()
    return out


def collect_task_cov_cpu(task_name, n_max=80):
    with scene(task_name):
        cs = task_cases(task_name, "practice_a")[:n_max]
        rs = label_grad.rows(tok, cs)
    suffix_seqs = [torch.cat([p, g]) for p, g in rs]
    return collect_input_cov_cpu(suffix_seqs, last_only=False, weight=1.0)


def eval_all(tag, tasks_to_eval=TASKS, extra=None):
    res = {}
    for t in tasks_to_eval:
        with scene(t):
            res[t] = round(float(lb.score(seal[t])["exact"].mean()), 4)
    gen = lb.general()
    res["arc"] = round(float(gen["arc"]), 4)
    res["kl"] = round(float(gen["kl"]), 5)
    if len(tasks_to_eval) == 3:
        res["mean_3tasks"] = round((res["energy"] + res["industrial"] + res["meter_query"]) / 3.0, 4)
    if extra:
        res.update(extra)
    print(f"  [{tag}]", json.dumps(res), flush=True)
    return res


results = {
    "Unedited_Base": {"energy": 0.085, "industrial": 0.105, "meter_query": 0.1462, "arc": 0.67, "kl": 0.0, "mean_3tasks": 0.1121},
    "Standard_Single_Full_SFT": {
        "energy": {"energy": 0.7688, "arc": 0.575, "kl": 0.02374},
        "industrial": {"industrial": 0.7525, "arc": 0.47, "kl": 0.06453},
        "meter_query": {"meter_query": 0.7006, "arc": 0.595, "kl": 0.01849},
    },
    "Standard_Single_Grad_LoRA": {
        "energy": {"energy": 0.7056, "arc": 0.655, "kl": 0.00575},
        "industrial": {"industrial": 0.7106, "arc": 0.67, "kl": 0.00409},
        "meter_query": {"meter_query": 0.565, "arc": 0.68, "kl": 0.00469},
    },
}

t0_cov = time.time()
lb.reset()
wiki, arc = protect_pool(tok)
cov_gen_cpu = collect_input_cov_cpu(wiki, last_only=False, weight=1.0)
cov_gen_cpu = collect_input_cov_cpu(arc, last_only=True, weight=8.0, init_cov=cov_gen_cpu)
print(f"Collected general protection covariance on CPU in {time.time() - t0_cov:.1f}s", flush=True)

task_covs_cpu = {}
for t in TASKS:
    lb.reset()
    task_covs_cpu[t] = collect_task_cov_cpu(t, n_max=80)
print("Collected task covariances on CPU for energy, industrial, meter_query", flush=True)


def build_nullspace_projectors(protect_cov_cpu, mu_null=3.0, hard_energy=0.80):
    """
    Dual Nullspace Protection:
    1) Soft ridge nullspace inverse Pi = (I + mu_null * C / mean_diag(C))^{-1} for preconditioning gradients,
    2) Top hard_energy orthonormal eigenvectors U of C for post-step cumulative delta clamping!
    """
    Pi = {}
    U_map = {}
    for key, C_cpu in protect_cov_cpu.items():
        c = C_cpu.to(model.device)
        d = c.shape[0]
        reg = torch.eye(d, device=c.device) + mu_null * (c / (c.trace() / d + 1e-8))
        Pi[key] = torch.linalg.inv(reg)
        val, vec = torch.linalg.eigh(c)
        val, vec = val.flip(0), vec.flip(1)
        r = int((val.cumsum(0) / val.sum() < hard_energy).sum()) + 1
        U_map[key] = vec[:, :r].contiguous()
    torch.cuda.empty_cache()
    return Pi, U_map


def compile_dataset_wsp(task_name, Pi_gpu, U_gpu=None, epochs=5, lr=None, reset_first=True):
    t0 = time.time()
    lr = lr or LRS[task_name]
    if reset_first:
        lb.reset()
    # Ensure shadow model does not sit in GPU VRAM during compilation
    if hasattr(lb, "_shadow"):
        lb._shadow = None
    torch.cuda.empty_cache()

    w0_map = {}
    if U_gpu is not None:
        for l, lay in enumerate(model.model.layers):
            for mname in mod_to_rname.keys():
                w0_map[(l, mname)] = lay.get_submodule(mname).weight.detach().clone()

    with scene(task_name):
        rs = label_grad.rows(tok, task_cases(task_name, "practice_a"))
    k = int((rs[0][0] == tok.convert_tokens_to_ids("<|im_start|>")).nonzero()[1])
    for p in model.parameters():
        p.requires_grad_(True)
    model.model.embed_tokens.weight.requires_grad_(False)
    model.lm_head.weight.requires_grad_(False)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)
    steps = epochs * ((len(rs) + 7) // 8)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, steps // 10)) * max(0.0, 1 - s / steps)
    )
    random.seed(0)
    # Use micro-batch size 1 for long-prefix industrial, 2 for others, with bfloat16 autocast!
    mbs = 1 if task_name == "industrial" else 2
    for ep in range(epochs):
        random.shuffle(rs)
        for i in range(0, len(rs), 8):
            batch = rs[i : i + 8]
            tot = sum(len(g) for _, g in batch)
            for j in range(0, len(batch), mbs):
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    loss, _ = batch_forward(model, batch[j : j + mbs], k, True, False)
                (loss / tot).backward()
            if Pi_gpu is not None:
                with torch.no_grad():
                    for l, lay in enumerate(model.model.layers):
                        for mname, rname in mod_to_rname.items():
                            w = lay.get_submodule(mname).weight
                            if w.grad is not None:
                                w.grad.copy_(w.grad @ Pi_gpu[(l, rname)])
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            if U_gpu is not None:
                with torch.no_grad():
                    for l, lay in enumerate(model.model.layers):
                        for mname, rname in mod_to_rname.items():
                            U = U_gpu[(l, rname)]
                            w = lay.get_submodule(mname).weight
                            w0 = w0_map[(l, mname)]
                            d = w.data - w0
                            w.data.copy_(w0 + d - 0.5 * ((d @ U) @ U.T))
        print(f"    [{task_name}] ep {ep+1}/{epochs} ({time.time()-t0:.1f}s)", flush=True)
    for p in model.parameters():
        p.requires_grad_(False)
    w0_map.clear()
    torch.cuda.empty_cache()
    return round(time.time() - t0, 2)


# 1. Single-Dataset Nullspace WSP Compilation across all 3 tasks
Pi_gen, U_gen = build_nullspace_projectors(cov_gen_cpu, mu_null=3.0, hard_energy=0.80)
wsp_single = {}
wsp_deltas = {}
wsp_saved_files = {}
for t in TASKS:
    sec = compile_dataset_wsp(t, Pi_gen, U_gen, epochs=5, reset_first=True)
    wsp_single[t] = eval_all(f"Nullspace_WSP_Single_{t}", [t], {"compile_sec": sec})
    deltas_t = {}
    save_dict = {}
    for l, lay in enumerate(model.model.layers):
        for mname in mod_to_rname.keys():
            pname = f"model.layers.{l}.{mname}.weight"
            d_cpu = (lay.get_submodule(mname).weight.detach().float().cpu() - lb.base[pname].float().cpu()).clone()
            deltas_t[(l, mname)] = d_cpu
            save_dict[pname] = d_cpu.to(torch.bfloat16)
    wsp_deltas[t] = deltas_t
    pt_path = L.ROOT / "data" / "edits" / f"wsp_nullspace_{t}.pt"
    torch.save(save_dict, pt_path)
    wsp_saved_files[t] = f"wsp_nullspace_{t}.pt"

Pi_gen.clear()
U_gen.clear()
torch.cuda.empty_cache()

results["Nullspace_WSP_Single"] = wsp_single
Path("/tmp/universal_wsp_vs_sft_lora.json").write_text(json.dumps(results, indent=2))


# 2. Functional-SVD Compiled LoRA Export (Rank 16 and Rank 32) across all 3 tasks
def apply_functional_svd_lora(task_name, rank=32):
    lb.reset()
    t0 = time.time()
    deltas_t = wsp_deltas[task_name]
    cov_t = task_covs_cpu[task_name]
    with torch.no_grad():
        for l, lay in enumerate(model.model.layers):
            for rname in reps_map.keys():
                C = cov_t[(l, rname)].to(model.device)
                d = C.shape[0]
                C_reg = C / (C.trace() / d + 1e-8) + 0.01 * torch.eye(d, device=C.device)
                evals, evecs = torch.linalg.eigh(C_reg)
                evals = evals.clamp(min=1e-6)
                C_half = evecs @ torch.diag(evals.sqrt()) @ evecs.T
                C_inv_half = evecs @ torch.diag(1.0 / evals.sqrt()) @ evecs.T
                for mname, rn in mod_to_rname.items():
                    if rn != rname:
                        continue
                    dW = deltas_t[(l, mname)].to(model.device)
                    U, S, Vh = torch.linalg.svd(dW @ C_half, full_matrices=False)
                    B_lora = U[:, :rank] * (S[:rank].sqrt().unsqueeze(0))
                    A_lora = (S[:rank].sqrt().unsqueeze(1)) * (Vh[:rank, :] @ C_inv_half)
                    w = lay.get_submodule(mname).weight
                    w.add_((B_lora @ A_lora).to(w.dtype))
    torch.cuda.empty_cache()
    return round(time.time() - t0, 2)


results["Functional_SVD_Compiled_LoRA"] = {}
for r_val in (16, 32):
    results["Functional_SVD_Compiled_LoRA"][f"rank_{r_val}"] = {}
    for t in TASKS:
        sec_svd = apply_functional_svd_lora(t, rank=r_val)
        results["Functional_SVD_Compiled_LoRA"][f"rank_{r_val}"][t] = eval_all(
            f"Functional_SVD_LoRA_r{r_val}_{t}", [t], {"svd_sec": sec_svd}
        )
Path("/tmp/universal_wsp_vs_sft_lora.json").write_text(json.dumps(results, indent=2))

# 3. Multi-Dataset Fusion (energy + industrial + meter_query = 4,800 Unseen Test Dialogs!)
lb.reset(*[FT[t] for t in TASKS])
results["Task_Arithmetic_SUM"] = eval_all("Task_Arithmetic_SUM")


def make_wsp_teacher(task_name):
    m = copy.deepcopy(lb.model)
    for n, p in m.named_parameters():
        p.data.copy_(lb.base[n])
    L.apply_edit(m, L.ROOT / "data" / "edits" / wsp_saved_files[task_name])
    return m


if hasattr(lb, "_shadow"):
    lb._shadow = None
torch.cuda.empty_cache()
groups = [(make_wsp_teacher(t),) + scene_rows(tok, t, task_cases(t, "practice_a"), "all") for t in TASKS]
lb.reset()
sec_align = align_write(tok, model, groups, mu=1.0)
groups.clear()
torch.cuda.empty_cache()
results["ClosedForm_MultiDataset_Aligned_WSP"] = eval_all(
    "ClosedForm_MultiDataset_Aligned_WSP", extra={"align_sec": round(sec_align, 2)}
)
Path("/tmp/universal_wsp_vs_sft_lora.json").write_text(json.dumps(results, indent=2))

# 4. Continual Sequential Learning (energy -> industrial -> meter_query)
lb.reset(FT["energy"])
for t in ("industrial", "meter_query"):
    compile_dataset_wsp(t, None, None, epochs=4, reset_first=False)
results["Standard_Continual_SFT_1to2to3"] = eval_all("Standard_Continual_SFT_1to2to3")

lb.reset(wsp_saved_files["energy"])
accum_prot_cpu = {k: v.clone() for k, v in cov_gen_cpu.items()}
for k in accum_prot_cpu:
    c_gen = accum_prot_cpu[k]
    c_t1 = task_covs_cpu["energy"][k]
    accum_prot_cpu[k] = c_gen + 4.0 * (c_gen.trace() / (c_t1.trace() + 1e-8)) * c_t1

for t in ("industrial", "meter_query"):
    Pi_c, U_c = build_nullspace_projectors(accum_prot_cpu, mu_null=5.0, hard_energy=0.85)
    compile_dataset_wsp(t, Pi_c, U_c, epochs=4, reset_first=False)
    Pi_c.clear()
    U_c.clear()
    for k in accum_prot_cpu:
        c_gen = cov_gen_cpu[k]
        c_tk = task_covs_cpu[t][k]
        accum_prot_cpu[k] = accum_prot_cpu[k] + 4.0 * (c_gen.trace() / (c_tk.trace() + 1e-8)) * c_tk

results["Recursive_Nullspace_Continual_WSP_1to2to3"] = eval_all("Recursive_Nullspace_Continual_WSP_1to2to3")

Path("/tmp/universal_wsp_vs_sft_lora.json").write_text(json.dumps(results, indent=2))
print("=== ALL 4 PARTS COMPLETE ===", flush=True)
