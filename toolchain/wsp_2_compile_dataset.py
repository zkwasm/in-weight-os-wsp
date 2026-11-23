#!/usr/bin/env python3
r"""
Step 2 of Universal WSP Toolchain: Nullspace-Guided Dataset-to-Weight Compilation
=================================================================================
Supports ANY SFT task type, ANY of the 5 standard SFT dataset formats (Messages,
Alpaca, ShareGPT, QA, or Raw Text), and ANY Hugging Face CausalLM architecture!
Automatically detects shared prompt prefixes (if any) for 3x faster KV-cached passes,
scales step size automatically by model hidden dimension if `--lr auto`, and constrains
updates via Dual Nullspace Navigation (Pi_gen + U_gen).
"""
import argparse
import random
import time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from wsp_0_common import (
    detect_shared_prefix_len,
    discover_architecture_groups,
    forward_sft_subbatch,
    get_transformer_layers,
    load_universal_sft_dataset,
)

def main():
    parser = argparse.ArgumentParser(description="Step 2: Compile Any SFT Dataset via Dual Nullspace WSP")
    parser.add_argument("--model", type=str, required=True, help="Base model ID or local path")
    parser.add_argument("--profile", type=str, required=True, help="Path to Step 1 nullspace profile .pt")
    parser.add_argument("--train_file", type=str, required=True, help="Path to training .jsonl/.json file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for compiled WSP artifact")
    parser.add_argument("--system_prompt", type=str, default=None, help="Optional system prompt override")
    parser.add_argument("--save_hf_model", action="store_true", help="Also save full Hugging Face model in output_dir/hf_model")
    parser.add_argument("--continual_from", type=str, default=None, help="Optional previous task artifact dir for continual learning")
    parser.add_argument("--epochs", type=int, default=5, help="Number of nullspace compilation passes (default: 5)")
    parser.add_argument("--batch_size", type=int, default=8, help="Effective batch size (default: 8)")
    parser.add_argument("--lr", type=str, default="auto", help="Peak step size or 'auto' (1.5e-5 for <=1.5B, 5e-6 for >=7B)")
    parser.add_argument("--rank_cap", type=int, default=32, help="Target spectral rank cap r for lossless LoRA export (default: 32)")
    parser.add_argument("--disable_nullspace", action="store_true", help="Run unconstrained Standard Full SFT (for baseline comparison)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    t0 = time.time()
    prof = torch.load(args.profile, map_location="cpu")
    cov_protect = {k: v.clone() for k, v in prof["cov_gen"].items()}

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(args.device)
    arch_info = prof.get("arch_info") or discover_architecture_groups(model, tok, args.device)
    _, layers = get_transformer_layers(model)
    mod_to_group = arch_info["mod_to_group"]
    group_to_rep = arch_info["group_to_rep"]

    d_model = getattr(model.config, "hidden_size", 1024)
    base_lr = (1.5e-5 if d_model <= 2048 else 5.0e-6) if args.lr == "auto" else float(args.lr)
    # Compensate step size by 1.30x when projecting through Pi_gpu so effective task step matches unconstrained SFT
    lr = base_lr if args.disable_nullspace else (base_lr * 1.30)

    base_weights = {
        (l, mname): lay.get_submodule(mname).weight.detach().cpu().clone()
        for l, lay in enumerate(layers)
        for mname in mod_to_group.keys()
    }

    if args.continual_from:
        prev_art = torch.load(Path(args.continual_from) / "wsp_artifact.pt", map_location="cpu")
        with torch.no_grad():
            for (l, mname), dW in prev_art["deltas"].items():
                layers[l].get_submodule(mname).weight.add_(dW.to(args.device).float())
        for k, c_prev in prev_art["cov_task"].items():
            c_g = cov_protect[k]
            cov_protect[k] = c_g + 4.0 * (c_g.trace() / (c_prev.trace() + 1e-8)) * c_prev
        print(f"[Step 2] Loaded continual checkpoint & accumulated prior task covariance from {args.continual_from}")

    Pi_gpu, U_gpu = {}, {}
    if not args.disable_nullspace:
        for key, C_cpu in cov_protect.items():
            c = C_cpu.to(args.device)
            d = c.shape[0]
            c_reg = c + 1e-4 * (c.trace() / d + 1e-8) * torch.eye(d, device=c.device)
            reg = torch.eye(d, device=c.device) + prof["mu_null"] * (c_reg / (c_reg.trace() / d + 1e-8))
            Pi_gpu[key] = torch.linalg.inv(reg)
            val, vec = torch.linalg.eigh(c_reg)
            val, vec = val.flip(0), vec.flip(1)
            r = int((val.cumsum(0) / val.sum() < prof["hard_energy"]).sum()) + 1
            U_gpu[key] = vec[:, :r].contiguous()

    w0_gpu = {
        (l, mname): lay.get_submodule(mname).weight.detach().clone()
        for l, lay in enumerate(layers)
        for mname in mod_to_group.keys()
    }

    pairs = load_universal_sft_dataset(tok, args.train_file, system_override=args.system_prompt)
    shared_k = detect_shared_prefix_len(pairs)
    max_seq_len = max(len(p) + len(g) for p, g in pairs)
    mbs = 1 if (max_seq_len > 900 or d_model > 2048) else 2
    print(f"[Step 2] Loaded {len(pairs)} training examples (shared_prefix_tokens={shared_k}, max_len={max_seq_len}, lr={lr})")

    for p in model.parameters():
        p.requires_grad_(False)
    trainable = []
    for l, lay in enumerate(layers):
        for mname in mod_to_group.keys():
            w = lay.get_submodule(mname).weight
            w.requires_grad_(True)
            trainable.append(w)

    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)
    bs = max(1, min(args.batch_size, len(pairs)))
    total_steps = max(1, args.epochs * ((len(pairs) + bs - 1) // bs))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, total_steps // 10)) * max(0.0, 1.0 - s / total_steps)
    )

    random.seed(0)
    use_amp = (args.device == "cuda")
    for ep in range(args.epochs):
        random.shuffle(pairs)
        for i in range(0, len(pairs), bs):
            batch = pairs[i : i + bs]
            tot_toks = sum(len(g) for _, g in batch)
            for j in range(0, len(batch), mbs):
                sub = batch[j : j + mbs]
                if use_amp:
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        loss = forward_sft_subbatch(model, tok, sub, shared_k=shared_k) / tot_toks
                else:
                    loss = forward_sft_subbatch(model, tok, sub, shared_k=shared_k) / tot_toks
                loss.backward()
            if not args.disable_nullspace:
                with torch.no_grad():
                    for l, lay in enumerate(layers):
                        for mname, gid in mod_to_group.items():
                            w = lay.get_submodule(mname).weight
                            if w.grad is not None:
                                w.grad.copy_(w.grad @ Pi_gpu[(l, gid)])
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            if not args.disable_nullspace:
                with torch.no_grad():
                    for l, lay in enumerate(layers):
                        for mname, gid in mod_to_group.items():
                            U = U_gpu[(l, gid)]
                            w = lay.get_submodule(mname).weight
                            w0 = w0_gpu[(l, mname)]
                            d = w.data - w0
                            w.data.copy_(w0 + d - 0.85 * ((d @ U) @ U.T))
        if not args.disable_nullspace and args.rank_cap > 0 and (ep >= 1 or ep == args.epochs - 1):
            with torch.no_grad():
                for l, lay in enumerate(layers):
                    for mname, gid in mod_to_group.items():
                        U_g = U_gpu[(l, gid)]
                        w = lay.get_submodule(mname).weight
                        w0 = w0_gpu[(l, mname)]
                        d = w.data - w0
                        d = d - (d @ U_g) @ U_g.T
                        r_eff = min(args.rank_cap, d.shape[0], d.shape[1])
                        u_s, s_s, vh_s = torch.linalg.svd(d, full_matrices=False)
                        d_low = (u_s[:, :r_eff] * s_s[:r_eff].unsqueeze(0)) @ vh_s[:r_eff, :]
                        d_low = d_low - 0.90 * ((d_low @ U_g) @ U_g.T)
                        w.data.copy_(w0 + d_low)
        print(f"[Step 2] Epoch {ep+1}/{args.epochs} complete ({time.time()-t0:.1f}s)", flush=True)

    for p in model.parameters():
        p.requires_grad_(False)
    Pi_gpu.clear()
    U_gpu.clear()
    w0_gpu.clear()
    if args.device == "cuda":
        torch.cuda.empty_cache()

    # Collect Answer-Weighted Task Input Covariance C_task (80% answer prediction positions + 20% prompt context)
    cov_ans_gpu, cov_prm_gpu = {}, {}
    cur_p_len = {"val": 0}
    hs = []
    for l, lay in enumerate(layers):
        for gid, rep_mod_name in group_to_rep.items():
            mod = lay.get_submodule(rep_mod_name)
            def make_hk(key):
                def f(_, inp):
                    x = inp[0][0].detach().float()
                    plen = cur_p_len["val"]
                    x_prm = x[: max(1, plen - 1)]
                    x_ans = x[max(0, plen - 1) : -1]
                    cp = x_prm.T @ x_prm
                    ca = x_ans.T @ x_ans
                    cov_prm_gpu[key] = cov_prm_gpu[key] + cp if key in cov_prm_gpu else cp
                    cov_ans_gpu[key] = cov_ans_gpu[key] + ca if key in cov_ans_gpu else ca
                return f
            hs.append(mod.register_forward_pre_hook(make_hk((l, gid))))

    with torch.no_grad():
        for p_ids, g_ids in pairs[:80]:
            cur_p_len["val"] = len(p_ids)
            full = torch.cat([p_ids, g_ids]).to(args.device)[None]
            model(input_ids=full, use_cache=False)
    for h in hs:
        h.remove()

    cov_task = {}
    for key in cov_ans_gpu.keys():
        ca = cov_ans_gpu[key]
        cp = cov_prm_gpu[key]
        d = ca.shape[0]
        ca_n = ca / (ca.trace() / d + 1e-8)
        cp_n = cp / (cp.trace() / d + 1e-8)
        cov_task[key] = (0.80 * ca_n + 0.20 * cp_n).cpu()
    deltas = {
        (l, mname): (lay.get_submodule(mname).weight.detach().cpu() - base_weights[(l, mname)]).to(torch.bfloat16)
        for l, lay in enumerate(layers)
        for mname in mod_to_group.keys()
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": args.model,
        "arch_info": arch_info,
        "train_file": str(args.train_file),
        "deltas": deltas,
        "cov_task": cov_task,
    }, out_dir / "wsp_artifact.pt")

    if args.save_hf_model:
        hf_out = out_dir / "hf_model"
        hf_out.mkdir(parents=True, exist_ok=True)
        model.to(torch.bfloat16).save_pretrained(hf_out)
        tok.save_pretrained(hf_out)
        print(f"[Step 2] Saved full Hugging Face model to {hf_out}")

    print(f"[Step 2] Saved compiled WSP artifact & task covariance to {out_dir / 'wsp_artifact.pt'}")

if __name__ == "__main__":
    main()
