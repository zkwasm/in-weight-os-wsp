#!/usr/bin/env python3
r"""
Step 2 of Universal WSP Toolchain: Nullspace-Guided Dataset-to-Weight Compilation
=================================================================================
Compiles any supervised dataset (.jsonl with {"prompt": ..., "response": ...},
{"system": ..., "prompt": ..., "response": ...}, or {"messages": [...]}) into a
pretrained base model while constraining parameter updates via Dual Nullspace Navigation:
  1) Right-multiplying parameter gradients by the Closed-Form Nullspace Projector:
       \widetilde{G} = G @ (I + \mu \tilde{C}_{protect})^{-1}
  2) Clamping cumulative weight deltas out of the top eigenspace U_{protect}:
       \Delta W <- \Delta W - 0.5 * (\Delta W @ U_{protect}) @ U_{protect}^T
Supports replay-free continual learning (--continual_from) by accumulating prior task
input covariances into C_{protect}!
"""
import argparse
import json
import random
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MOD_TO_RNAME = {
    "self_attn.q_proj": "qkv",
    "self_attn.k_proj": "qkv",
    "self_attn.v_proj": "qkv",
    "self_attn.o_proj": "o",
    "mlp.gate_proj": "gu",
    "mlp.up_proj": "gu",
    "mlp.down_proj": "down",
}

def load_pairs(tok, jsonl_path):
    pairs = []
    for line in Path(jsonl_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if "messages" in item:
            msgs = item["messages"]
            p_txt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
            g_txt = msgs[-1]["content"]
        else:
            msgs = []
            if "system" in item and item["system"]:
                msgs.append({"role": "system", "content": item["system"]})
            msgs.append({"role": "user", "content": item["prompt"]})
            p_txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            g_txt = item["response"]
        p_ids = tok.encode(p_txt, add_special_tokens=False)
        g_ids = tok.encode(g_txt, add_special_tokens=False) + [tok.eos_token_id]
        pairs.append((torch.tensor(p_ids, dtype=torch.long), torch.tensor(g_ids, dtype=torch.long)))
    return pairs

def main():
    parser = argparse.ArgumentParser(description="Step 2: Compile a Supervised JSONL Dataset via Nullspace WSP")
    parser.add_argument("--model", type=str, required=True, help="Base model ID or local path")
    parser.add_argument("--profile", type=str, required=True, help="Path to Step 1 nullspace profile .pt")
    parser.add_argument("--train_file", type=str, required=True, help="Path to training .jsonl file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for compiled WSP artifact")
    parser.add_argument("--save_hf_model", action="store_true", help="Also save full Hugging Face model in output_dir/hf_model")
    parser.add_argument("--continual_from", type=str, default=None, help="Optional previous task artifact dir for continual learning")
    parser.add_argument("--epochs", type=int, default=5, help="Number of nullspace compilation passes (default: 5)")
    parser.add_argument("--batch_size", type=int, default=8, help="Effective batch size (default: 8)")
    parser.add_argument("--lr", type=float, default=1.5e-5, help="Peak step size (default: 1.5e-5)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    t0 = time.time()
    prof = torch.load(args.profile, map_location="cpu")
    cov_protect = {k: v.clone() for k, v in prof["cov_gen"].items()}

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(args.device)
    base_weights = {
        (l, mname): lay.get_submodule(mname).weight.detach().cpu().clone()
        for l, lay in enumerate(model.model.layers)
        for mname in MOD_TO_RNAME.keys()
    }

    if args.continual_from:
        prev_art = torch.load(Path(args.continual_from) / "wsp_artifact.pt", map_location="cpu")
        with torch.no_grad():
            for (l, mname), dW in prev_art["deltas"].items():
                model.model.layers[l].get_submodule(mname).weight.add_(dW.to(args.device).float())
        for k, c_prev in prev_art["cov_task"].items():
            c_g = cov_protect[k]
            cov_protect[k] = c_g + 4.0 * (c_g.trace() / (c_prev.trace() + 1e-8)) * c_prev
        print(f"[Step 2] Loaded continual state & accumulated task covariance from {args.continual_from}")

    Pi_gpu, U_gpu = {}, {}
    for key, C_cpu in cov_protect.items():
        c = C_cpu.to(args.device)
        d = c.shape[0]
        c_reg = c + 1e-3 * (c.trace() / d + 1e-8) * torch.eye(d, device=c.device)
        reg = torch.eye(d, device=c.device) + prof["mu_null"] * (c_reg / (c_reg.trace() / d + 1e-8))
        Pi_gpu[key] = torch.linalg.inv(reg)
        val, vec = torch.linalg.eigh(c_reg)
        val, vec = val.flip(0), vec.flip(1)
        r = int((val.cumsum(0) / val.sum() < prof["hard_energy"]).sum()) + 1
        U_gpu[key] = vec[:, :r].contiguous()

    w0_gpu = {
        (l, mname): lay.get_submodule(mname).weight.detach().clone()
        for l, lay in enumerate(model.model.layers)
        for mname in MOD_TO_RNAME.keys()
    }

    pairs = load_pairs(tok, args.train_file)
    for p in model.parameters():
        p.requires_grad_(True)
    model.model.embed_tokens.weight.requires_grad_(False)
    model.lm_head.weight.requires_grad_(False)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
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
            for p_ids, g_ids in batch:
                full = torch.cat([p_ids, g_ids]).to(args.device)[None]
                if use_amp:
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        logits = model(input_ids=full, use_cache=False).logits[0, len(p_ids) - 1 : -1].float()
                        loss = F.cross_entropy(logits, g_ids.to(args.device), reduction="sum") / tot_toks
                else:
                    logits = model(input_ids=full, use_cache=False).logits[0, len(p_ids) - 1 : -1].float()
                    loss = F.cross_entropy(logits, g_ids.to(args.device), reduction="sum") / tot_toks
                loss.backward()
            with torch.no_grad():
                for l, lay in enumerate(model.model.layers):
                    for mname, rname in MOD_TO_RNAME.items():
                        w = lay.get_submodule(mname).weight
                        if w.grad is not None:
                            w.grad.copy_(w.grad @ Pi_gpu[(l, rname)])
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                for l, lay in enumerate(model.model.layers):
                    for mname, rname in MOD_TO_RNAME.items():
                        U = U_gpu[(l, rname)]
                        w = lay.get_submodule(mname).weight
                        w0 = w0_gpu[(l, mname)]
                        d = w.data - w0
                        w.data.copy_(w0 + d - 0.5 * ((d @ U) @ U.T))
        print(f"[Step 2] Epoch {ep+1}/{args.epochs} complete ({time.time()-t0:.1f}s)")

    for p in model.parameters():
        p.requires_grad_(False)

    reps_map = {
        "qkv": lambda lay: lay.self_attn.q_proj,
        "o": lambda lay: lay.self_attn.o_proj,
        "gu": lambda lay: lay.mlp.gate_proj,
        "down": lambda lay: lay.mlp.down_proj,
    }
    cov_task = {}
    hs = []
    for l, lay in enumerate(model.model.layers):
        for rname, getter in reps_map.items():
            def make_hk(key):
                def f(_, inp):
                    x = inp[0][0].detach().float()
                    c = (x.T @ x).cpu()
                    cov_task[key] = cov_task[key] + c if key in cov_task else c
                return f
            hs.append(getter(lay).register_forward_pre_hook(make_hk((l, rname))))

    with torch.no_grad():
        for p_ids, g_ids in pairs[:80]:
            full = torch.cat([p_ids, g_ids]).to(args.device)[None]
            model(input_ids=full, use_cache=False)
    for h in hs:
        h.remove()

    deltas = {
        (l, mname): (lay.get_submodule(mname).weight.detach().cpu() - base_weights[(l, mname)]).to(torch.bfloat16)
        for l, lay in enumerate(model.model.layers)
        for mname in MOD_TO_RNAME.keys()
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": args.model,
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
