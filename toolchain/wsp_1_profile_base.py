#!/usr/bin/env python3
r"""
Step 1 of Universal WSP Toolchain: Base Model General-Capability Nullspace Profiling
====================================================================================
Automatically inspects ANY Hugging Face CausalLM architecture (`discover_architecture_groups`),
runs a read-only forward pass over the full-rank general protection pool:
  - `data/wikitext_protect_60.jsonl` (60 long WikiText passages, ~12,000 tokens, weight=1.0)
  - `data/arc_easy_protect_300.jsonl` (300 ARC-Easy questions, decision token weight=8.0)
and precomputes per shared input group `(layer_idx, group_id)`:
  1) Soft Ridge Nullspace Projector: Pi_gen = (I + mu_null * C / (tr(C)/d))^{-1}
  2) Orthonormal Top Eigenspace Basis: U_gen covering `hard_energy` (default 0.80)
"""
import argparse
import json
import time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from wsp_0_common import (
    REPO_DATA_DIR,
    discover_architecture_groups,
    format_messages_safe,
    get_transformer_layers,
)

def main():
    parser = argparse.ArgumentParser(description="Step 1: Profile Base Model General-Capability Nullspace")
    parser.add_argument("--model", type=str, required=True, help="Hugging Face model ID or local path")
    parser.add_argument("--output_profile", type=str, required=True, help="Output .pt path for nullspace profile")
    parser.add_argument("--wiki_jsonl", type=str, default=str(REPO_DATA_DIR / "wikitext_protect_60.jsonl"))
    parser.add_argument("--arc_jsonl", type=str, default=str(REPO_DATA_DIR / "arc_easy_protect_300.jsonl"))
    parser.add_argument("--mu_null", type=float, default=6.0, help="Ridge nullspace penalty weight mu (default: 6.0)")
    parser.add_argument("--hard_energy", type=float, default=0.86, help="Top spectral energy fraction for U_gen (default: 0.86)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    t0 = time.time()
    print(f"[Step 1] Loading base model: {args.model} on {args.device}...")
    tok = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(args.device).eval()

    arch_info = discover_architecture_groups(model, tok, args.device)
    _, layers = get_transformer_layers(model)
    group_to_rep = arch_info["group_to_rep"]
    print(f"[Step 1] Auto-discovered {len(layers)} layers x {len(group_to_rep)} shared input groups per layer: {list(group_to_rep.keys())}")

    wiki_texts = [json.loads(l)["text"] for l in Path(args.wiki_jsonl).read_text(encoding="utf-8").splitlines() if l.strip()]
    arc_items = [json.loads(l) for l in Path(args.arc_jsonl).read_text(encoding="utf-8").splitlines() if l.strip()]

    cov_gpu = {}
    state = {"last_only": False, "weight": 1.0}
    hs = []
    for l, lay in enumerate(layers):
        for gid, rep_mod_name in group_to_rep.items():
            mod = lay.get_submodule(rep_mod_name)
            def make_hk(key):
                def f(_, inp):
                    x = inp[0][0].detach().float()
                    if state["last_only"]:
                        x = x[-16:]
                    c = state["weight"] * (x.T @ x)
                    cov_gpu[key] = cov_gpu[key] + c if key in cov_gpu else c
                return f
            hs.append(mod.register_forward_pre_hook(make_hk((l, gid))))

    with torch.no_grad():
        state.update(last_only=False, weight=1.0)
        for txt in wiki_texts:
            ids = tok(txt, return_tensors="pt", truncation=True, max_length=256).input_ids.to(args.device)
            model(input_ids=ids, use_cache=False)

        state.update(last_only=True, weight=8.0)
        for item in arc_items:
            p_txt = format_messages_safe(tok, [{"role": "user", "content": item["prompt"]}], add_generation_prompt=True)
            ids = tok(p_txt, return_tensors="pt").input_ids.to(args.device)
            model(input_ids=ids, use_cache=False)

    for h in hs:
        h.remove()

    cov_cpu, Pi_cpu, U_cpu = {}, {}, {}
    for key, c in cov_gpu.items():
        d = c.shape[0]
        c_reg = c + 1e-4 * (c.trace() / d + 1e-8) * torch.eye(d, device=c.device)
        reg = torch.eye(d, device=c.device) + args.mu_null * (c_reg / (c_reg.trace() / d + 1e-8))
        Pi_cpu[key] = torch.linalg.inv(reg).cpu()
        val, vec = torch.linalg.eigh(c_reg)
        val, vec = val.flip(0), vec.flip(1)
        r = int((val.cumsum(0) / val.sum() < args.hard_energy).sum()) + 1
        U_cpu[key] = vec[:, :r].contiguous().cpu()
        cov_cpu[key] = c.cpu()

    out_path = Path(args.output_profile)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": args.model,
        "arch_info": arch_info,
        "mu_null": args.mu_null,
        "hard_energy": args.hard_energy,
        "cov_gen": cov_cpu,
        "Pi_gen": Pi_cpu,
        "U_gen": U_cpu,
    }, out_path)
    print(f"[Step 1] Saved full-rank nullspace profile ({len(Pi_cpu)} group matrices) to {out_path} in {time.time()-t0:.1f}s.")

if __name__ == "__main__":
    main()
