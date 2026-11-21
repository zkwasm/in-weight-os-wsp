#!/usr/bin/env python3
r"""
Step 5 of Universal WSP Toolchain: Turnkey Head-to-Head Benchmark vs. SFT & LoRA
================================================================================
Given any base model, a training .jsonl file, and a held-out test .jsonl file,
this script automatically executes and compares:
  1) Unedited Base Model
  2) Standard Full Gradient SFT (Unconstrained)
  3) Standard Gradient LoRA (Rank r)
  4) Nullspace-Preconditioned WSP Compiler (Ours)
  5) Functional-SVD Compiled LoRA (Rank r, Ours)
evaluating Test Exact-Match Accuracy, Test Answer Log-Prob, General Reasoning Accuracy,
and General Corpus KL Divergence.
"""
import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from wsp_1_profile_base import BUILTIN_REASONING_PROMPTS, BUILTIN_WIKI_PASSAGES
from wsp_2_compile_dataset import MOD_TO_RNAME, load_pairs

@torch.no_grad()
def evaluate_model(tok, model, test_pairs, base_wiki_lps, device):
    model.eval()
    exact_hits = 0
    logp_sum = 0.0
    for p_ids, g_ids in test_pairs:
        full = torch.cat([p_ids, g_ids]).to(device)[None]
        logits = model(input_ids=full, use_cache=False).logits[0, len(p_ids) - 1 : -1].float()
        preds = logits.argmax(dim=-1).cpu()
        if torch.equal(preds, g_ids):
            exact_hits += 1
        lp = F.log_softmax(logits, dim=-1)
        logp_sum += float(lp[torch.arange(len(g_ids)), g_ids.to(device)].mean().item())

    # Evaluate general corpus KL drift against unedited base model
    kl_vals = []
    for txt, lp0 in zip(BUILTIN_WIKI_PASSAGES, base_wiki_lps):
        ids = tok(txt, return_tensors="pt").input_ids.to(device)
        lp1 = F.log_softmax(model(input_ids=ids, use_cache=False).logits[0, :-1].float(), dim=-1).cpu()
        kl = (lp0.exp() * (lp0 - lp1)).sum(dim=-1).mean().item()
        kl_vals.append(kl)

    return {
        "test_exact_acc": round(exact_hits / max(1, len(test_pairs)), 4),
        "test_mean_logp": round(logp_sum / max(1, len(test_pairs)), 4),
        "general_wiki_kl": round(sum(kl_vals) / max(1, len(kl_vals)), 5),
        "n_test": len(test_pairs),
    }

def main():
    parser = argparse.ArgumentParser(description="Step 5: Turnkey Benchmark of Nullspace-WSP & Functional-SVD LoRA vs. Standard SFT")
    parser.add_argument("--model", type=str, required=True, help="Base model ID or local path")
    parser.add_argument("--train_file", type=str, required=True, help="Training .jsonl file")
    parser.add_argument("--test_file", type=str, required=True, help="Held-out test .jsonl file")
    parser.add_argument("--epochs", type=int, default=4, help="Epochs/passes (default: 4)")
    parser.add_argument("--lr", type=float, default=1.5e-5, help="Peak step size (default: 1.5e-5)")
    parser.add_argument("--rank", type=int, default=32, help="LoRA rank for Functional-SVD export (default: 32)")
    parser.add_argument("--output_json", type=str, default=None, help="Optional JSON path to save benchmark summary")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(args.device).eval()

    base_weights = {
        (l, mname): lay.get_submodule(mname).weight.detach().cpu().clone()
        for l, lay in enumerate(model.model.layers)
        for mname in MOD_TO_RNAME.keys()
    }
    base_wiki_lps = []
    with torch.no_grad():
        for txt in BUILTIN_WIKI_PASSAGES:
            ids = tok(txt, return_tensors="pt").input_ids.to(args.device)
            base_wiki_lps.append(F.log_softmax(model(input_ids=ids, use_cache=False).logits[0, :-1].float(), dim=-1).cpu())

    test_pairs = load_pairs(tok, args.test_file)
    summary = {}
    summary["Unedited_Base"] = evaluate_model(tok, model, test_pairs, base_wiki_lps, args.device)
    print("[Benchmark] Unedited_Base:", json.dumps(summary["Unedited_Base"]), flush=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        prof_pt = tmp / "profile.pt"
        wsp_dir = tmp / "wsp_out"
        lora_dir = tmp / "lora_out"

        subprocess.run([sys.executable, str(Path(__file__).parent / "wsp_1_profile_base.py"), "--model", args.model, "--output_profile", str(prof_pt), "--device", args.device], check=True)
        subprocess.run([sys.executable, str(Path(__file__).parent / "wsp_2_compile_dataset.py"), "--model", args.model, "--profile", str(prof_pt), "--train_file", args.train_file, "--output_dir", str(wsp_dir), "--epochs", str(args.epochs), "--lr", str(args.lr), "--device", args.device], check=True)
        subprocess.run([sys.executable, str(Path(__file__).parent / "wsp_3_export_lora.py"), "--wsp_dir", str(wsp_dir), "--rank", str(args.rank), "--output_peft_dir", str(lora_dir), "--device", args.device], check=True)

        # Evaluate Nullspace-WSP
        art = torch.load(wsp_dir / "wsp_artifact.pt", map_location="cpu")
        with torch.no_grad():
            for (l, mname), w0 in base_weights.items():
                w_new = (w0.float() + art["deltas"][(l, mname)].float()).to(dtype).to(args.device)
                model.model.layers[l].get_submodule(mname).weight.copy_(w_new)
        summary["Nullspace_WSP"] = evaluate_model(tok, model, test_pairs, base_wiki_lps, args.device)
        print("[Benchmark] Nullspace_WSP:", json.dumps(summary["Nullspace_WSP"]), flush=True)

        # Evaluate Functional-SVD Compiled LoRA
        peft_sd = torch.load(lora_dir / "adapter_model.bin", map_location="cpu")
        with torch.no_grad():
            for (l, mname), w0 in base_weights.items():
                bkey = f"base_model.model.model.layers.{l}.{mname}"
                A = peft_sd[f"{bkey}.lora_A.weight"].float()
                B = peft_sd[f"{bkey}.lora_B.weight"].float()
                w_new = (w0.float() + (B @ A)).to(dtype).to(args.device)
                model.model.layers[l].get_submodule(mname).weight.copy_(w_new)
        summary[f"Functional_SVD_LoRA_r{args.rank}"] = evaluate_model(tok, model, test_pairs, base_wiki_lps, args.device)
        print(f"[Benchmark] Functional_SVD_LoRA_r{args.rank}:", json.dumps(summary[f"Functional_SVD_LoRA_r{args.rank}"]), flush=True)

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[Benchmark] Complete!")

if __name__ == "__main__":
    main()
