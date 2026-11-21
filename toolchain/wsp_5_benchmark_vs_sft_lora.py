#!/usr/bin/env python3
r"""
Step 5 of Universal WSP Toolchain: Complete 5-Way Benchmark vs. Standard SFT & LoRA
===================================================================================
Given ANY base model, ANY training .jsonl/.json file, and ANY held-out test .jsonl/.json file,
this script runs a complete head-to-head comparison of all 5 methods:
  1) Unedited_Base
  2) Standard_Full_SFT           (Unconstrained AdamW SFT)
  3) Standard_Grad_LoRA          (Standard gradient-trained rank-r LoRA)
  4) Nullspace_WSP               (Our Dual Nullspace Compiler)
  5) Functional_SVD_LoRA         (Our Functional-SVD compiled rank-r PEFT LoRA)
and reports universal metrics suitable for both structured and open-ended SFT tasks:
  - `test_exact_acc`: Sequence-level exact match on held-out test set
  - `test_token_acc`: Token-level accuracy on held-out test responses
  - `test_mean_logp` & `test_ppl`: Held-out test log-likelihood & perplexity
  - `arc_easy_acc`: 200-question ARC-Easy general reasoning accuracy (`data/arc_easy_eval_200.jsonl`)
  - `wiki_kl_drift`: 40-passage WikiText KL divergence from base model (`data/wikitext_eval_40.jsonl`)
"""
import argparse
import json
import math
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from wsp_0_common import (
    REPO_DATA_DIR,
    detect_shared_prefix_len,
    discover_architecture_groups,
    format_messages_safe,
    forward_sft_subbatch,
    get_transformer_layers,
    load_universal_sft_dataset,
)

@torch.no_grad()
def evaluate_suite(tok, model, test_pairs, shared_k_test, wiki_eval_ids, base_wiki_lps, arc_eval_items, device):
    model.eval()
    exact_hits = 0
    total_toks = 0
    correct_toks = 0
    logp_sum = 0.0
    nll_sum = 0.0

    end_id = tok.convert_tokens_to_ids("<|im_end|>") or tok.eos_token_id
    for p_ids, g_ids in test_pairs:
        full = torch.cat([p_ids, g_ids]).to(device)[None]
        logits = model(input_ids=full, use_cache=False).logits[0, len(p_ids) - 1 : -1].float()
        preds = logits.argmax(dim=-1).cpu()
        is_exact = torch.equal(preds, g_ids)
        if not is_exact:
            # Also check alternative special-token spelling if response starts with <tool_call>
            raw_resp = tok.decode(g_ids[:-1], skip_special_tokens=False)
            alt_ids = torch.tensor(tok.encode(raw_resp, add_special_tokens=False) + [end_id], dtype=torch.long)
            if not torch.equal(alt_ids, g_ids):
                full_alt = torch.cat([p_ids, alt_ids]).to(device)[None]
                logits_alt = model(input_ids=full_alt, use_cache=False).logits[0, len(p_ids) - 1 : -1].float()
                if torch.equal(logits_alt.argmax(dim=-1).cpu(), alt_ids):
                    is_exact = True
                    logits, g_ids, preds = logits_alt, alt_ids, alt_ids
        if is_exact:
            exact_hits += 1
        correct_toks += int((preds == g_ids).sum().item())
        total_toks += len(g_ids)
        lp = F.log_softmax(logits, dim=-1)
        tok_lps = lp[torch.arange(len(g_ids)), g_ids.to(device)]
        logp_sum += float(tok_lps.mean().item())
        nll_sum -= float(tok_lps.sum().item())

    # Evaluate WikiText KL drift
    kl_vals = []
    for ids, lp0 in zip(wiki_eval_ids, base_wiki_lps):
        lp1 = F.log_softmax(model(input_ids=ids, use_cache=False).logits[0, :-1].float(), dim=-1).cpu()
        kl = (lp0.exp() * (lp0 - lp1)).sum(dim=-1).mean().item()
        kl_vals.append(kl)

    # Evaluate 200-question ARC-Easy accuracy
    arc_correct = 0
    for item in arc_eval_items:
        logits = model(input_ids=item["ids"], use_cache=False).logits[0, -1]
        pred_idx = int(torch.argmax(logits[item["letter_ids"]]).item())
        if item["labels"][pred_idx] == item["answer"]:
            arc_correct += 1

    mean_nll = nll_sum / max(1, total_toks)
    return {
        "test_exact_acc": round(exact_hits / max(1, len(test_pairs)), 4),
        "test_token_acc": round(correct_toks / max(1, total_toks), 4),
        "test_mean_logp": round(logp_sum / max(1, len(test_pairs)), 4),
        "test_ppl": round(math.exp(min(20.0, mean_nll)), 4),
        "arc_easy_acc": round(arc_correct / max(1, len(arc_eval_items)), 4),
        "wiki_kl_drift": round(sum(kl_vals) / max(1, len(kl_vals)), 5),
        "n_test": len(test_pairs),
    }


def train_standard_grad_lora(model, tok, layers, mod_to_group, train_pairs, shared_k, rank, epochs, lr, bs, device):
    """Trains a standard Kaiming-initialized rank-r LoRA baseline using AdamW."""
    lora_A, lora_B = {}, {}
    orig_forwards = {}
    params = []
    scale = 1.0
    for l, lay in enumerate(layers):
        for mname in mod_to_group.keys():
            mod = lay.get_submodule(mname)
            d_out, d_in = mod.weight.shape
            r_eff = min(rank, d_out, d_in)
            A = nn.Parameter(torch.empty((r_eff, d_in), device=device, dtype=torch.float32))
            B = nn.Parameter(torch.zeros((d_out, r_eff), device=device, dtype=torch.float32))
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))
            lora_A[(l, mname)] = A
            lora_B[(l, mname)] = B
            params.extend([A, B])
            orig_f = mod.forward
            orig_forwards[(l, mname)] = (mod, orig_f)
            def make_lora_fwd(orig_fn, A_p, B_p):
                def fwd(x):
                    out = orig_fn(x)
                    lora_out = (x.float() @ A_p.T @ B_p.T).to(out.dtype) * scale
                    return out + lora_out
                return fwd
            mod.forward = make_lora_fwd(orig_f, A, B)

    for p in model.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(params, lr=lr * 10.0, weight_decay=0.0)
    total_steps = max(1, epochs * ((len(train_pairs) + bs - 1) // bs))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, total_steps // 10)) * max(0.0, 1.0 - s / total_steps)
    )
    random.seed(0)
    pairs = list(train_pairs)
    mbs = 1 if max(len(p) + len(g) for p, g in pairs) > 900 else 2
    for ep in range(epochs):
        random.shuffle(pairs)
        for i in range(0, len(pairs), bs):
            batch = pairs[i : i + bs]
            tot_toks = sum(len(g) for _, g in batch)
            for j in range(0, len(batch), mbs):
                sub = batch[j : j + mbs]
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                    loss = forward_sft_subbatch(model, tok, sub, shared_k=shared_k) / tot_toks
                loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
        print(f"[Grad-LoRA] Epoch {ep+1}/{epochs} complete", flush=True)

    # Fold LoRA into weights for evaluation and restore original forward methods
    with torch.no_grad():
        for key, (mod, orig_f) in orig_forwards.items():
            mod.forward = orig_f
            dW = (lora_B[key] @ lora_A[key]).to(mod.weight.dtype)
            mod.weight.add_(dW)


def main():
    parser = argparse.ArgumentParser(description="Step 5: Complete 5-Way Benchmark (Base vs SFT vs Grad-LoRA vs Nullspace-WSP vs Functional-SVD LoRA)")
    parser.add_argument("--model", type=str, required=True, help="Base model ID or local path")
    parser.add_argument("--train_file", type=str, required=True, help="Training .jsonl/.json file")
    parser.add_argument("--test_file", type=str, required=True, help="Held-out test .jsonl/.json file")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs/passes (default: 5)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--lr", type=str, default="auto", help="Step size or 'auto'")
    parser.add_argument("--rank", type=int, default=32, help="LoRA rank r (default: 32)")
    parser.add_argument("--output_json", type=str, default=None, help="Optional JSON path to save benchmark summary")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(args.device).eval()
    arch_info = discover_architecture_groups(model, tok, args.device)
    _, layers = get_transformer_layers(model)
    mod_to_group = arch_info["mod_to_group"]
    d_model = getattr(model.config, "hidden_size", 1024)
    lr_val = (1.5e-5 if d_model <= 2048 else 5.0e-6) if args.lr == "auto" else float(args.lr)

    base_weights = {
        (l, mname): lay.get_submodule(mname).weight.detach().cpu().clone()
        for l, lay in enumerate(layers)
        for mname in mod_to_group.keys()
    }

    def restore_base():
        with torch.no_grad():
            for (l, mname), w0 in base_weights.items():
                layers[l].get_submodule(mname).weight.copy_(w0.to(dtype).to(args.device))

    # Prepare WikiText eval (40 passages) and ARC-Easy eval (200 questions)
    wiki_eval_path = REPO_DATA_DIR / "wikitext_eval_40.jsonl"
    wiki_texts = [json.loads(l)["text"] for l in wiki_eval_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    wiki_eval_ids, base_wiki_lps = [], []
    with torch.no_grad():
        for txt in wiki_texts:
            ids = tok(txt, return_tensors="pt", truncation=True, max_length=256).input_ids.to(args.device)
            wiki_eval_ids.append(ids)
            base_wiki_lps.append(F.log_softmax(model(input_ids=ids, use_cache=False).logits[0, :-1].float(), dim=-1).cpu())

    arc_eval_path = REPO_DATA_DIR / "arc_easy_eval_200.jsonl"
    arc_raw = [json.loads(l) for l in arc_eval_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    arc_eval_items = []
    for r in arc_raw:
        p_txt = format_messages_safe(tok, [{"role": "user", "content": r["prompt"]}], add_generation_prompt=True)
        ids = tok(p_txt, return_tensors="pt").input_ids.to(args.device)
        letter_ids = [tok.encode(lbl, add_special_tokens=False)[0] for lbl in r["labels"]]
        arc_eval_items.append({"ids": ids, "letter_ids": letter_ids, "labels": r["labels"], "answer": r["answer"]})

    train_pairs = load_universal_sft_dataset(tok, args.train_file)
    test_pairs = load_universal_sft_dataset(tok, args.test_file)
    shared_k_tr = detect_shared_prefix_len(train_pairs)
    shared_k_te = detect_shared_prefix_len(test_pairs)

    summary = {}
    # 1. Unedited Base
    summary["1_Unedited_Base"] = evaluate_suite(tok, model, test_pairs, shared_k_te, wiki_eval_ids, base_wiki_lps, arc_eval_items, args.device)
    print("  [1_Unedited_Base]", json.dumps(summary["1_Unedited_Base"]), flush=True)

    # Free evaluation model from GPU while subprocesses run Step 1, Step 2 (SFT & WSP), and Step 3
    del model
    if args.device == "cuda":
        torch.cuda.empty_cache()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        prof_pt = tmp / "profile.pt"
        sft_dir = tmp / "sft_out"
        wsp_dir = tmp / "wsp_out"
        lora_dir = tmp / "lora_out"

        subprocess.run([sys.executable, str(Path(__file__).parent / "wsp_1_profile_base.py"), "--model", args.model, "--output_profile", str(prof_pt), "--device", args.device], check=True)
        subprocess.run([sys.executable, str(Path(__file__).parent / "wsp_2_compile_dataset.py"), "--model", args.model, "--profile", str(prof_pt), "--train_file", args.train_file, "--output_dir", str(sft_dir), "--epochs", str(args.epochs), "--batch_size", str(args.batch_size), "--lr", str(lr_val), "--disable_nullspace", "--device", args.device], check=True)
        subprocess.run([sys.executable, str(Path(__file__).parent / "wsp_2_compile_dataset.py"), "--model", args.model, "--profile", str(prof_pt), "--train_file", args.train_file, "--output_dir", str(wsp_dir), "--epochs", str(args.epochs), "--batch_size", str(args.batch_size), "--lr", str(lr_val), "--device", args.device], check=True)
        subprocess.run([sys.executable, str(Path(__file__).parent / "wsp_3_export_lora.py"), "--wsp_dir", str(wsp_dir), "--rank", str(args.rank), "--output_peft_dir", str(lora_dir), "--device", args.device], check=True)

        # Reload model on GPU for evaluation & Grad-LoRA baseline
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(args.device).eval()
        _, layers = get_transformer_layers(model)

        # 2. Evaluate Standard Full SFT
        sft_art = torch.load(sft_dir / "wsp_artifact.pt", map_location="cpu")
        with torch.no_grad():
            for (l, mname), w0 in base_weights.items():
                w_new = (w0.float() + sft_art["deltas"][(l, mname)].float()).to(dtype).to(args.device)
                layers[l].get_submodule(mname).weight.copy_(w_new)
        summary["2_Standard_Full_SFT"] = evaluate_suite(tok, model, test_pairs, shared_k_te, wiki_eval_ids, base_wiki_lps, arc_eval_items, args.device)
        print("  [2_Standard_Full_SFT]", json.dumps(summary["2_Standard_Full_SFT"]), flush=True)

        # 3. Train & Evaluate Standard Gradient LoRA
        restore_base()
        train_standard_grad_lora(model, tok, layers, mod_to_group, train_pairs, shared_k_tr, args.rank, args.epochs, lr_val, args.batch_size, args.device)
        summary[f"3_Standard_Grad_LoRA_r{args.rank}"] = evaluate_suite(tok, model, test_pairs, shared_k_te, wiki_eval_ids, base_wiki_lps, arc_eval_items, args.device)
        print(f"  [3_Standard_Grad_LoRA_r{args.rank}]", json.dumps(summary[f"3_Standard_Grad_LoRA_r{args.rank}"]), flush=True)

        # 4. Evaluate Nullspace-WSP (Ours)
        wsp_art = torch.load(wsp_dir / "wsp_artifact.pt", map_location="cpu")
        with torch.no_grad():
            for (l, mname), w0 in base_weights.items():
                w_new = (w0.float() + wsp_art["deltas"][(l, mname)].float()).to(dtype).to(args.device)
                layers[l].get_submodule(mname).weight.copy_(w_new)
        summary["4_Nullspace_WSP_Ours"] = evaluate_suite(tok, model, test_pairs, shared_k_te, wiki_eval_ids, base_wiki_lps, arc_eval_items, args.device)
        print("  [4_Nullspace_WSP_Ours]", json.dumps(summary["4_Nullspace_WSP_Ours"]), flush=True)

        # 5. Evaluate Functional-SVD Compiled LoRA (Ours)
        peft_sd = torch.load(lora_dir / "adapter_model.bin", map_location="cpu")
        layer_prefix = arch_info["layer_prefix"]
        with torch.no_grad():
            for (l, mname), w0 in base_weights.items():
                bkey = f"base_model.model.{layer_prefix}.{l}.{mname}"
                A = peft_sd[f"{bkey}.lora_A.weight"].float()
                B = peft_sd[f"{bkey}.lora_B.weight"].float()
                w_new = (w0.float() + (B @ A)).to(dtype).to(args.device)
                layers[l].get_submodule(mname).weight.copy_(w_new)
        summary[f"5_Functional_SVD_LoRA_r{args.rank}_Ours"] = evaluate_suite(tok, model, test_pairs, shared_k_te, wiki_eval_ids, base_wiki_lps, arc_eval_items, args.device)
        print(f"  [5_Functional_SVD_LoRA_r{args.rank}_Ours]", json.dumps(summary[f"5_Functional_SVD_LoRA_r{args.rank}_Ours"]), flush=True)

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("=== 5-WAY BENCHMARK COMPLETE ===", flush=True)

if __name__ == "__main__":
    main()
