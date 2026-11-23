#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B", help="Model path or Hugging Face ID")
    ap.add_argument("--output_json", default=None, help="Optional JSON output path")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    is_local = Path(args.model).exists()
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=is_local)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, local_files_only=is_local).to(args.device).eval()
    L = model.config.num_hidden_layers
    d = model.config.hidden_size
    fsm_layer = L - 1

    calib_pos = [
        "Quick factual check: What is the chemical symbol for gold? Answer in one word.",
        "Quick factual check: What is the capital city of Japan? Answer in one word.",
        "Quick arithmetic: What is 14 + 29? Give the direct number.",
        "Quick arithmetic: What is 12 * 11? Give the direct number.",
        "What is the chemical symbol for silver? Answer in one word.",
        "What is the capital of Italy? One word only.",
        "Compute 18 + 37. Give just the number.",
        "What is 15 * 12? Reply with the number only.",
        "How many continents are there on Earth? Just the number.",
        "Is 19 a prime number? Answer yes or no.",
        "What planet is known as the Red Planet? One word.",
        "What is the positive square root of 144? Just the number.",
    ]
    calib_neg = [
        "Solve for x: 3x + 7 = 25.",
        "Solve the system of equations: 2x + y = 11 and x - y = 1.",
        "A store sells notebooks for $4 each and pens for $2 each. If Maria buys 5 notebooks and 6 pens, how much change does she get from a $50 bill?",
        "A train leaves Station A at 60 mph. Two hours later, a second train leaves Station A at 90 mph. How many hours will it take to catch the first?",
        "Explain the difference between mitosis and meiosis in cellular biology.",
        "Explain the difference between a SQL INNER JOIN and a LEFT OUTER JOIN with a short example.",
        "Write a Python function to check whether a given string is a palindrome.",
        "Summarize the main causes of the Industrial Revolution in three concise bullet points.",
        "Find all real solutions to x^2 - 5x + 6 = 0.",
        "Explain how photosynthesis converts light energy into chemical energy.",
    ]
    disjoint_para = [
        "Chemical symbol for potassium? One word.",
        "Capital city of Portugal? One word.",
        "What is 25 plus 38? Number only.",
        "How many semitones are in an octave? Just the number.",
        "Is 31 a prime number? Yes or no.",
        "Closest planet to the Sun? One word.",
        "Positive square root of 169? Number only.",
        "Freezing point of pure water in Kelvin? Number only.",
    ]
    disjoint_hard_neg = [
        "If 4x - 7 = 21, what is the value of 3x + 2? Explain your algebraic steps.",
        "A baker bakes 120 loaves a day and sells 85% of them at $5 each. What is the total revenue over 4 days?",
        "Compare the parliamentary system and the presidential system of government.",
        "Explain how a relational database B-tree index speeds up range queries.",
    ]

    def fmt(q):
        try:
            return tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)

    cand_layers = sorted(list({int(L * 0.5), int(L * 0.72), int(L * 0.83), fsm_layer}))
    acts = {l: [] for l in cand_layers}
    hooks = [
        model.model.layers[l].mlp.register_forward_pre_hook(
            (lambda l_idx: lambda m, inp: acts[l_idx].append(inp[0][0, -1].detach().float().cpu()))(l)
        )
        for l in cand_layers
    ]

    def collect(q_list):
        for l in cand_layers:
            acts[l].clear()
        with torch.no_grad():
            for q in q_list:
                model(**tok([fmt(q)], return_tensors="pt").to(args.device), use_cache=False, logits_to_keep=1)
        return {l: torch.stack(acts[l], dim=0) for l in cand_layers}

    A_pos, A_neg, A_para, A_hard = collect(calib_pos), collect(calib_neg), collect(disjoint_para), collect(disjoint_hard_neg)
    for h in hooks:
        h.remove()

    print("=== [1] Across-Layer Semantic Separability (2-Cone Recall @ 0% Hard-Negative FP) ===")
    layer_recalls = {}
    for l in cand_layers:
        Xp = F.normalize(A_pos[l], dim=-1)
        Xn = F.normalize(A_neg[l], dim=-1)
        X_para = F.normalize(A_para[l], dim=-1)
        X_hard = F.normalize(A_hard[l], dim=-1)
        cov = Xp.T @ Xp + 2.0 * (Xn.T @ Xn) + 0.4 * torch.eye(d)
        k1 = F.normalize(torch.linalg.solve(cov, Xp[:4].mean(0) - Xn.mean(0)), dim=0)
        k2 = F.normalize(torch.linalg.solve(cov, Xp[4:].mean(0) - Xn.mean(0)), dim=0)
        tau1 = max((Xn @ k1).max().item(), (X_hard @ k1).max().item()) + 0.005
        tau2 = max((Xn @ k2).max().item(), (X_hard @ k2).max().item()) + 0.005
        hit = ((X_para @ k1) > tau1) | ((X_para @ k2) > tau2)
        rec_val = round(hit.float().mean().item(), 4)
        layer_recalls[f"layer_{l}"] = rec_val
        print(f"  Layer {l:2d}: Disjoint Paraphrase Recall @ 0% FP = {rec_val*100:5.1f}%")

    print("\n=== [2] Nullspace Beacon Bus SNR & Residual Cosine Preservation ===")
    X_last_all = torch.cat([A_pos[fsm_layer], A_neg[fsm_layer], A_hard[fsm_layer]], dim=0)
    _, _, Vh = torch.linalg.svd(X_last_all, full_matrices=False)
    U_nat = Vh[:min(16, Vh.shape[0] - 2), :].T
    gm_last = model.model.layers[fsm_layer].post_attention_layernorm.weight.detach().float().cpu()
    inv_gm = 1.0 / (gm_last.abs() + 0.1)
    l_sens = cand_layers[-2]
    Omega = 500.0 * (U_nat @ U_nat.T)
    with torch.no_grad():
        for l in range(l_sens + 1, L):
            lay = model.model.layers[l]
            ga = lay.input_layernorm.weight.detach().float().cpu() * inv_gm
            for proj in [lay.self_attn.k_proj, lay.self_attn.v_proj]:
                W = proj.weight.detach().float().cpu() * ga.unsqueeze(0)
                Omega += (W.T @ W) / W.shape[0]
    _, evecs = torch.linalg.eigh(Omega)
    w_readout = F.normalize(evecs[:, 0] - U_nat @ (U_nat.T @ evecs[:, 0]), dim=0)
    v_beacon = F.normalize(w_readout * inv_gm, dim=0).to(args.device).to(torch.bfloat16)

    l_clean, l_inj = [], []
    hk_c = model.model.layers[fsm_layer].mlp.register_forward_pre_hook(lambda m, inp: l_clean.append(inp[0][0, -1].detach().float().cpu()))
    with torch.no_grad():
        for q in disjoint_para + disjoint_hard_neg:
            model(**tok([fmt(q)], return_tensors="pt").to(args.device), use_cache=False, logits_to_keep=1)
    hk_c.remove()

    state = {"inject": False}
    def hk_s_fn(m, inp, out):
        if state["inject"]:
            out = out.clone()
            r_norm = out[0, -1].float().norm().item()
            out[0, -1] = out[0, -1] + (1.2 * r_norm) * v_beacon
        return out

    hk_s = model.model.layers[l_sens].mlp.register_forward_hook(hk_s_fn)
    hk_i = model.model.layers[fsm_layer].mlp.register_forward_pre_hook(lambda m, inp: l_inj.append(inp[0][0, -1].detach().float().cpu()))
    with torch.no_grad():
        for idx, q in enumerate(disjoint_para + disjoint_hard_neg):
            state["inject"] = idx < len(disjoint_para)
            model(**tok([fmt(q)], return_tensors="pt").to(args.device), use_cache=False, logits_to_keep=1)
    hk_s.remove()
    hk_i.remove()

    C_clean, C_inj = torch.stack(l_clean), torch.stack(l_inj)
    n_p = len(disjoint_para)
    proj_trig = C_inj[:n_p] @ w_readout
    proj_untrig = C_inj[n_p:] @ w_readout
    snr = round(proj_trig.min().item() / (proj_untrig.abs().max().item() + 1e-6), 2)
    v_cpu = F.normalize(v_beacon.float().cpu() * gm_last, dim=0)
    strip_v = lambda X: X - (X @ v_cpu).unsqueeze(1) * v_cpu.unsqueeze(0)
    cos_p = round(F.cosine_similarity(strip_v(C_clean[:n_p]), strip_v(C_inj[:n_p]), dim=-1).mean().item(), 4)
    print(f"  Sensor Layer {l_sens} -> Actuator Layer {fsm_layer}: Triggered Min={proj_trig.min().item():+6.2f} | Max Untriggered Noise={proj_untrig.abs().max().item():.3f} | SNR={snr:.2f}x | Residual Cosine Preservation={cos_p:.4f}")
    if args.output_json:
        Path(args.output_json).write_text(json.dumps({
            "sensor_layer": l_sens,
            "actuator_layer": fsm_layer,
            "triggered_min_readout": round(proj_trig.min().item(), 3),
            "untriggered_max_noise": round(proj_untrig.abs().max().item(), 4),
            "snr_ratio": snr,
            "residual_cosine_preservation": cos_p,
            "layer_recalls_at_zero_fp": layer_recalls,
        }, indent=2))


if __name__ == "__main__":
    main()
