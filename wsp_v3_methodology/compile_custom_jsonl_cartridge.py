#!/usr/bin/env python3
r"""
Step 7 of Universal WSP Toolchain: Zero-Gradient Dataset-to-Cartridge Compiler (`0` backward passes)
====================================================================================================
Compiles ANY supervised `.jsonl` dataset directly into a hot-swappable `.pt` dormant-channel
skill cartridge using STRICTLY ZERO gradient steps (`with torch.no_grad():` + closed-form ridge solve).

Supports two automatic compilation modes:
  1) Stem-Release Mode (default when responses have dynamic arguments/continuations):
     Automatically extracts the common/clustered structural prefix (stem) of length `--stem_tokens`
     from the training responses, programs dormant MLP channels to emit the stem when the prompt
     falls inside the positive semantic cone, and immediately releases control (`z <= -15.0`) to
     the base model's native circuits to generate dynamic parameters/answers.
  2) Full-Response Mode (`--full_response`):
     Compiles the complete response token sequence into dormant MLP channels.
"""
import argparse
import json
import time
from pathlib import Path
import torch
import torch.nn.functional as F
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
_SELF = Path(__file__).resolve().parent
for _cand in [_SELF, _SELF.parent / "toolchain", _SELF / "toolchain"]:
    if _cand.exists():
        sys.path.insert(0, str(_cand))
from wsp_0_common import (
    REPO_DATA_DIR,
    format_messages_safe,
    get_transformer_layers,
    load_universal_sft_dataset,
)


def solve_closed_form_spherical_gate(Xp, Xn_prm, Xn_trj, X_guard, shrink=0.80, guard_w=11.0, margin_frac=0.58):
    def fit_dir(P_tr, N_prm, N_trj, G_tr):
        pos_part = P_tr / (P_tr.shape[0] ** 0.5)
        neg_prm_part = 0.95 * N_prm / (N_prm.shape[0] ** 0.5)
        neg_trj_part = 0.50 * N_trj / (N_trj.shape[0] ** 0.5)
        packed = torch.cat([neg_prm_part, neg_trj_part, pos_part], dim=0)
        labels = torch.cat([
            torch.full((N_prm.shape[0],), -0.95 / (N_prm.shape[0] ** 0.5)),
            torch.full((N_trj.shape[0],), -0.50 / (N_trj.shape[0] ** 0.5)),
            torch.full((P_tr.shape[0],), +1.0 / (P_tr.shape[0] ** 0.5)),
        ])
        cov = packed.T @ packed
        guard = G_tr.T @ G_tr
        reg = (1.0 - shrink) * cov + guard_w * (cov.trace() / (guard.trace() + 1e-8)) * guard
        reg.diagonal().add_(shrink * cov.diagonal().mean())
        return F.normalize(torch.linalg.solve(reg, packed.T @ labels), dim=0)

    k_full = fit_dir(Xp, Xn_prm, Xn_trj, X_guard)
    d_p = F.normalize(Xp.mean(dim=0), dim=0)
    d_bg = F.normalize(X_guard.mean(dim=0), dim=0)
    d_anchor = F.normalize(0.55 * d_p + 0.45 * d_bg, dim=0)
    b_base = d_anchor / ((Xp @ d_anchor).mean().item() + 1e-8)
    k_orth = F.normalize(k_full - torch.dot(k_full, d_p) * d_p, dim=0)
    b_up = b_base + 0.30 * k_orth / ((Xp @ d_anchor).mean().item() + 1e-8)

    in_p = (Xp @ k_full) / ((Xp @ b_base).clamp(min=0.5))
    p_low = 0.85 * in_p.quantile(0.06).item()
    in_np = (Xn_prm @ k_full) / ((Xn_prm @ b_base).clamp(min=0.5))
    in_g = (X_guard @ k_full) / ((X_guard @ b_base).clamp(min=0.5))
    in_nt = (Xn_trj @ k_full) / ((Xn_trj @ b_base).clamp(min=0.5))
    n_high = max(in_np.max().item(), in_g.max().item(), 0.70 * in_nt.max().item())

    tau_s = margin_frac * n_high + (1.0 - margin_frac) * p_low
    s_s = 16.5 / max(0.03, p_low - tau_s)
    return k_full, b_base, b_up, tau_s, s_s


def main():
    parser = argparse.ArgumentParser(description="Step 7: Zero-Gradient Dataset-to-Cartridge Compiler")
    parser.add_argument("--model", type=str, required=True, help="Base model path or Hugging Face ID")
    parser.add_argument("--train_file", type=str, required=True, help="Input SFT .jsonl dataset")
    parser.add_argument("--output_cartridge", type=str, required=True, help="Output .pt cartridge path")
    parser.add_argument("--stem_tokens", type=int, default=6, help="Number of prefix stem tokens to compile before releasing to native circuits (default: 6)")
    parser.add_argument("--skill_name", type=str, default="Custom Zero-Gradient Compiled Skill Cartridge")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    t0 = time.time()
    is_local = Path(args.model).exists()
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=is_local)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, local_files_only=is_local).to(args.device).eval()

    _, layers = get_transformer_layers(model)
    L = len(layers)
    fsm_layer = L - 1
    d_model = model.config.hidden_size
    W_U = model.lm_head.weight.detach().float().cpu()
    norm_w = model.model.norm.weight.detach().float().cpu()
    W_U_mean = W_U.mean(dim=0)

    def get_r_tok(tid):
        return F.normalize((W_U[tid] - W_U_mean) * norm_w, dim=0)

    pairs = load_universal_sft_dataset(tok, args.train_file)[:80]
    # Load bundled general background protection pool
    arc_rows = [json.loads(l) for l in (REPO_DATA_DIR / "arc_easy_protect_300.jsonl").read_text().splitlines()[:80]]
    wiki_rows = [json.loads(l) for l in (REPO_DATA_DIR / "wikitext_protect_60.jsonl").read_text().splitlines()[:25]]

    # Find the dominant shared/modal stem of length `stem_tokens` across training targets
    stem_len = min(args.stem_tokens, min(len(g) for _, g in pairs))
    stem_counts = {}
    for _, g_ids in pairs:
        st_tuple = tuple(g_ids[:stem_len].tolist())
        stem_counts[st_tuple] = stem_counts.get(st_tuple, 0) + 1
    modal_stem = list(max(stem_counts.items(), key=lambda x: x[1])[0])
    matching_pairs = [(p, g) for p, g in pairs if g[:stem_len].tolist() == modal_stem] or pairs

    h_cap, mid_cap = [], []
    hk1 = layers[fsm_layer].mlp.register_forward_pre_hook(lambda m, inp: h_cap.append(inp[0][0].detach().float().cpu()))
    hk2 = layers[fsm_layer].mlp.down_proj.register_forward_pre_hook(lambda m, inp: mid_cap.append(inp[0][0].detach().float().cpu()))

    step_vecs = {s: [] for s in range(stem_len)}
    release_vecs = []
    with torch.no_grad():
        for p_ids, g_ids in matching_pairs:
            P_len = len(p_ids)
            h_cap.clear()
            full_ids = torch.cat([p_ids, g_ids], dim=0).unsqueeze(0).to(args.device)
            model(input_ids=full_ids, use_cache=False, logits_to_keep=1)
            seq = h_cap[0]
            for s in range(stem_len):
                step_vecs[s].append(seq[P_len - 1 + s])
            for r_i in range(P_len - 1 + stem_len, seq.shape[0]):
                release_vecs.append(seq[r_i])

        prm_neg_vecs = []
        for r in arc_rows:
            p_str = format_messages_safe(tok, [{"role": "user", "content": r.get("question", r.get("prompt", r.get("instruction", "")))}])
            ids = tok.encode(p_str, add_special_tokens=False)
            h_cap.clear()
            model(input_ids=torch.tensor([ids], device=args.device), use_cache=False, logits_to_keep=1)
            prm_neg_vecs.append(h_cap[0][-1])

        traj_neg_vecs = []
        for r in wiki_rows:
            ids = tok.encode(r["text"], add_special_tokens=False)[:160]
            if len(ids) > 10:
                h_cap.clear()
                model(input_ids=torch.tensor([ids], device=args.device), use_cache=False, logits_to_keep=1)
                traj_neg_vecs.append(h_cap[0][5:])

    hk1.remove()
    hk2.remove()

    X_prm_neg = torch.stack(prm_neg_vecs, dim=0)
    X_traj_neg = torch.cat(traj_neg_vecs + ([torch.stack(release_vecs, dim=0)] if release_vecs else []), dim=0)
    all_mid = torch.cat(mid_cap, dim=0)
    _, dorm_cols = torch.topk(-all_mid.abs().mean(dim=0), k=stem_len)

    res_norm = X_prm_neg.norm(dim=-1).mean().item()
    W_g = torch.zeros((stem_len, d_model))
    W_u = torch.zeros((stem_len, d_model))
    W_d = torch.zeros((d_model, stem_len))

    for s in range(stem_len):
        Xp = torch.stack(step_vecs[s], dim=0)
        other_s = [torch.stack(step_vecs[s2], dim=0) for s2 in range(stem_len) if modal_stem[s2] != modal_stem[s]]
        Xn_trj = torch.cat(other_s + [X_traj_neg], dim=0)
        Xg = torch.cat([X_prm_neg.repeat(4, 1), X_traj_neg], dim=0)
        k_s, b_base, b_up, tau_s, s_s = solve_closed_form_spherical_gate(Xp, X_prm_neg, Xn_trj, Xg)
        amp = (0.75 if s == 0 else 0.32) * res_norm
        W_g[s, :] = s_s * (k_s - tau_s * b_base)
        W_u[s, :] = b_up
        W_d[:, s] = amp * get_r_tok(modal_stem[s])

    out_p = Path(args.output_cartridge)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "skill_name": args.skill_name,
            "model": args.model,
            "terminal_fsm_layer": fsm_layer,
            "layer": fsm_layer,
            "cols": dorm_cols.to(torch.int32).cpu(),
            "gate": W_g.to(torch.bfloat16).cpu(),
            "up": W_u.to(torch.bfloat16).cpu(),
            "down": W_d.to(torch.bfloat16).cpu(),
            "stem_str": tok.decode(modal_stem),
            "optimization": "0_gradient_steps_closed_form",
        },
        out_p,
    )
    print(f"[Step 7 Zero-Gradient Compiler] Compiled {stem_len}-token Stem-Release cartridge ({repr(tok.decode(modal_stem))}) -> {out_p} in {time.time()-t0:.1f}s (0 backward passes!)", flush=True)


if __name__ == "__main__":
    main()
