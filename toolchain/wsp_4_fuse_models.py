#!/usr/bin/env python3
r"""
Step 4 of Universal WSP Toolchain: Closed-Form Multi-Dataset / Multi-Skill Fusion
=================================================================================
Fuses K independently compiled WSP artifacts (`wsp_artifact.pt`) into a single unified
Hugging Face model checkpoint with ZERO gradient steps!
Supports both:
  - `layerwise_align` (Sequential Layer-by-Layer Closed-Form Residual Alignment across
    both attention output and MLP output projections when training JSONLs are available),
  - `gram_closed_form` (One-Shot Closed-Form Gram-Resolved Matrix Fusion across all linear
    groups when only `wsp_artifact.pt` files are provided).
"""
import argparse
import time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from wsp_0_common import get_transformer_layers, load_universal_sft_dataset

def out_of(o):
    return (o[0] if isinstance(o, tuple) else o)[0]

def main():
    parser = argparse.ArgumentParser(description="Step 4: Fuse Multiple Compiled WSP Datasets in Closed Form")
    parser.add_argument("--model", type=str, required=True, help="Base model ID or local path")
    parser.add_argument("--profile", type=str, required=True, help="Path to Step 1 nullspace profile .pt")
    parser.add_argument("--wsp_dirs", nargs="+", required=True, help="List of Step 2 artifact directories to fuse")
    parser.add_argument("--train_files", nargs="*", default=None, help="Optional list of training .jsonl files for sequential layerwise alignment")
    parser.add_argument("--mode", type=str, default="auto", choices=["auto", "layerwise_align", "gram_closed_form"])
    parser.add_argument("--output_merged_model", type=str, required=True, help="Output directory for fused HF model")
    parser.add_argument("--mu_gen", type=float, default=1.0, help="General covariance protection weight during fusion")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    t0 = time.time()
    prof = torch.load(args.profile, map_location="cpu")
    cov_gen = prof["cov_gen"]
    arts = [torch.load(Path(d) / "wsp_artifact.pt", map_location="cpu") for d in args.wsp_dirs]
    arch_info = arts[0]["arch_info"]
    mod_to_group = arch_info["mod_to_group"]
    group_to_rep = arch_info["group_to_rep"]

    tok = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(args.device).eval()
    _, layers = get_transformer_layers(model)

    train_files = args.train_files
    if not train_files:
        inferred = [a.get("train_file") for a in arts if a.get("train_file") and Path(a.get("train_file")).exists()]
        if len(inferred) == len(arts):
            train_files = inferred

    use_layerwise = (args.mode == "layerwise_align") or (args.mode == "auto" and train_files and len(train_files) == len(arts))

    if use_layerwise:
        print(f"[Step 4] Running Sequential Layer-by-Layer Closed-Form Residual Alignment across {len(arts)} datasets...")
        base_state = {
            (l, mname): lay.get_submodule(mname).weight.detach().cpu().clone()
            for l, lay in enumerate(layers)
            for mname in mod_to_group.keys()
        }
        # Find the MLP output module (e.g. mlp.down_proj) in each layer
        res_mods = arch_info.get("residual_out_mods", [])
        down_gid, down_mname = res_mods[-1]

        teach_groups = []
        for art, tf in zip(arts, train_files):
            with torch.no_grad():
                for (l, mname), w0 in base_state.items():
                    w_t = (w0.float() + art["deltas"][(l, mname)].float()).to(dtype).to(args.device)
                    layers[l].get_submodule(mname).weight.copy_(w_t)
            pairs = load_universal_sft_dataset(tok, tf)[:120]
            seqs = [torch.cat([p_ids, g_ids]).to(args.device) for p_ids, g_ids in pairs]
            cur, teach_seqs = {}, []
            hs = [
                lay.register_forward_hook(lambda _, inp, o, l=l: cur.__setitem__(l, out_of(o).detach().cpu()))
                for l, lay in enumerate(layers)
            ]
            with torch.no_grad():
                for s in seqs:
                    cur.clear()
                    model(input_ids=s[None], use_cache=False)
                    teach_seqs.append({l: cur[l].clone() for l in range(len(layers))})
            for h in hs:
                h.remove()
            teach_groups.append((seqs, teach_seqs))

        with torch.no_grad():
            for (l, mname), w0 in base_state.items():
                layers[l].get_submodule(mname).weight.copy_(w0.to(dtype).to(args.device))

        for l, lay in enumerate(layers):
            mod = lay.get_submodule(down_mname)
            H_list, D_list = [], []
            for seqs, teach_seqs in teach_groups:
                for s, t_dict in zip(seqs, teach_seqs):
                    captured = {}
                    h1 = mod.register_forward_pre_hook(lambda _, inp: captured.__setitem__("in", inp[0][0].detach().float()))
                    h2 = lay.register_forward_hook(lambda _, inp, o: captured.__setitem__("out", out_of(o).detach().float()))
                    with torch.no_grad():
                        model(input_ids=s[None], use_cache=False)
                    h1.remove()
                    h2.remove()
                    H_list.append(captured["in"])
                    D_list.append(t_dict[l].to(args.device).float() - captured["out"])
            H = torch.cat(H_list, dim=0)
            D = torch.cat(D_list, dim=0)
            hh = H.T @ H
            tr = hh.trace()
            C = cov_gen[(l, down_gid)].to(args.device).float()
            reg = hh + args.mu_gen * (tr / (C.trace() + 1e-8)) * C + 1e-2 * (tr / len(hh) + 1e-8) * torch.eye(len(hh), device=args.device)
            dW = torch.linalg.solve(reg, H.T @ D).T
            with torch.no_grad():
                mod.weight.add_(dW.to(mod.weight.dtype))
        print(f"[Step 4] Completed Sequential Layer-by-Layer Closed-Form Alignment in {time.time()-t0:.1f}s.")
    else:
        print(f"[Step 4] Running One-Shot Closed-Form Gram-Resolved Matrix Fusion across {len(arts)} datasets...")
        with torch.no_grad():
            for l, lay in enumerate(layers):
                for gid in group_to_rep.keys():
                    key = (l, gid)
                    Cg = cov_gen[key].to(args.device).float()
                    d = Cg.shape[0]
                    G_sum = args.mu_gen * (Cg / (Cg.trace() / d + 1e-8)) + 0.05 * torch.eye(d, device=args.device)
                    G_list = []
                    for art in arts:
                        Gt = art["cov_task"][key].to(args.device).float()
                        Gn = Gt / (Gt.trace() / d + 1e-8) + 0.02 * torch.eye(d, device=args.device)
                        G_list.append(Gn)
                        G_sum = G_sum + Gn
                    G_inv = torch.linalg.inv(G_sum)
                    for mname, mgid in mod_to_group.items():
                        if mgid != gid:
                            continue
                        rhs = 0
                        for art, Gn in zip(arts, G_list):
                            rhs = rhs + art["deltas"][(l, mname)].to(args.device).float() @ Gn
                        dW_star = (rhs @ G_inv).to(dtype)
                        lay.get_submodule(mname).weight.add_(dW_star)

    out_dir = Path(args.output_merged_model)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print(f"[Step 4] Saved unified multi-dataset Hugging Face model to {out_dir} in {time.time()-t0:.1f}s.")

if __name__ == "__main__":
    main()
