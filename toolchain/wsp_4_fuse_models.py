#!/usr/bin/env python3
r"""
Step 4 of Universal WSP Toolchain: Closed-Form Multi-Dataset / Multi-Skill Fusion
=================================================================================
Fuses K independently compiled WSP artifacts (wsp_artifact.pt) into a single unified
model checkpoint by solving the Closed-Form Gram-Resolved Normal Equation:
  dW* = (\sum_{k=1}^K dW_k G_k) (\sum_{k=1}^K G_k + \mu_gen C_gen + \lambda I)^{-1}
Zero gradient steps required! Overcomes the catastrophic interference of Task Arithmetic (SUM).
"""
import argparse
import time
from pathlib import Path
import torch
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

def main():
    parser = argparse.ArgumentParser(description="Step 4: Fuse Multiple Compiled WSP Datasets in Closed Form")
    parser.add_argument("--model", type=str, required=True, help="Base model ID or local path")
    parser.add_argument("--profile", type=str, required=True, help="Path to Step 1 nullspace profile .pt")
    parser.add_argument("--wsp_dirs", nargs="+", required=True, help="List of Step 2 artifact directories to fuse")
    parser.add_argument("--output_merged_model", type=str, required=True, help="Output directory for fused HF model")
    parser.add_argument("--mu_gen", type=float, default=0.5, help="General covariance protection weight during fusion")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    t0 = time.time()
    prof = torch.load(args.profile, map_location="cpu")
    cov_gen = prof["cov_gen"]
    arts = [torch.load(Path(d) / "wsp_artifact.pt", map_location="cpu") for d in args.wsp_dirs]

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(args.device)

    with torch.no_grad():
        for l, lay in enumerate(model.model.layers):
            for rname in ("qkv", "o", "gu", "down"):
                key = (l, rname)
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

                for mname, rn in MOD_TO_RNAME.items():
                    if rn != rname:
                        continue
                    rhs = 0
                    for art, Gn in zip(arts, G_list):
                        rhs = rhs + art["deltas"][(l, mname)].to(args.device).float() @ Gn
                    dW_star = (rhs @ G_inv).to(torch.bfloat16)
                    lay.get_submodule(mname).weight.add_(dW_star)

    out_dir = Path(args.output_merged_model)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print(f"[Step 4] Fused {len(arts)} datasets in closed form and saved unified model to {out_dir} in {time.time()-t0:.1f}s.")

if __name__ == "__main__":
    main()
