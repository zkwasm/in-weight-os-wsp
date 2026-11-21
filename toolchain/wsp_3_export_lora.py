#!/usr/bin/env python3
r"""
Step 3 of Universal WSP Toolchain: Functional-SVD Export to Standard Hugging Face PEFT LoRA
===========================================================================================
Factorizes the compiled weight delta dW in activation space via Covariance-Weighted SVD:
  dW @ C_task^{1/2} = U_r S_r V_r^T
  => lora_B = U_r S_r^{1/2},  lora_A = S_r^{1/2} V_r^T C_task^{-1/2}
Works on ANY architecture discovered in Step 1/2 and saves a standard Hugging Face PEFT
directory (`adapter_config.json` + `adapter_model.bin`).
"""
import argparse
import json
import time
from pathlib import Path
import torch

def main():
    parser = argparse.ArgumentParser(description="Step 3: Export Compiled WSP Artifact to Standard Hugging Face PEFT LoRA")
    parser.add_argument("--wsp_dir", type=str, required=True, help="Directory containing wsp_artifact.pt from Step 2")
    parser.add_argument("--rank", type=int, default=32, help="LoRA rank r (e.g. 16, 32, 64)")
    parser.add_argument("--output_peft_dir", type=str, required=True, help="Output directory for PEFT LoRA adapter")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    t0 = time.time()
    art = torch.load(Path(args.wsp_dir) / "wsp_artifact.pt", map_location="cpu")
    arch_info = art["arch_info"]
    mod_to_group = arch_info["mod_to_group"]
    layer_prefix = arch_info["layer_prefix"]
    deltas = art["deltas"]
    cov_task = art["cov_task"]
    r = args.rank

    sqrt_cache = {}
    for key, C_cpu in cov_task.items():
        C = C_cpu.to(args.device).float()
        d = C.shape[0]
        C_reg = C / (C.trace() / d + 1e-8) + 0.01 * torch.eye(d, device=C.device)
        evals, evecs = torch.linalg.eigh(C_reg)
        evals = evals.clamp(min=1e-6)
        C_half = evecs @ torch.diag(evals.sqrt()) @ evecs.T
        C_inv_half = evecs @ torch.diag(1.0 / evals.sqrt()) @ evecs.T
        sqrt_cache[key] = (C_half, C_inv_half)

    peft_state = {}
    with torch.no_grad():
        for (l, mname), dW_cpu in deltas.items():
            gid = mod_to_group[mname]
            C_half, C_inv_half = sqrt_cache[(l, gid)]
            dW = dW_cpu.to(args.device).float()
            r_eff = min(r, dW.shape[0], dW.shape[1])
            U, S, Vh = torch.linalg.svd(dW @ C_half, full_matrices=False)
            lora_B = (U[:, :r_eff] * S[:r_eff].sqrt().unsqueeze(0)).to(torch.bfloat16).cpu()
            lora_A = (S[:r_eff].sqrt().unsqueeze(1) * (Vh[:r_eff, :] @ C_inv_half)).to(torch.bfloat16).cpu()
            base_key = f"base_model.model.{layer_prefix}.{l}.{mname}"
            peft_state[f"{base_key}.lora_A.weight"] = lora_A
            peft_state[f"{base_key}.lora_B.weight"] = lora_B

    out_dir = Path(args.output_peft_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(peft_state, out_dir / "adapter_model.bin")

    adapter_cfg = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": art["model"],
        "r": r,
        "lora_alpha": r,
        "lora_dropout": 0.0,
        "bias": "none",
        "target_modules": arch_info["target_leaf_names"],
    }
    (out_dir / "adapter_config.json").write_text(json.dumps(adapter_cfg, indent=2), encoding="utf-8")
    print(f"[Step 3] Exported Functional-SVD Rank-{r} PEFT LoRA adapter to {out_dir} in {time.time()-t0:.1f}s.")

if __name__ == "__main__":
    main()
