#!/usr/bin/env python3
"""
Step 1 of Universal WSP Toolchain: Base Model General-Capability Nullspace Profiling
====================================================================================
Runs a single read-only forward pass over general calibration text & reasoning prompts
to extract the 4 shared input-space Gram covariance matrices per Transformer layer:
  - `qkv`  (shared input to self_attn.q_proj, k_proj, v_proj)
  - `o`    (input to self_attn.o_proj)
  - `gu`   (shared input to mlp.gate_proj, up_proj)
  - `down` (input to mlp.down_proj)
and precomputes the Closed-Form Nullspace Projector Pi_gen = (I + mu * C_tilde)^{-1}
plus the top orthonormal general eigenspace basis U_gen.
"""
import argparse
import json
import time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_GENERAL_PROMPTS = [
    "Explain the fundamental theorem of calculus and how definite integrals relate to antiderivatives.",
    "Write a clean Python implementation of Dijkstra's shortest-path algorithm using a priority queue.",
    "Why does ice float on liquid water, and how does hydrogen bonding structure the crystal lattice?",
    "In distributed systems, explain how the Raft consensus algorithm elects a leader and replicates logs.",
    "Solve step by step: If 3x + 7 = 22, what is the value of 5x - 4?",
    "Describe the role of mitochondria in cellular respiration and ATP synthesis across eukaryotic cells.",
    "Compare TCP and UDP transport protocols in terms of reliability, ordering, and congestion control.",
    "How does garbage collection work in modern managed runtimes using generational mark-and-sweep?",
]

def main():
    parser = argparse.ArgumentParser(description="Step 1: Profile Base Model General-Capability Nullspace")
    parser.add_argument("--model", type=str, required=True, help="Hugging Face model ID or local path")
    parser.add_argument("--output_profile", type=str, required=True, help="Output .pt path for nullspace profile")
    parser.add_argument("--general_jsonl", type=str, default=None, help="Optional JSONL with {'text': ...} general texts")
    parser.add_argument("--mu_null", type=float, default=3.0, help="Ridge nullspace penalty weight mu")
    parser.add_argument("--hard_energy", type=float, default=0.80, help="Top spectral energy fraction for U_gen")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    t0 = time.time()
    print(f"[Step 1] Loading base model: {args.model} on {args.device}...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(args.device).eval()

    texts = list(DEFAULT_GENERAL_PROMPTS)
    if args.general_jsonl and Path(args.general_jsonl).exists():
        for line in Path(args.general_jsonl).read_text(encoding="utf-8").splitlines():
            if line.strip():
                obj = json.loads(line)
                texts.append(obj.get("text") or obj.get("prompt") or str(obj))

    reps_map = {
        "qkv": lambda lay: lay.self_attn.q_proj,
        "o": lambda lay: lay.self_attn.o_proj,
        "gu": lambda lay: lay.mlp.gate_proj,
        "down": lambda lay: lay.mlp.down_proj,
    }

    cov_cpu = {}
    hs = []
    for l, lay in enumerate(model.model.layers):
        for rname, getter in reps_map.items():
            def make_hk(key):
                def f(_, inp):
                    x = inp[0][0].detach().float()
                    c = (x.T @ x).cpu()
                    cov_cpu[key] = cov_cpu[key] + c if key in cov_cpu else c
                return f
            hs.append(getter(lay).register_forward_pre_hook(make_hk((l, rname))))

    with torch.no_grad():
        for txt in texts:
            ids = tok(txt, return_tensors="pt").input_ids.to(args.device)
            model(input_ids=ids, use_cache=False)

    for h in hs:
        h.remove()

    Pi_cpu, U_cpu = {}, {}
    for key, C in cov_cpu.items():
        c = C.to(args.device)
        d = c.shape[0]
        reg = torch.eye(d, device=c.device) + args.mu_null * (c / (c.trace() / d + 1e-8))
        Pi_cpu[key] = torch.linalg.inv(reg).cpu()
        val, vec = torch.linalg.eigh(c)
        val, vec = val.flip(0), vec.flip(1)
        r = int((val.cumsum(0) / val.sum() < args.hard_energy).sum()) + 1
        U_cpu[key] = vec[:, :r].contiguous().cpu()

    out_path = Path(args.output_profile)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": args.model,
        "mu_null": args.mu_null,
        "hard_energy": args.hard_energy,
        "cov_gen": cov_cpu,
        "Pi_gen": Pi_cpu,
        "U_gen": U_cpu,
    }, out_path)
    print(f"[Step 1] Saved general-capability nullspace profile ({len(Pi_cpu)} matrices) to {out_path} in {time.time()-t0:.1f}s.")

if __name__ == "__main__":
    main()
