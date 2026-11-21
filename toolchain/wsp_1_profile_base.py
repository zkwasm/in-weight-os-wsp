#!/usr/bin/env python3
"""
Step 1 of Universal WSP Toolchain: Base Model General-Capability Nullspace Profiling
====================================================================================
Runs a read-only forward pass over a full-rank general corpus (encyclopedic passages
at all token positions with weight=1.0 + multi-domain reasoning prompts at the answer
decision boundary with weight=8.0) to extract the 4 shared input-space Gram covariance
matrices per Transformer layer:
  - `qkv`  (shared input to self_attn.q_proj, k_proj, v_proj)
  - `o`    (input to self_attn.o_proj)
  - `gu`   (shared input to mlp.gate_proj, up_proj)
  - `down` (input to mlp.down_proj)
and precomputes:
  1) Soft Ridge Nullspace Projector: Pi_gen = (I + mu_null * C / (tr(C)/d))^{-1}
  2) Orthonormal Top Eigenspace Basis: U_gen covering `hard_energy` (default 0.80)
"""
import argparse
import json
import time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Built-in diverse encyclopedic + technical passages and reasoning prompts ensuring well-conditioned covariance
BUILTIN_WIKI_PASSAGES = [
    "In linear algebra, the singular value decomposition (SVD) is a factorization of a real or complex matrix that generalizes the eigendecomposition of a square normal matrix to any m x n matrix via an extension of the polar decomposition. Specifically, the singular value decomposition of an m x n complex matrix M is a factorization of the form U Sigma V*, where U is an m x m complex unitary matrix, Sigma is an m x n rectangular diagonal matrix with non-negative real numbers on the diagonal, and V is an n x n complex unitary matrix.",
    "Photosynthesis is a biological process used by many cellular organisms to convert light energy into chemical energy, which is stored in organic compounds that can later be metabolized through cellular respiration to fuel the organism's activities. The process occurs in chloroplasts in plants and algae, where chlorophyll pigments absorb photons to drive the light-dependent reactions, splitting water molecules and generating ATP and NADPH for the Calvin cycle.",
    "The Transmission Control Protocol (TCP) is one of the main protocols of the Internet protocol suite. It originated in the initial network implementation in which it complemented the Internet Protocol (IP). TCP provides reliable, ordered, and error-checked delivery of a stream of octets between applications running on hosts communicating via an IP network, utilizing sequence numbers, acknowledgments, sliding windows, and congestion control algorithms.",
    "Thermodynamics is a branch of physics that deals with heat, work, and temperature, and their relation to energy, entropy, and the physical properties of matter and radiation. The behavior of these quantities is governed by the four laws of thermodynamics, which convey a quantitative description using measurable macroscopic physical quantities, but may be explained in terms of microscopic constituents by statistical mechanics.",
    "A compiler is a computer program that translates computer code written in one programming language (the source language) into another language (the target language). A typical compiler performs lexical analysis, preprocessing, parsing, semantic analysis, conversion of input programs to an intermediate representation (IR), register allocation, instruction scheduling, and target-specific code optimization.",
    "In probability theory and statistics, Bayes' theorem describes the probability of an event, based on prior knowledge of conditions that might be related to the event. Formally, P(A|B) = P(B|A)P(A) / P(B). In Bayesian inference, the theorem is used to update the probability distribution over hypotheses as more evidence or information becomes available.",
    "Plate tectonics is the scientific theory that Earth's lithosphere comprises a number of large tectonic plates which have been slowly moving since about 3.4 billion years ago. Where the plates meet, their relative motion determines the type of plate boundary: convergent, divergent, or transform. Earthquakes, volcanic activity, mountain-building, and oceanic trench formation occur along these plate boundaries.",
    "Operating systems manage computer hardware and software resources and provide common services for computer programs. Key kernel subsystems include virtual memory management via page tables and translation lookaside buffers (TLBs), preemptive process scheduling, inter-process communication (IPC), file system abstraction layers, and interrupt-driven device drivers.",
]

BUILTIN_REASONING_PROMPTS = [
    "Question: Which subatomic particle carries a negative electrical charge?\nA) Proton\nB) Neutron\nC) Electron\nD) Photon\nAnswer:",
    "Question: If a train travels at 90 km/h for 2.5 hours, what distance does it cover?\nA) 180 km\nB) 225 km\nC) 250 km\nD) 275 km\nAnswer:",
    "Question: Why does copper conduct electricity better than glass at room temperature?\nA) Copper has a higher density\nB) Copper has delocalized valence electrons in its conduction band\nC) Glass absorbs photons\nD) Copper has fewer protons\nAnswer:",
    "Question: In Python, what is the time complexity of looking up a key in a standard dict on average?\nA) O(1)\nB) O(log n)\nC) O(n)\nD) O(n log n)\nAnswer:",
    "Question: What is the derivative of f(x) = 3x^4 - 5x^2 + 7 with respect to x?\nA) 12x^3 - 10x\nB) 12x^3 - 5x + 7\nC) 3x^3 - 10x\nD) 4x^3 - 10x\nAnswer:",
    "Question: Which organelle is primarily responsible for synthesizing proteins from mRNA templates?\nA) Lysosome\nB) Ribosome\nC) Golgi apparatus\nD) Peroxisome\nAnswer:",
    "Question: If an algorithm divides the problem size in half at each step with O(1) work per step, what is its recurrence complexity?\nA) O(n)\nB) O(log n)\nC) O(n^2)\nD) O(2^n)\nAnswer:",
    "Question: Which gas makes up approximately 78% of Earth's dry atmosphere by volume?\nA) Oxygen\nB) Carbon dioxide\nC) Argon\nD) Nitrogen\nAnswer:",
]

def main():
    parser = argparse.ArgumentParser(description="Step 1: Profile Base Model General-Capability Nullspace")
    parser.add_argument("--model", type=str, required=True, help="Hugging Face model ID or local path")
    parser.add_argument("--output_profile", type=str, required=True, help="Output .pt path for nullspace profile")
    parser.add_argument("--general_jsonl", type=str, default=None, help="Optional JSONL with {'text': ...} general texts")
    parser.add_argument("--reasoning_jsonl", type=str, default=None, help="Optional JSONL with {'prompt': ...} reasoning prompts")
    parser.add_argument("--mu_null", type=float, default=3.0, help="Ridge nullspace penalty weight mu (default: 3.0)")
    parser.add_argument("--hard_energy", type=float, default=0.80, help="Top spectral energy fraction for U_gen (default: 0.80)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    t0 = time.time()
    print(f"[Step 1] Loading base model: {args.model} on {args.device}...")
    tok = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(args.device).eval()

    wiki_texts = list(BUILTIN_WIKI_PASSAGES)
    if args.general_jsonl and Path(args.general_jsonl).exists():
        for line in Path(args.general_jsonl).read_text(encoding="utf-8").splitlines():
            if line.strip():
                obj = json.loads(line)
                wiki_texts.append(obj.get("text") or obj.get("prompt") or str(obj))

    arc_prompts = list(BUILTIN_REASONING_PROMPTS)
    if args.reasoning_jsonl and Path(args.reasoning_jsonl).exists():
        for line in Path(args.reasoning_jsonl).read_text(encoding="utf-8").splitlines():
            if line.strip():
                obj = json.loads(line)
                arc_prompts.append(obj.get("prompt") or obj.get("text") or str(obj))

    reps_map = {
        "qkv": lambda lay: lay.self_attn.q_proj,
        "o": lambda lay: lay.self_attn.o_proj,
        "gu": lambda lay: lay.mlp.gate_proj,
        "down": lambda lay: lay.mlp.down_proj,
    }

    cov_cpu = {}
    state = {"last_only": False, "weight": 1.0}
    hs = []
    for l, lay in enumerate(model.model.layers):
        for rname, getter in reps_map.items():
            def make_hk(key):
                def f(_, inp):
                    x = inp[0][0].detach().float()
                    if state["last_only"]:
                        x = x[-1:]
                    c = state["weight"] * (x.T @ x).cpu()
                    cov_cpu[key] = cov_cpu[key] + c if key in cov_cpu else c
                return f
            hs.append(getter(lay).register_forward_pre_hook(make_hk((l, rname))))

    with torch.no_grad():
        state.update(last_only=False, weight=1.0)
        for txt in wiki_texts:
            ids = tok(txt, return_tensors="pt").input_ids.to(args.device)
            model(input_ids=ids, use_cache=False)

        state.update(last_only=True, weight=8.0)
        for p_txt in arc_prompts:
            ids = tok(p_txt, return_tensors="pt").input_ids.to(args.device)
            model(input_ids=ids, use_cache=False)

    for h in hs:
        h.remove()

    Pi_cpu, U_cpu = {}, {}
    for key, C in cov_cpu.items():
        c = C.to(args.device)
        d = c.shape[0]
        # Add small isotropic floor so covariance is strictly positive-definite even on small calibration sets
        c_reg = c + 1e-3 * (c.trace() / d + 1e-8) * torch.eye(d, device=c.device)
        reg = torch.eye(d, device=c.device) + args.mu_null * (c_reg / (c_reg.trace() / d + 1e-8))
        Pi_cpu[key] = torch.linalg.inv(reg).cpu()
        val, vec = torch.linalg.eigh(c_reg)
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
