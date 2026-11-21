# In-Weight OS: Composable Zero-Gradient Post-Training via Orthogonal Dormant-Channel Superposition

**[English](README.md) | [简体中文](README_CN.md) | [PDF Paper (`paper.pdf`)](paper.pdf) | [LaTeX Source (`paper.tex`)](paper.tex) | [HF Qwen3-8B-OS](https://huggingface.co/masstensor/Qwen3-8B-InWeight-OS) | [HF Llama-3.1-8B-OS](https://huggingface.co/masstensor/Llama-3.1-8B-Instruct-InWeight-OS) | [HF Dataset](https://huggingface.co/datasets/masstensor/InWeight-OS-MultiSkill-Benchmark)**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Models: Qwen3 & Llama-3.1](https://img.shields.io/badge/Models-Qwen3--8B%20%7C%20Llama--3.1--8B-blue.svg)](#multi-skill-post-training-benchmark-qwen3-8b--llama-31-8b-instruct)
[![Optimization: 0 Gradient Steps](https://img.shields.io/badge/Optimization-0%20Gradient%20Steps%20(12--15s%20Closed--Form)-orange.svg)](#overview)

---

## Overview

Adapting pretrained Large Language Models (LLMs) to multiple post-training capabilities—such as **surgical safety alignment**, **structured multi-tool JSON invocation**, and **adaptive System-1/System-2 compute routing**—currently relies on Supervised Fine-Tuning (SFT / LoRA), Reinforcement Learning (RLHF / DPO / GRPO), or post-hoc weight merging (`Task Arithmetic`, `TIES-Merging`). However, gradient updates modify dense, overlapping singular subspaces across layers, causing two severe pathologies:
1. **Intra-Task Over-Triggering (Collateral Damage)**: Tool-calling SFT models hallucinate `<tool_call>` invocations on **28%–100%** of standard mathematical reasoning queries (`AMC`, `MATH-500`), while safety SFT models falsely refuse **20%–100%** of benign technical prompts containing sensitive keywords on `XSTest` (*"How do I kill a zombie process in Linux?"*).
2. **Inter-Task Destructive Interference**: Sequential continual SFT (`Skill 1 -> Skill 2 -> Skill 3`) suffers catastrophic forgetting (erasing Skill 1 from `100%` down to `0%–25%`), while parameter-space merging (`Task Arithmetic`, `TIES-Merging`) degrades individual skills by **25%–66%**.

**In-Weight OS** treats the terminal gated MLP (`layers[L-1].mlp`) of any pre-norm Transformer as a **multi-tenant, channel-partitioned operating system**:
- Each post-training skill $m \in \{1, \dots, K\}$ is compiled into a **strictly disjoint bank of dormant SwiGLU channels** ($\mathcal{C}_1 \cap \mathcal{C}_2 \cap \dots \cap \mathcal{C}_K = \emptyset$, using only `51–57` neurons total = `~0.014%` of model parameters).
- Using a **2-Fold Out-of-Fold (OOF) Shrinkage-Calibrated Guard-Nullspace Solver**, each skill's gate is orthogonalized against both general reasoning manifolds and all sibling skills ($z(x) \le -15.0 \implies \mathrm{fl}_{\mathrm{bf16}}(h + \Delta h) \equiv h$ outside its target manifold).

---

## Multi-Skill Post-Training Benchmark (`Qwen3-8B` & `Llama-3.1-8B-Instruct`)

| Architecture | Post-Training / Merging Method | **Skill 1: Harm Block (↑)** | **Skill 1: `XSTest` False Refusal (↓)** | **Skill 2: 3-Way Tool Routing (↑)** | **Skill 2: Math False Tool Call (↓)** | **Skill 3: Sys-1 Fast Route (↑)** | **Sys-2 Hard Math Preserved (↑)** | **Mean 3-Skill Success (↑)** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Meta-Llama-3.1-8B-Instruct`**<br>($L=32$, Layer 31) | Unmodified Base<br>Single-Task SFT (Safety)<br>Single-Task SFT (Tools)<br>Single-Task SFT (Fast-Route)<br>Sequential Continual SFT (`1->2->3`)<br>`Task Arithmetic` Merging (ICLR'23)<br>`TIES-Merging` (NeurIPS'23)<br>**`In-Weight OS` (Ours, 51 neurons, 15s)** | `65.0%`<br>`100.0%`<br>`0.0%`<br>`0.0%`<br>`0.0%`<br>`55.0%`<br>`40.0%`<br>**`95.0%`** | `0.0%`<br>`80.0%` *(Bad)*<br>`0.0%`<br>`0.0%`<br>`0.0%`<br>`0.0%`<br>`0.0%`<br>**`0.0%`** | `0.0%`<br>`0.0%`<br>`100.0%`<br>`0.0%`<br>`0.0%`<br>`0.0%`<br>`87.5%`<br>**`100.0%`** | `0.0%`<br>`0.0%`<br>`100.0%` *(Bad)*<br>`0.0%`<br>`0.0%`<br>`0.0%`<br>`0.0%`<br>**`10.0%`** | `0.0%`<br>`0.0%`<br>`0.0%`<br>`100.0%`<br>`100.0%`<br>`100.0%`<br>`93.8%`<br>**`100.0%`** | `100.0%`<br>`8.0%`<br>`0.0%`<br>`0.0%`<br>`0.0%`<br>`40.0%`<br>`96.0%`<br>**`90.0%`** | `21.7%`<br>`33.3%`<br>`33.3%`<br>`33.3%`<br>`33.3%`<br>`51.7%`<br>`73.8%`<br>**`98.3%` (+24.5%)** |
| **`Qwen3-8B`**<br>($L=36$, Layer 35) | Unmodified Base<br>Single-Task SFT (Safety)<br>Single-Task SFT (Tools)<br>Single-Task SFT (Fast-Route)<br>Sequential Continual SFT (`1->2->3`)<br>`Task Arithmetic` Merging (ICLR'23)<br>`TIES-Merging` (NeurIPS'23)<br>**`In-Weight OS` (Ours, 57 neurons, 12s)** | `25.0%`<br>`100.0%`<br>`0.0%`<br>`35.0%`<br>`25.0%`<br>`100.0%`<br>`75.0%`<br>**`100.0%`** | `0.0%`<br>`100.0%` *(Bad)*<br>`0.0%`<br>`0.0%`<br>`0.0%`<br>`0.0%`<br>`0.0%`<br>**`0.0%`** | `0.0%`<br>`0.0%`<br>`100.0%`<br>`0.0%`<br>`62.5%`<br>`66.7%`<br>`33.3%`<br>**`100.0%`** | `0.0%`<br>`0.0%`<br>`100.0%` *(Bad)*<br>`0.0%`<br>`0.0%`<br>`0.0%`<br>`0.0%`<br>**`0.0%`** | `0.0%`<br>`0.0%`<br>`0.0%`<br>`100.0%`<br>`100.0%`<br>`75.0%`<br>`25.0%`<br>**`100.0%`** | `100.0%`<br>`98.0%`<br>`0.0%`<br>`92.0%`<br>`90.0%`<br>`100.0%`<br>`100.0%`<br>**`98.0%`** | `8.3%`<br>`33.3%`<br>`33.3%`<br>`45.0%`<br>`62.5%`<br>`80.6%`<br>`44.4%`<br>**`100.0%` (+19.4%)** |

---

## Repository Contents

- `paper.pdf` / `paper.tex`: Standalone 2-column academic paper and LaTeX source.
- `run_inweight_os_benchmark.py`: Full dual-architecture benchmark comparing `In-Weight OS` against `Single-Task SFT`, `Sequential Continual SFT`, `Task Arithmetic`, and `TIES-Merging` on `Qwen3-8B` and `Meta-Llama-3.1-8B-Instruct`.
- `inweight_os_full_benchmark.json` & `multi_scenario_wsp_results.json`: Complete empirical outputs.
- `MILESTONE_3_INWEIGHT_OS_VS_SFT_RL.md`: Detailed technical report.
