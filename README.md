# In-Weight OS: Composable Zero-Gradient Post-Training via Orthogonal Dormant-Channel Superposition

**[English](README.md) | [Architecture Spec (`WSP_V3_SENSOR_BEACON_ACTUATOR_DESIGN.md`)](WSP_V3_SENSOR_BEACON_ACTUATOR_DESIGN.md) | [PDF Paper (`paper.pdf`)](paper.pdf) | [LaTeX Source (`paper.tex`)](paper.tex) | [HF Qwen3-8B-OS](https://huggingface.co/masstensor/Qwen3-8B-InWeight-OS) | [HF Llama-3.1-8B-OS](https://huggingface.co/masstensor/Llama-3.1-8B-Instruct-InWeight-OS) | [HF Dataset](https://huggingface.co/datasets/masstensor/InWeight-OS-MultiSkill-Benchmark)**

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

## Scaled 660-Instance Benchmark & Concurrent Skill Capacity Scaling ($k = 1 \to 4$ Heterogeneous Skills)

To address three core questions—(1) extending beyond routing switches to **non-routing novel domain knowledge & symbolic operator injection**, (2) eliminating **boundary leakage on static Python code comprehension (`CRUXEval-O`)**, and (3) scaling evaluation size by $10\times$—we introduce **Dual-Gate Conjunctive Hard-Negative Orthogonalization** (`run_inweight_os_v2_scaled.py`, `inweight_os_v2_scaled_results.json`):

- **Skill 4 (Novel Domain Knowledge & Symbolic Operator Associative Memory, $N=60$ Unseen Queries)**: Compiles three novel enterprise/scientific facts and custom algebraic operators absent from pretraining (**`0.0%` Base accuracy**) into `58` dormant channels ($\mathcal{C}_4$): (i) *Arkreen-v4 consensus parameters* (`72%` BFT quorum, `18s` epoch), (ii) *the custom Zenith-Nabla operator* ($a \nabla b = 4a - 3b + 9$), and (iii) *Helios-9 kernel fault `ERR-HX-Alpha` diagnostics*.
- **Dual-Gate Boundary Verification ($N=200$: `120` Hard Math + `80` `CRUXEval-O` Static Python Code Problems)**: Shapes both $w_{\mathrm{gate}}$ and $w_{\mathrm{up}}$ against balanced mathematical proofs and static code comprehension guards, reducing `Llama-3.1-8B`'s false tool-call rate on `CRUXEval-O` code problems from `10.0%` to **`0.0%` (`0 / 80`)**.

| Architecture | Concurrent Skills Active ($k$) | Active Neurons | **S1: Harm Block ($N=100$)** | **S1: `XSTest` False Ref ($N=100$)** | **S2: 3-Way Tool Acc ($N=120$)** | **S2: Math False Tool ($N=120$)** | **S2: `CRUXEval` Code False Tool ($N=80$)** | **S4: Novel Knowledge Recall ($N=60$)** | **Sys-2 Math & Code Preserved ($N=200$)** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Meta-Llama-3.1-8B-Instruct`**<br>($L=32$, Layer 31) | Unmodified Base ($k=0$)<br>$k=1$: Safety Only ($\mathcal{C}_1$)<br>$k=2$: + 3-Way Tools ($\mathcal{C}_1 \cup \mathcal{C}_2$)<br>$k=3$: + Sys-1 Fast Router ($\bigcup_{m=1}^3 \mathcal{C}_m$)<br>**$k=4$: + Novel Knowledge ($\bigcup_{m=1}^4 \mathcal{C}_m$)** | `0`<br>`15`<br>`45`<br>`51`<br>**`109`** | `57.0%`<br>**`99.0%`**<br>**`99.0%`**<br>**`99.0%`**<br>**`99.0%`** | `0.0%`<br>`5.0%`<br>`6.0%`<br>**`3.0%`**<br>**`3.0%`** | `0.0%`<br>—<br>**`100.0%`**<br>**`100.0%`**<br>**`100.0%`** | `0.0%`<br>`0.0%`<br>**`0.0%`**<br>**`0.0%`**<br>**`0.0%`** | `0.0%`<br>`0.0%`<br>**`0.0%`**<br>**`0.0%`**<br>**`0.0%` (0/80)** | `0.0%`<br>—<br>—<br>—<br>**`100.0%` (60/60)** | `100.0%`<br>`94.5%`<br>`94.5%`<br>`94.5%`<br>**`100.0%` (200/200)** |
| **`Qwen3-8B`**<br>($L=36$, Layer 35) | Unmodified Base ($k=0$)<br>$k=1$: Safety Only ($\mathcal{C}_1$)<br>$k=2$: + 3-Way Tools ($\mathcal{C}_1 \cup \mathcal{C}_2$)<br>$k=3$: + Sys-1 Fast Router ($\bigcup_{m=1}^3 \mathcal{C}_m$)<br>**$k=4$: + Novel Knowledge ($\bigcup_{m=1}^4 \mathcal{C}_m$)** | `0`<br>`15`<br>`45`<br>`51`<br>**`109`** | `71.0%`<br>**`100.0%`**<br>**`100.0%`**<br>**`100.0%`**<br>**`100.0%`** | `0.0%`<br>**`0.0%`**<br>**`0.0%`**<br>**`0.0%`**<br>**`0.0%`** | `0.0%`<br>—<br>**`99.2%`**<br>**`99.2%`**<br>**`99.2%`** | `0.0%`<br>`0.0%`<br>**`0.0%`**<br>**`0.0%`**<br>**`0.0%`** | `0.0%`<br>`0.0%`<br>**`0.0%`**<br>**`0.0%`**<br>**`0.0%` (0/80)** | `0.0%`<br>—<br>—<br>—<br>**`100.0%` (60/60)** | `100.0%`<br>`100.0%`<br>`100.0%`<br>`99.5%`<br>**`99.5%` (199/200)** |


---

## Part III: Beyond Static Switches — Multi-Hop Numerical Reasoning & Dynamic Tool Argument Generation (`inweight_os_v3_results.json`)

By combining **In-Weight Premise-Recall** with **Stem-Release** (placing all post-header tokens in $\mathcal{X}_-$ so $z(x) \le -15.0$ and $\mathrm{fl}_{\mathrm{bf16}}(h + \Delta h) \equiv h$ immediately after the axiom or schema prefix is emitted), **In-Weight OS v3.0** hands control back to the base model's native arithmetic and code-synthesis circuits to solve **unseen multi-hop mathematical word problems** and **dynamically extract SQL / Python / Search arguments**:

| Model | Generative / Multi-Hop Task | Unedited Base | **In-Weight OS v3.0 (Ours)** | Verbatim Generated Output Example |
| :--- | :--- | :---: | :---: | :--- |
| **`Qwen3-8B`** | **Zenith-Nabla Multi-Step Algebra** $(a \nabla b) + c$ *(20 unseen $(a,b,c)$ triples)*<br>**Arkreen-v4 Multi-Epoch & BFT Quorum Math** *(20 unseen word problems)*<br>**Helios-9 Fault Remaining Capacity** $12S - 3F$ *(20 unseen word problems)*<br>**Total 60 Unseen Multi-Hop Numerical Problems**<br>**Dynamic Tool Routing + Argument Grounding** *(30 unseen queries)*<br>**Hard Math & `CRUXEval` Code False Triggers** *(80 guard problems)* | `0.0%` (0/20)<br>`0.0%` (0/20)<br>`0.0%` (0/20)<br>`0.0%` (0/60)<br>`0.0%` (0/30)<br>`0.0%` (0/80) | **`85.0%` (17/20)**<br>**`100.0%` (20/20)**<br>**`95.0%` (19/20)**<br>**`93.3%` (56/60)**<br>**`90.0%` (27/30)**<br>**`0.0%` (0/80)** | `Recall_Rule: [Zenith Nabla maps a,b to four*a minus triple*b plus nine]. Step 1: Zenith Nabla(9, 2) = 4*9 - 3*2 + 9 = 39. Step 2: 39 + (-4) = 35. _final_answer_: (35)`<br>`{"tool":"sql_engine","stmt": "SELECT email, COUNT(*) as count FROM newsletter_subscribers GROUP BY email HAVING COUNT(*) > 1;"}` |
| **`Llama-3.1-8B`** | **Zenith-Nabla Multi-Step Algebra** $(a \nabla b) + c$ *(20 unseen triples)*<br>**Arkreen-v4 Multi-Epoch & BFT Quorum Math** *(20 unseen word problems)*<br>**Dynamic Tool Routing + Argument Grounding** *(30 unseen queries)*<br>**Hard Math & `CRUXEval` Code False Triggers** *(80 guard problems)* | `0.0%` (0/20)<br>`0.0%` (0/20)<br>`0.0%` (0/30)<br>`0.0%` (0/80) | **`75.0%` (15/20)**<br>**`100.0%` (20/20)**<br>**`83.3%` (25/30)**<br>**`0.0%` (0/80)** | `Recall_Rule: [Zenith Nabla maps a,b to four*a minus triple*b plus nine]. Step=1: (11 ∇ 6) = 4*11 - 3*6 + 9 = 35. Step=2: 35 + 2 = 37.`<br>`{"tool":"sql_engine","stmt": "SELECT email FROM newsletter_subscribers GROUP BY email HAVING COUNT(email) > 1"}` |


---

## Part IV: Universal Dataset-to-Weight Nullspace Compiler & 4-Step CLI Toolchain (Nullspace-Projected Post-Training & Closed-Form LoRA/Fusion Across 4,800 Unseen Test Instances)

Whereas Parts I–III and V compile sparse skill cartridges into dormant MLP channels in **strictly zero gradient steps**, Part IV provides a general-purpose **Nullspace-Projected Dataset Compiler (`toolchain/`, `run_universal_compiler_exp.py`, `universal_wsp_vs_sft_lora.json`)** for dense multi-layer knowledge adaptation on **any base model + any supervised `.jsonl` dataset $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$** (where every example has a distinct multi-parameter output, with **zero pre-trained teacher models** and **zero hand-coded FSM templates**):

1. **Step 1 (`toolchain/wsp_1_profile_base.py`, Zero-Gradient)**: Profiles the base model's general protection covariance $C_{\mathrm{gen}}$ (`WikiText` + `ARC`) across the 4 shared input spaces per layer (`qkv`, `o`, `gu`, `down`), precomputing $\Pi_{\mathrm{gen}} = (I + \mu \tilde{C}_{\mathrm{gen}})^{-1}$ and top eigenspace $U_{\mathrm{gen}}$.
2. **Step 2 (`toolchain/wsp_2_compile_dataset.py`, Nullspace-Projected Post-Training)**: Compiles any `.jsonl` dataset by right-multiplying layer updates by $\Pi_{\mathrm{gen}}$ and clamping cumulative weight deltas out of $U_{\mathrm{gen}}$. Supports **replay-free continual learning (`--continual_from`)** by recursively accumulating prior task input covariances $C_{\mathcal{D}_k}$.
3. **Step 3 (`toolchain/wsp_3_export_lora.py`, Zero-Gradient Functional-SVD)**: Exports standard Hugging Face `peft` LoRA adapters `(lora_A, lora_B)` in 25 seconds via **Covariance-Weighted Functional SVD** ($\Delta W C_{\mathrm{task}}^{1/2} = U_r \Sigma_r V_r^\top$).
4. **Step 4 (`toolchain/wsp_4_fuse_models.py`, Zero-Gradient Closed-Form Fusion)**: Fuses $K$ independently compiled dataset models in closed form ($\Delta W_l^* = (\sum_k D_l^{(k)} H_l^{(k)\top})(\sum_k H_l^{(k)} H_l^{(k)\top} + \mu' C_l + \lambda I)^{-1}$) with zero gradient steps.

👉 **See [`TOOLCHAIN_GUIDE.md`](./TOOLCHAIN_GUIDE.md) for the step-by-step CLI commands.**

### 4,800-Instance Benchmark (`energy`, `industrial`, `meter_query` — 1,600 Held-Out Test Dialogs Each)

| Evaluation Regime | Method | `energy`<br>*(N=1,600)* | `industrial`<br>*(N=1,600)* | `meter_query`<br>*(N=1,600)* | **3-Domain Mean**<br>*(N=4,800)* | **General `ARC-Easy`**<br>*(↑)* | **`WikiText KL` Drift**<br>*(↓)* |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Single-Dataset Compilation**<br>*(240 train examples per domain)* | Unedited Base<br>Standard Gradient LoRA (`r=16`)<br>Standard Full SFT<br>**Functional-SVD Compiled LoRA (`r=32`, Ours)**<br>**Nullspace-WSP Compiler (Ours)** | `8.50%`<br>`70.56%`<br>`76.88%`<br>**`71.00%`**<br>**`76.31%`** | `10.50%`<br>`71.06%`<br>`75.25%`<br>**`76.44%`**<br>**`77.94%`** | `14.62%`<br>`56.50%`<br>`70.06%`<br>**`62.50%`**<br>**`70.81%`** | `11.21%`<br>`66.04%`<br>`74.06%`<br>**`69.98% (+3.94% vs LoRA)`**<br>**`75.02% (Best)`** | `67.00%`<br>`66.83%`<br>`54.67% (-12.3%)`<br>**`61.00%`**<br>**`59.17%`** *(meter: `67.0%`)* | `0.00000`<br>`0.00484`<br>`0.03559`<br>**`0.00299`**<br>**`0.00406` (`8.8x` lower)** |
| **Multi-Dataset Model Fusion**<br>*(3 domains merged into 1 model)* | Standard `Task Arithmetic (SUM)`<br>**Closed-Form Multi-Dataset Aligned WSP (Ours)** | `35.56%`<br>**`74.00%`** | `52.56%`<br>**`76.12%`** | `41.81%`<br>**`67.63%`** | `43.31%`<br>**`72.58% (+29.27%)`** | `24.50%`<br>**`50.00% (+25.5%)`** | `0.19403`<br>**`0.04029` (`4.8x` lower)** |
| **Replay-Free Continual Learning**<br>*($\mathcal{D}_1 \to \mathcal{D}_2 \to \mathcal{D}_3$ sequentially)* | Standard Continual SFT<br>**Recursive Nullspace Continual WSP (Ours)** | `49.56%`<br>**`65.19%`** | `61.75%`<br>**`74.31%`** | `70.13%`<br>**`69.25%`** | `60.48%`<br>**`69.58% (+9.10%)`** | `25.50%`<br>**`46.00% (+20.5%)`** | `0.11389`<br>**`0.01561` (`7.3x` lower)** |

---

## Part V: Trajectory-Safe Universal Background Guard (`In-Weight OS v2.1`) & 2,180-Prompt Long-Generation Stress Verification

Why can a naive prompt-end linear gate leak or emit skill fragments in the middle of a 400-token `GSM8K` or `MATH-500` chain-of-thought? We identify and solve four fundamental geometric mechanisms in `rebuild_and_verify_inweight_os_universal.py` (with updated modular cartridges in `cartridges/{qwen3_8b,llama31_8b}/*.pt` and full evaluation logs in `independent_eval_verification/`):

1. **Template-Suffix Shortcut Elimination**: If calibration guard prompts contain an artificial formatting suffix (such as `_final_answer_: (<your answer>)`) absent from positive skill prompts, a linear solver learns to rely on that suffix and misfires on unadorned user prompts (`GSM8K`, `Dolly`, casual chat). We strip all artificial suffixes so positive and guard prompts share identical unadorned chat formatting.
2. **Full-Trajectory Mid-Generation Guarding with Separate Group Normalization**: Protecting only token $t=0$ (`<|im_start|>assistant\n`) leaves intermediate FSM neurons vulnerable when the assistant naturally writes colons `:`, parentheses `(12)`, or numbers at token $t \in [1, 420]$. We record 1,088–2,172 natural free-generation trajectory states $\mathcal{X}_{\mathrm{traj}}$ from the unedited base model and normalize prompt-boundary negatives ($\mathcal{X}_{\mathrm{prm}}$, weight $0.95 / \sqrt{|\mathcal{X}_{\mathrm{prm}}|}$) separately from mid-trajectory negatives ($\mathcal{X}_{\mathrm{traj}}$, weight $0.50 / \sqrt{|\mathcal{X}_{\mathrm{traj}}|}$) across every entry, branch, and mid-step neuron, while removing the `)`-triggered EOS neuron on Skill 3.
3. **Composite Residual-Carrier Bias Anchor**: Defining the synthetic bias direction as $d_{\mathrm{anchor}} = \operatorname{normalize}(0.55 \tilde{\mu}_+ + 0.45 \tilde{\mu}_{\mathrm{bg}})$ keeps $\langle b_{\mathrm{base}}, x \rangle \approx 1.0$ strictly well-conditioned across all mid-sentence tokens, enabling strict `max()` margin enforcement over 100% of background states.
4. **Per-Hyperplane Dual Thinking-Mode Calibration (`Qwen3-8B` `enable_thinking=False` & `True`)**: Because `enable_thinking=False` (after `</think>\n\n`) and `enable_thinking=True` (at `<|im_start|>assistant\n`) occupy parallel affine hyperplanes $\mathcal{H}_0, \mathcal{H}_1$, solving mode-specific entry neurons against same-hyperplane prompt negatives while placing opposite-hyperplane states in the guard covariance $G$ enables seamless skill activation in **both** non-thinking and thinking modes.

### Independent 2,180-Prompt Public Stress Suite (`GSM8K`, `MATH-500`, `Dolly`, `CRUXEval`, `MMLU`, `ARC-C`, `JBB`)

| Evaluation Metric / Public Benchmark | **`Qwen3-8B` Base** | **`Qwen3-8B` OS v1.0** *(Prompt-End Guard Only)* | **`Qwen3-8B` OS v3.0** *(Multi-Cone Trajectory-Safe)* | **`Llama-3.1-8B` Base** | **`Llama-3.1-8B` OS v1.0** *(Prompt-End Guard Only)* | **`Llama-3.1-8B` OS v3.0** *(Multi-Cone Trajectory-Safe)* |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`GSM8K` Long-Gen Accuracy** *(N=300, 420 tokens)* | `85.00%` | `32.33%` | **`86.67%` (+1.67% vs Base)** | `86.00%` | `0.00%` | **`85.00%` (Restored)** |
| **`GSM8K` Trigger Marker Contamination** *(N=300)* | `0.00%` | `94.33%` | **`0.33%` (1 / 300)** | `0.00%` | `100.00%` | **`3.00%` (`0/300` 1st-tok chg)** |
| **`MATH-500` Long-Gen Boxed Accuracy** *(N=250, 700 tokens)* | `52.40%` | `2.40%` | **`52.00%` (52.03% non-trig)** | `45.60%` | `~0.00%` | **`48.09%` (non-trig)** |
| **`MATH-500` False `Direct_Answer:` / `Refuse`** *(N=500)* | `0.0% / 0.0%` | `77.2% / 0.0%` | **`0.0% / 0.0%` (0 / 500)** | `0.0% / 0.0%` | `70.0% / 23.2%` | **`1.6% / 0.4%` (`0/500` 1st-tok)** |
| **`CRUXEval (200)` / `MMLU (300)` / `ARC-C (200)` False Markers** | `0.00%` | `15.00%+` | **`0.00%` (0 / 700)** | `0.00%` | `20.00%+` | **`0.00%` on MMLU & ARC-C (`0/500`)** |
| **`heldout_tool_neg` False Tool Triggers** *(N=30)* | `0.00%` | `10.00%+` | **`0.00%` (`sys`/`think`), `3.3%` (`plain`)** | `0.00%` | `10.00%` | **`0.00%` (`0/30` across `plain`/`sys`/`min`)** |
| **Paraphrased `heldout_sys1_para` Fast-Route Recall** *(N=20)* | `0.00%` | `85.00%` | **`70% (plain) / 80% (sys) / 90% (think)`** | `0.00%` | `95.00%` | **`65.0%` (`0.0%` on `sys1_hard`)** |
| **Paraphrased Held-Out Harm / Tool / Benign False-Ref** | — | `96% / 100% / 0%` | **`96% / 98.3–100% / 0.0%`** | — | `95% / 100% / 0%` | **`100% / 95% / 0.0%`** |
| **`Qwen3` Default Thinking Mode (`enable_thinking=True`)** | — | `0% Harm / 0% Tool / 54% Math Err` | **`96% Harm / 100% Tool (SQL 100, Py 100, Web 100) / 90% Sys1 / 0% Math Err`** | — | N/A | N/A |

---

## Repository Contents

- `paper.pdf` / `paper.tex`: Standalone 2-column academic paper and LaTeX source.
- `run_inweight_os_benchmark.py`: Full dual-architecture benchmark comparing `In-Weight OS` against `Single-Task SFT`, `Sequential Continual SFT`, `Task Arithmetic`, and `TIES-Merging` on `Qwen3-8B` and `Meta-Llama-3.1-8B-Instruct`.
- `inweight_os_full_benchmark.json` & `multi_scenario_wsp_results.json`: Complete empirical outputs.
- `MILESTONE_3_INWEIGHT_OS_VS_SFT_RL.md`: Detailed technical report.
