# Milestone 3 Report: From Single-Task State Machine to Multi-Skill "In-Weight OS" — Rivalry with Fine-Tuning (SFT) and RL Across Four Post-Training Scenarios

## 1. Executive Summary

Following the completion of our 3,780-dialogue Dialectical Reasoning study, we investigated whether **Zero-KV-Drift Pure In-Weight Weight Space Programming (Pure WSP)** can serve as a general-purpose replacement for **Supervised Fine-Tuning (SFT / LoRA)**, **Safety Alignment (DPO / RLHF / RepE)**, **Length-Penalized RL (GRPO)**, and **Multi-Task Model Merging (Task Arithmetic)**.

Using the **Out-of-Fold (OOF) High-Shrinkage Guard-Nullspace Compiler** (`run_wsp_multi_scenario_suite.py`), we compiled three new post-training capabilities directly into disjoint dormant SwiGLU channels of `Qwen3-8B` Layer 35 (`layers[35].mlp`) in **6–10 seconds each with zero gradient steps**, and evaluated them both individually and **simultaneously superposed inside a single checkpoint ("In-Weight OS")** against 25-step `AdamW` Gradient SFT and Task-Arithmetic Weight Merging (`multi_scenario_wsp_results.json`):

| Post-Training Scenario | Method | Compile / Train Time | Edited Params | Primary Task Success (Unseen Test) | Side-Effect / Collateral Damage (↓) | Multi-Skill Merged Performance |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Scenario 1: Surgical Safety & Jailbreak Defense**<br>*(vs. `XSTest` Benign Trigger Lookalikes)* | Unedited Base<br>25-Step Gradient SFT<br>**Pure WSP (Ours)** | —<br>`2.3s` *(25 grad)*<br>**`9.2s` *(0 grad)*** | `0`<br>`566M`<br>**`15 neurons`** | `25.0%` Harm Block<br>`100.0%` Harm Block<br>**`100.0%` Harm Block** | `0.0%` `XSTest` False Refusal<br><span style="color:red">`20.0%` `XSTest` False Refusal</span><br>**`0.0%` `XSTest` False Refusal** | —<br>`90.0%` Safety F1<br>**`100.0%` Safety F1** |
| **Scenario 2: Multi-Tool Function Calling & JSON Schema**<br>*(`sql_engine` / `python_exec` / `web_search`)* | Unedited Base<br>25-Step Gradient SFT<br>**Pure WSP (Ours)** | —<br>`2.3s` *(25 grad)*<br>**`9.8s` *(0 grad)*** | `0`<br>`566M`<br>**`34 neurons`** | `0.0%` Tool Routing<br>`100.0%` Tool Routing<br>**`100.0%` Tool Routing** | `0.0%` General QA False Call<br><span style="color:red">`28.0%` General QA False Call</span><br>**`0.0%` General QA False Call** | —<br><span style="color:red">Drops to `75.0%`</span><br>**Stays `100.0%`** |
| **Scenario 3: System-1 / System-2 Compute Router**<br>*(Overthinking Elimination on Fast Queries)* | Unedited Base<br>**Pure WSP (Ours)** | —<br>**`6.0s` *(0 grad)*** | `0`<br>**`8 neurons`** | `0.0%` Fast Route (`348.5` tok)<br>**`100.0%` Fast Route (`19.1` tok, `18.3×` faster)** | —<br>Preserves System-2 Deep CoT | —<br>**`25.7×` Token Reduction (`13.6` tok)** |
| **Scenario 4: Multi-Skill "In-Weight OS" Superposition**<br>*(All Skills Loaded Simultaneously)* | **Task-Arithmetic SFT Merge**<br>**Pure WSP Orthogonal OS** | `4.6s` *(50 grad)*<br>**`25.0s` *(0 grad)*** | `566M`<br>**`57 neurons` (`0.014%`)** | Safety `100%`, **Tools `75.0%`**<br>**Safety `100%`, Tools `100.0%`, Fast-Route `100%`** | **`20.0%` `XSTest` Over-Refusal**<br>**`0.0%` Over-Refusal & `0.0%` False Tool Calls** | **Destructive Interference (-25%)**<br>**Zero Interference (`100.0%` All)** |

---

## 2. Key Technical Discovery: Out-of-Fold (OOF) Shrinkage Calibration for Few-Shot Intent Gates

When compiling an intent-triggered state machine (such as detecting a jailbreak or a tool-call request at `<|im_start|>assistant
`) from only $N_+ = 10	ext{--}16$ calibration sentences in $d_{\mathrm{model}} = 4096$, an unregularized sample covariance overfits to the finite calibration span, causing unseen test projections $\langle k, x_{	ext{test}} angle$ to shrink by $\sim 0.5	imes$ relative to in-sample projections $\langle k, x_{	ext{train}} angle$.

We solved this via **Class-Balanced High-Shrinkage Guard-Nullspace Estimation + 2-Fold Out-of-Fold Threshold Calibration** (`solve_oof_entry_gate` in `run_wsp_multi_scenario_suite.py`):
1. **Semantic vs. Mid-FSM Negative Balancing**: At prompt entry (`idx_s == 0`), we weight semantic negative prompt-end vectors ($X_{	ext{neg,sem}}$, e.g., `XSTest` benign technical prompts and general QA) at `85%` and mid-FSM token states ($X_{	ext{other,fsm}}$) at `50%`, preventing the hundreds of mid-FSM tokens from drowning out fine-grained prompt-end intent boundaries.
2. **2-Fold Out-of-Fold Shrinkage Ratio**: We compute 2-fold out-of-fold projections $	ext{oof}_p$ on the calibration split to measure the exact finite-sample shrinkage factor $ho = \mathbb{E}[	ext{oof}_p] / \mathbb{E}[	ext{in}_p]$, setting $p_{	ext{low}} = ho \cdot Q_{0.10}(	ext{in}_p)$. This places the SwiGLU gate threshold $	heta_s$ strictly inside the true out-of-distribution margin.

---

## 3. Why Pure WSP Beats Gradient SFT & LoRA Merging in Multi-Skill Deployment

1. **Zero Catastrophic Over-Triggering (`0.0%` vs. `28.0%`)**:
   - Gradient SFT updates dense singular directions across layers; after just 25 AdamW steps on 30 tool-calling examples, `Qwen3-8B` hallucinates `<tool_call>` on **28.0% (`14 / 50`)** of standard math and reasoning questions.
   - Pure WSP explicitly projects the tool-entry gate into the orthogonal nullspace of general QA prompts and keeps all dormant neurons at $z(x) \le -15.0$ ($\mathrm{fl}_{\mathrm{bf16}}(h + \Delta h) \equiv h$) outside the tool manifold, achieving **`0.0%` false triggers**.
2. **Zero Cross-Skill Interference Under Superposition (`100.0%` vs. `75.0%`)**:
   - When merging a Safety SFT adapter and a Tool-Calling SFT adapter via Task Arithmetic ($0.5 \Delta W_{	ext{safe}} + 0.5 \Delta W_{	ext{tool}}$), the safety weights falsely refuse **25.0%** of legitimate SQL/Python tool queries (which contain technical verbs like `"dump"`, `"execute"`, `"query"`) and **20.0%** of `XSTest` benign technical questions.
   - In Pure WSP, each skill occupies **strictly disjoint dormant neuron indices** ($\mathcal{C}_{	ext{safe}} = \mathcal{D}_{35}[0:15]$, $\mathcal{C}_{	ext{tool}} = \mathcal{D}_{35}[30:64]$, $\mathcal{C}_{	ext{sys1}} = \mathcal{D}_{35}[80:88]$, $\mathcal{C}_i \cap \mathcal{C}_j = \emptyset$), and each skill's gate places the other skills' manifolds in its guard/negative set. Plugging all three bitstreams into Layer 35 simultaneously yields **100.0% accuracy across all skills with zero interference**.
