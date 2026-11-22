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
`) from only $N_+ = 10	ext{--}16$ calibration sentences in $d_{\mathrm{model}} = 4096$, an unregularized sample covariance overfits to the finite calibration span, causing unseen test projections $\langle k, x_{	ext{test}} 
angle$ to shrink by $\sim 0.5	imes$ relative to in-sample projections $\langle k, x_{	ext{train}} 
angle$.

We solved this via **Class-Balanced High-Shrinkage Guard-Nullspace Estimation + 2-Fold Out-of-Fold Threshold Calibration** (`solve_oof_entry_gate` in `run_wsp_multi_scenario_suite.py`):
1. **Semantic vs. Mid-FSM Negative Balancing**: At prompt entry (`idx_s == 0`), we weight semantic negative prompt-end vectors ($X_{	ext{neg,sem}}$, e.g., `XSTest` benign technical prompts and general QA) at `85%` and mid-FSM token states ($X_{	ext{other,fsm}}$) at `50%`, preventing the hundreds of mid-FSM tokens from drowning out fine-grained prompt-end intent boundaries.
2. **2-Fold Out-of-Fold Shrinkage Ratio**: We compute 2-fold out-of-fold projections $	ext{oof}_p$ on the calibration split to measure the exact finite-sample shrinkage factor $
ho = \mathbb{E}[	ext{oof}_p] / \mathbb{E}[	ext{in}_p]$, setting $p_{	ext{low}} = 
ho \cdot Q_{0.10}(	ext{in}_p)$. This places the SwiGLU gate threshold $	heta_s$ strictly inside the true out-of-distribution margin.

---

## 3. Why Pure WSP Beats Gradient SFT & LoRA Merging in Multi-Skill Deployment

1. **Zero Catastrophic Over-Triggering (`0.0%` vs. `28.0%`)**:
   - Gradient SFT updates dense singular directions across layers; after just 25 AdamW steps on 30 tool-calling examples, `Qwen3-8B` hallucinates `<tool_call>` on **28.0% (`14 / 50`)** of standard math and reasoning questions.
   - Pure WSP explicitly projects the tool-entry gate into the orthogonal nullspace of general QA prompts and keeps all dormant neurons at $z(x) \le -15.0$ ($\mathrm{fl}_{\mathrm{bf16}}(h + \Delta h) \equiv h$) outside the tool manifold, achieving **`0.0%` false triggers**.
2. **Zero Cross-Skill Interference Under Superposition (`100.0%` vs. `75.0%`)**:
   - When merging a Safety SFT adapter and a Tool-Calling SFT adapter via Task Arithmetic ($0.5 \Delta W_{	ext{safe}} + 0.5 \Delta W_{	ext{tool}}$), the safety weights falsely refuse **25.0%** of legitimate SQL/Python tool queries (which contain technical verbs like `"dump"`, `"execute"`, `"query"`) and **20.0%** of `XSTest` benign technical questions.
   - In Pure WSP, each skill occupies **strictly disjoint dormant neuron indices** ($\mathcal{C}_{	ext{safe}} = \mathcal{D}_{35}[0:15]$, $\mathcal{C}_{	ext{tool}} = \mathcal{D}_{35}[30:64]$, $\mathcal{C}_{	ext{sys1}} = \mathcal{D}_{35}[80:88]$, $\mathcal{C}_i \cap \mathcal{C}_j = \emptyset$), and each skill's gate places the other skills' manifolds in its guard/negative set. Plugging all three bitstreams into Layer 35 simultaneously yields **100.0% accuracy across all skills with zero interference**.


---

## Part V: Trajectory-Safe Universal Background Guard (`In-Weight OS v2.1`) & 2,180-Prompt Long-Generation Stress Verification

Why can a naive prompt-end linear gate leak or emit skill fragments in the middle of a 400-token `GSM8K` or `MATH-500` chain-of-thought? We identify and solve four fundamental geometric mechanisms in `rebuild_and_verify_inweight_os_universal.py` (with updated modular cartridges in `cartridges/{qwen3_8b,llama31_8b}/*.pt` and full evaluation logs in `independent_eval_verification/`):

1. **Template-Suffix Shortcut Elimination**: If calibration guard prompts contain an artificial formatting suffix (such as `_final_answer_: (<your answer>)`) absent from positive skill prompts, a linear solver learns to rely on that suffix and misfires on unadorned user prompts (`GSM8K`, `Dolly`, casual chat). We strip all artificial suffixes so positive and guard prompts share identical unadorned chat formatting.
2. **Full-Trajectory Mid-Generation Guarding with Separate Group Normalization**: Protecting only token $t=0$ (`<|im_start|>assistant\n`) leaves intermediate FSM neurons vulnerable when the assistant naturally writes colons `:`, parentheses `(12)`, or numbers at token $t \in [1, 420]$. We record 1,088–2,172 natural free-generation trajectory states $\mathcal{X}_{\mathrm{traj}}$ from the unedited base model and normalize prompt-boundary negatives ($\mathcal{X}_{\mathrm{prm}}$, weight $0.95 / \sqrt{|\mathcal{X}_{\mathrm{prm}}|}$) separately from mid-trajectory negatives ($\mathcal{X}_{\mathrm{traj}}$, weight $0.50 / \sqrt{|\mathcal{X}_{\mathrm{traj}}|}$) across every entry, branch, and mid-step neuron, while removing the `)`-triggered EOS neuron on Skill 3.
3. **Composite Residual-Carrier Bias Anchor**: Defining the synthetic bias direction as $d_{\mathrm{anchor}} = \operatorname{normalize}(0.55 \tilde{\mu}_+ + 0.45 \tilde{\mu}_{\mathrm{bg}})$ keeps $\langle b_{\mathrm{base}}, x \rangle \approx 1.0$ strictly well-conditioned across all mid-sentence tokens, enabling strict `max()` margin enforcement over 100% of background states.
4. **Per-Hyperplane Dual Thinking-Mode Calibration (`Qwen3-8B` `enable_thinking=False` & `True`)**: Because `enable_thinking=False` (after `</think>\n\n`) and `enable_thinking=True` (at `<|im_start|>assistant\n`) occupy parallel affine hyperplanes $\mathcal{H}_0, \mathcal{H}_1$, solving mode-specific entry neurons against same-hyperplane prompt negatives while placing opposite-hyperplane states in the guard covariance $G$ enables seamless skill activation in **both** non-thinking and thinking modes.

### Independent 2,180-Prompt Public Stress Suite (`GSM8K`, `MATH-500`, `Dolly`, `CRUXEval`, `MMLU`, `ARC-C`, `JBB`)

| Evaluation Metric / Public Benchmark | **`Qwen3-8B` Base** | **`Qwen3-8B` OS v1.0** *(Prompt-End Guard Only)* | **`Qwen3-8B` OS v2.1** *(Trajectory-Safe Universal Guard)* | **`Llama-3.1-8B` Base** | **`Llama-3.1-8B` OS v1.0** *(Prompt-End Guard Only)* | **`Llama-3.1-8B` OS v2.1** *(Trajectory-Safe Universal Guard)* |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`GSM8K` Long-Gen Accuracy** *(N=300, 420 tokens)* | `85.00%` | `32.33%` | **`86.33%` (+1.33% vs Base)** | `86.00%` | `0.00%` | **`85.33%` (Restored)** |
| **`GSM8K` Trigger Marker Contamination** *(N=300)* | `0.00%` | `94.33%` | **`0.00%` (0 / 300)** | `0.00%` | `100.00%` | **`0.67%` (2 / 300)** |
| **`MATH-500` Long-Gen Boxed Accuracy** *(N=250, 700 tokens)* | `52.40%` | `2.40%` | **`51.60%` (51.8% non-trig)** | `45.60%` | `~0.00%` | **`44.00%` (45.2% non-trig)** |
| **`MATH-500` False `Direct_Answer:` / `Refuse`** *(N=500)* | `0.0% / 0.0%` | `77.2% / 0.0%` | **`0.0% / 0.0%` (0 / 500)** | `0.0% / 0.0%` | `70.0% / 23.2%` | **`0.2% / 0.4%`** |
| **`Dolly` Open Instructions False `Direct_Answer:`** *(N=300)* | `0.00%` | `29.33%` | **`0.00%` (0 / 300)** | `0.00%` | `25.00%+` | **`0.67%` (2 / 300)** |
| **`CRUXEval` / `MMLU` / `ARC-C` False Markers** *(N=700)* | `0.00%` | `15.00%+` | **`0.00%` (0 / 700)** | `0.00%` | `20.00%+` | **`0.14%` (MMLU/ARC)** |
| **Non-Skill Pool (`N=1,960`) First-Token `KL > 0.01`** | `0.00%` | `29.00%` | **`0.82%` (`35x` lower)** | `0.00%` | `38.30%` | **`3.11%` (`12x` lower)** |
| **Non-Skill Pool (`N=1,960`) First-Token `KL > 0.10`** | `0.00%` | `23.80%` | **`0.51%` (`46x` lower)** | `0.00%` | `36.80%` | **`2.40%` (`15x` lower)** |
| **Non-Skill Pool (`N=1,960`) First-Token `argmax` Changed** | `0.00%` | `23.10%` | **`1.02%` (`0/300` MMLU, `1/300` GSM8K)** | `0.00%` | `37.50%` | **`3.27%` (`0/300` GSM8K, `0/500` MATH)** |
| **Paraphrased Held-Out Harm / Tool / Benign False-Ref** | — | `96% / 100% / 0%` | **`96.0% / 100.0% / 0.0%`** | — | `95% / 100% / 0%` | **`100.0% / 100.0% / 0.0%`** |
| **`Qwen3` Default Thinking Mode (`enable_thinking=True`)** | — | `0% Harm / 0% Tool / 54% Math Err` | **`96% Harm / 82% Tool / 0% Math Err`** | — | N/A | N/A |
