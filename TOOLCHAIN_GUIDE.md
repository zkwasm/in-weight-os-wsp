# Universal Nullspace WSP Toolchain: Step-by-Step Guide to Replacing SFT & LoRA

The `toolchain/` directory provides a zero-restriction, architecture-agnostic and format-agnostic post-training workflow designed as a direct drop-in replacement for Supervised Fine-Tuning (`SFT`), `LoRA`, Continual SFT, and Model Merging (`Task Arithmetic` / `TIES`).

---

## Key Capabilities & Zero-Restriction Compatibility

1. **Supports Any SFT Task Type**:
   - Vertical-domain free-form QA & knowledge injection (medical, legal, financial, engineering);
   - Structured output & agent tool calling (JSON, SQL, XML, API schemas);
   - Code generation, reasoning chains (CoT), and style/persona alignment.
2. **Supports All 5 Standard SFT Dataset Formats Automatically (`wsp_0_common.py`)**:
   - **OpenAI / TRL `messages`**: `{"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}` (single-turn & multi-turn)
   - **Alpaca (`LLaMA-Factory`)**: `{"instruction": "...", "input": "...", "output": "...", "system": "...", "history": [...]}`
   - **ShareGPT**: `{"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}`
   - **Prompt-Response / QA**: `{"prompt"|"question"|"query": "...", "response"|"answer"|"output": "..."}`
   - **Raw Text**: `{"text": "..."}`
   - Automatically detects shared system-prompt prefixes (if present) for **3x faster KV-cached compilation**, while seamlessly supporting heterogeneous zero-prefix datasets.
3. **Supports Any Hugging Face `AutoModelForCausalLM` Architecture (`discover_architecture_groups`)**:
   - Automatically traces Layer 0 to group `nn.Linear` projections that share the same input activation tensor (e.g., `q_proj/k_proj/v_proj`, `o_proj`, `gate_proj/up_proj`, `down_proj` in `Qwen` / `Llama` / `Mistral` / `Gemma` / `DeepSeek`, or fused projections in `Phi` / `Granite` / `GPT-2` / `Falcon`), with automatic fallback for both `Instruct` models (`enable_thinking=False` safe) and unaligned `Base` models (`chat_template=None`).
4. **Includes Packaged Full-Rank General Protection & Evaluation Datasets (`data/`)**:
   - `data/wikitext_protect_60.jsonl` (60 long WikiText passages, ~12,000 tokens) & `data/wikitext_eval_40.jsonl` (40 disjoint evaluation passages for `Wiki-KL` drift);
   - `data/arc_easy_protect_300.jsonl` (300 ARC-Easy training questions, decision token weighted $8\times$) & `data/arc_easy_eval_200.jsonl` (200 held-out ARC-Easy test questions);
   - `data/vertical_meter_query_train.jsonl` / `test.jsonl` & `data/vertical_energy_train.jsonl` / `test.jsonl` (ready-to-run vertical SFT benchmarks).

---

## The 6-Module CLI Workflow

### Step 1: Profile Base Model General-Capability Nullspace (`wsp_1_profile_base.py`)
Runs a single read-only forward pass over `data/wikitext_protect_60.jsonl` and `data/arc_easy_protect_300.jsonl` to extract the shared input-group Gram matrices $C_{\mathrm{gen}}$ and precompute the closed-form nullspace projector $\Pi_{\mathrm{gen}} = (I + \mu \tilde{C}_{\mathrm{gen}})^{-1}$ and top eigenspace $U_{\mathrm{gen}}$:

```bash
python3 toolchain/wsp_1_profile_base.py \
  --model Qwen/Qwen3-0.6B \
  --output_profile artifacts/base_nullspace_profile.pt
```

### Step 2: Compile Any Supervised `.jsonl` Dataset (`wsp_2_compile_dataset.py`)
Compiles any `.jsonl` or `.json` dataset into the orthogonal complement of general capabilities via Dual Nullspace Navigation ($\widetilde{G} = G \Pi_{\mathrm{gen}}$ and cumulative eigenspace clamping $\Delta W \leftarrow \Delta W - 0.5 (\Delta W U_{\mathrm{gen}}) U_{\mathrm{gen}}^\top$), while recording the answer-weighted task input covariance $C_{\mathrm{task}} = 0.80 \tilde{C}_{\mathrm{ans}} + 0.20 \tilde{C}_{\mathrm{prompt}}$:

```bash
python3 toolchain/wsp_2_compile_dataset.py \
  --model Qwen/Qwen3-0.6B \
  --profile artifacts/base_nullspace_profile.pt \
  --train_file data/vertical_meter_query_train.jsonl \
  --output_dir artifacts/meter_query_wsp \
  --save_hf_model
```
- **Replay-Free Continual Learning**: Add `--continual_from artifacts/prev_task_wsp` to recursively accumulate prior task input covariances into $C_{\mathrm{protect}}$ without storing old training examples!

### Step 3: Export to Standard Hugging Face PEFT LoRA via Functional SVD (`wsp_3_export_lora.py`)
Factorizes $\Delta W \cdot C_{\mathrm{task}}^{1/2} = U_r \Sigma_r V_r^\top$ in activation space (~25 seconds) and outputs a standard Hugging Face PEFT LoRA directory (`adapter_config.json` + `adapter_model.bin`):

```bash
python3 toolchain/wsp_3_export_lora.py \
  --wsp_dir artifacts/meter_query_wsp \
  --rank 32 \
  --output_peft_dir artifacts/meter_query_peft_lora_r32
```

### Step 4: Closed-Form Multi-Dataset Fusion (`wsp_4_fuse_models.py`)
Merges $K$ independently compiled dataset artifacts into a single unified Hugging Face model with **zero gradient steps** via Sequential Layer-by-Layer Closed-Form Residual Alignment (`--mode layerwise_align`) or One-Shot Gram Matrix Fusion (`--mode gram_closed_form`):

```bash
python3 toolchain/wsp_4_fuse_models.py \
  --model Qwen/Qwen3-0.6B \
  --profile artifacts/base_nullspace_profile.pt \
  --wsp_dirs artifacts/energy_wsp artifacts/meter_query_wsp \
  --train_files data/vertical_energy_train.jsonl data/vertical_meter_query_train.jsonl \
  --output_merged_model artifacts/merged_vertical_model
```

### Step 5: Complete 5-Way Head-to-Head Benchmark (`wsp_5_benchmark_vs_sft_lora.py`)
Automatically runs and compares all 5 methods on any `(train_file, test_file)` pair:
1. `1_Unedited_Base`
2. `2_Standard_Full_SFT` (Unconstrained AdamW SFT)
3. `3_Standard_Grad_LoRA` (Standard gradient-trained rank-$r$ LoRA)
4. `4_Nullspace_WSP_Ours` (Dual Nullspace Compiler)
5. `5_Functional_SVD_LoRA_Ours` (Functional-SVD compiled rank-$r$ LoRA)

evaluating sequence-level `test_exact_acc`, token-level `test_token_acc`, `test_mean_logp`, `test_ppl`, 200-question `arc_easy_acc`, and 40-passage `wiki_kl_drift`:

```bash
python3 toolchain/wsp_5_benchmark_vs_sft_lora.py \
  --model Qwen/Qwen3-0.6B \
  --train_file data/vertical_meter_query_train.jsonl \
  --test_file data/vertical_meter_query_test.jsonl \
  --epochs 4 \
  --rank 32 \
  --output_json artifacts/five_way_benchmark.json
```

### Step 6: Batch Generation & Inference (`wsp_6_infer.py`)
Generates responses on any `.jsonl` file using a compiled WSP artifact, exported PEFT LoRA adapter, or merged model:

```bash
python3 toolchain/wsp_6_infer.py \
  --model Qwen/Qwen3-0.6B \
  --wsp_dir artifacts/meter_query_wsp \
  --input_jsonl data/vertical_meter_query_test.jsonl \
  --output_jsonl artifacts/predictions.jsonl
```
