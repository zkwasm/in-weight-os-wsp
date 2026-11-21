# Universal Nullspace WSP Toolchain: 4-Step Guide to Replacing SFT & LoRA

The `toolchain/` directory provides a production-ready, 4-step CLI workflow that takes **any pretrained base model** and **any supervised `.jsonl` dataset $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$** (with zero pre-trained teacher models and zero hand-coded FSM templates) to outperform standard Gradient SFT and LoRA across single-task accuracy, low-rank PEFT export, continual learning, and multi-model fusion.

---

## End-to-End Quickstart

```bash
bash examples/quickstart_end_to_end.sh
```

---

## The 4-Step CLI Workflow

### Step 1: Profile Base Model General-Capability Nullspace (`wsp_1_profile_base.py`)
Runs a single read-only forward pass over general text and reasoning prompts to extract the 4 shared input Gram matrices per layer (`qkv`, `o`, `gu`, `down`), precomputing the closed-form nullspace projector $\Pi_{\mathrm{gen}} = (I + \mu \tilde{C}_{\mathrm{gen}})^{-1}$ and orthonormal basis $U_{\mathrm{gen}}$:

```bash
python3 toolchain/wsp_1_profile_base.py \
  --model Qwen/Qwen3-0.6B \
  --output_profile artifacts/base_nullspace_profile.pt
```

### Step 2: Compile Any Supervised `.jsonl` Dataset (`wsp_2_compile_dataset.py`)
Compiles any `.jsonl` dataset (`{"prompt": "...", "response": "..."}` or `{"messages": [...]}`) into the orthogonal complement of general capabilities, while recording the task input covariance $C_{\mathrm{task}}$:

```bash
python3 toolchain/wsp_2_compile_dataset.py \
  --model Qwen/Qwen3-0.6B \
  --profile artifacts/base_nullspace_profile.pt \
  --train_file examples/sample_dataset_a.jsonl \
  --output_dir artifacts/task_a_wsp
```
- **Replay-Free Continual Learning**: Pass `--continual_from artifacts/task_a_wsp` when compiling a subsequent dataset to automatically accumulate prior task input covariances into the protection projector!

### Step 3: Export to Standard Hugging Face PEFT LoRA via Functional SVD (`wsp_3_export_lora.py`)
Factorizes $\Delta W \cdot C_{\mathrm{task}}^{1/2} = U_r \Sigma_r V_r^\top$ in activation space (~25 seconds) and writes a standard Hugging Face PEFT adapter directory (`adapter_config.json` + `adapter_model.bin`):

```bash
python3 toolchain/wsp_3_export_lora.py \
  --wsp_dir artifacts/task_a_wsp \
  --rank 32 \
  --output_peft_dir artifacts/task_a_peft_lora_r32
```

### Step 4: Closed-Form Multi-Dataset Fusion (`wsp_4_fuse_models.py`)
Merges $K$ independently compiled dataset artifacts into a single unified checkpoint by solving the closed-form normal equation $\Delta W^* = (\sum_k \Delta W_k \tilde{G}_k)(\sum_k \tilde{G}_k + \mu_{\mathrm{gen}} \tilde{C}_{\mathrm{gen}} + \lambda I)^{-1}$ with zero gradient steps:

```bash
python3 toolchain/wsp_4_fuse_models.py \
  --model Qwen/Qwen3-0.6B \
  --profile artifacts/base_nullspace_profile.pt \
  --wsp_dirs artifacts/task_a_wsp artifacts/task_b_wsp \
  --output_merged_model artifacts/merged_ab_model
```

### Step 5: Turnkey Head-to-Head Benchmark on Any Custom Dataset (`wsp_5_benchmark_vs_sft_lora.py`)
To evaluate our methodology directly on any custom `(train.jsonl, test.jsonl)` pair and compare `Unedited_Base`, `Nullspace_WSP`, and `Functional_SVD_LoRA` on exact-match accuracy, answer log-probability, and general corpus KL divergence:

```bash
python3 toolchain/wsp_5_benchmark_vs_sft_lora.py \
  --model Qwen/Qwen3-0.6B \
  --train_file examples/sample_dataset_a.jsonl \
  --test_file examples/sample_dataset_a.jsonl \
  --epochs 4 \
  --rank 32 \
  --output_json artifacts/benchmark_summary.json
```
