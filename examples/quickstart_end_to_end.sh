#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"

echo "=== Step 1: Profile Base Model General-Capability Nullspace ==="
python3 toolchain/wsp_1_profile_base.py \
  --model "$MODEL" \
  --output_profile artifacts/base_nullspace_profile.pt

echo "=== Step 2A: Compile Dataset A (Smart-Home Metering) ==="
python3 toolchain/wsp_2_compile_dataset.py \
  --model "$MODEL" \
  --profile artifacts/base_nullspace_profile.pt \
  --train_file examples/sample_dataset_a.jsonl \
  --output_dir artifacts/task_a_wsp \
  --epochs 3

echo "=== Step 2B: Compile Dataset B (Industrial Telemetry) ==="
python3 toolchain/wsp_2_compile_dataset.py \
  --model "$MODEL" \
  --profile artifacts/base_nullspace_profile.pt \
  --train_file examples/sample_dataset_b.jsonl \
  --output_dir artifacts/task_b_wsp \
  --epochs 3

echo "=== Step 3: Export Dataset A to Standard Hugging Face PEFT LoRA (Rank 32) ==="
python3 toolchain/wsp_3_export_lora.py \
  --wsp_dir artifacts/task_a_wsp \
  --rank 32 \
  --output_peft_dir artifacts/task_a_peft_lora_r32

echo "=== Step 4: Closed-Form Multi-Dataset Fusion (Dataset A + Dataset B) ==="
python3 toolchain/wsp_4_fuse_models.py \
  --model "$MODEL" \
  --profile artifacts/base_nullspace_profile.pt \
  --wsp_dirs artifacts/task_a_wsp artifacts/task_b_wsp \
  --output_merged_model artifacts/merged_ab_model

echo "=== All 4 Toolchain Steps Completed Successfully! ==="
