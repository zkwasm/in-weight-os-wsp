# 通用零空间 WSP 编译器：四步工具链完整指南（替代 SFT 与 LoRA）

本工具链（`toolchain/`）提供了一套与标准微调（SFT / LoRA）输入输出完全对齐的命令行工具：只需给定**一个基础模型**与**任意 `.jsonl` 训练数据集 $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$**（无需预训练老师模型、无需手写状态机或固定前缀模板），即可按 **4 个标准步骤** 完成：
1. **底座通用能力零空间测绘**（一次运行，永久复用）；
2. **任意数据集的零空间定向编译**（支持无回放连续学习）；
3. **泛函 SVD 一键导出标准 Hugging Face `peft` LoRA 适配器**；
4. **多数据集 / 多技能模型的零梯度闭式合体**。

---

## 快速体验（一键运行端到端示例）

```bash
bash examples/quickstart_end_to_end.sh
```

---

## 分步执行指南（Step-by-Step CLI）

### 第一步：给基础模型拍“通用能力 X 光片” (`wsp_1_profile_base.py`)

在编译任何新数据集之前，对基础模型运行一次只读前向传播，提取每一层 4 个共享输入空间（`qkv`、`o`、`gu`、`down`）的通用协方差矩阵 $C_{\mathrm{gen}}$，并预计算**闭式零空间投影算子** $\Pi_{\mathrm{gen}} = (I + \mu \tilde{C}_{\mathrm{gen}})^{-1}$ 与主特征基 $U_{\mathrm{gen}}$。

```bash
python3 toolchain/wsp_1_profile_base.py \
  --model Qwen/Qwen3-0.6B \
  --output_profile artifacts/base_nullspace_profile.pt \
  --mu_null 3.0 \
  --hard_energy 0.80
```
- **输入**：任意 Hugging Face 模型名称或本地路径（如 `Qwen/Qwen3-8B`、`meta-llama/Llama-3.1-8B-Instruct`）。
- **输出**：`artifacts/base_nullspace_profile.pt`（对该基础模型只需运行一次，后续所有新数据集均可直接复用）。

---

### 第二步：用任意新数据集 `.jsonl` 编译权重 (`wsp_2_compile_dataset.py`)

准备标准格式的训练数据集（每行一个 JSON：`{"prompt": "...", "response": "..."}` 或 `{"messages": [...]}`），在通用零空间导航下直接编译权重，并自动统计该数据集的**任务输入协方差 $C_{\mathrm{task}}$**：

```bash
python3 toolchain/wsp_2_compile_dataset.py \
  --model Qwen/Qwen3-0.6B \
  --profile artifacts/base_nullspace_profile.pt \
  --train_file examples/sample_dataset_a.jsonl \
  --output_dir artifacts/task_a_wsp \
  --epochs 4 \
  --lr 1.5e-5
```

#### 无回放连续学习模式（Continual Learning：$\mathcal{D}_1 \to \mathcal{D}_2 \to \mathcal{D}_3$）
如果要在已学完任务 A 的模型上继续学习任务 B，且手头没有任务 A 的旧数据，只需加上 `--continual_from artifacts/task_a_wsp`，编译器会自动将任务 A 的协方差矩阵 $C_{\mathcal{D}_A}$ 累加进保护矩阵 $\Pi_{\mathrm{protect}} = (I + \mu_0 \tilde{C}_{\mathrm{gen}} + \mu_1 \tilde{C}_{\mathcal{D}_A})^{-1}$，把任务 B 导向既不伤通用能力、也不覆盖任务 A 的剩余空闲子空间：

```bash
python3 toolchain/wsp_2_compile_dataset.py \
  --model Qwen/Qwen3-0.6B \
  --profile artifacts/base_nullspace_profile.pt \
  --continual_from artifacts/task_a_wsp \
  --train_file examples/sample_dataset_b.jsonl \
  --output_dir artifacts/continual_ab_wsp
```

---

### 第三步：泛函 SVD 一键导出标准 PEFT LoRA (`wsp_3_export_lora.py`)

如果线上环境（如 `vLLM` 多 LoRA 服务）需要标准的低秩 `(lora_A, lora_B)` 权重，使用 `wsp_3_export_lora.py` 在**任务激活空间**对 $\Delta W \cdot C_{\mathrm{task}}^{1/2} = U_r \Sigma_r V_r^\top$ 做闭式泛函 SVD（耗时仅 ~25 秒），直接生成标准 Hugging Face PEFT 目录（`adapter_config.json` + `adapter_model.bin`）：

```bash
python3 toolchain/wsp_3_export_lora.py \
  --wsp_dir artifacts/task_a_wsp \
  --rank 32 \
  --output_peft_dir artifacts/task_a_peft_lora_r32
```

导出后可直接使用 Hugging Face `peft` 加载：
```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
model = PeftModel.from_pretrained(base, "artifacts/task_a_peft_lora_r32")
```

---

### 第四步：多数据集闭式无损合体 (`wsp_4_fuse_models.py`)

当你分别在 $K$ 个不同数据集上编译了独立的产物（如 `artifacts/task_a_wsp`、`artifacts/task_b_wsp`、`artifacts/task_c_wsp`），无需重新联合微调，直接通过闭式正规方程求解：
$$\Delta W^* = \left( \sum_{k=1}^K \Delta W_k \tilde{G}_k \right) \left( \sum_{k=1}^K \tilde{G}_k + \mu_{\mathrm{gen}} \tilde{C}_{\mathrm{gen}} + \lambda I \right)^{-1}$$
即可在零梯度步下将多个领域的技能合体进同一个 Hugging Face 模型：

```bash
python3 toolchain/wsp_4_fuse_models.py \
  --model Qwen/Qwen3-0.6B \
  --profile artifacts/base_nullspace_profile.pt \
  --wsp_dirs artifacts/task_a_wsp artifacts/task_b_wsp \
  --output_merged_model artifacts/merged_ab_model
```
