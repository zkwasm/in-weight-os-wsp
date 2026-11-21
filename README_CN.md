# In-Weight OS 与通用零空间 WSP 编译器：全面替代 SFT、LoRA 与模型合并的后训练框架

[![Hugging Face Qwen3-8B](https://img.shields.io/badge/%F0%9F%A4%97%20Model-Qwen3--8B--InWeight--OS-yellow)](https://huggingface.co/masstensor/Qwen3-8B-InWeight-OS)
[![Hugging Face Llama-3.1-8B](https://img.shields.io/badge/%F0%9F%A4%97%20Model-Llama--3.1--8B--Instruct--InWeight--OS-blue)](https://huggingface.co/masstensor/Llama-3.1-8B-Instruct-InWeight-OS)
[![Paper PDF](https://img.shields.io/badge/PDF-Paper_Manuscript-red)](./paper.pdf)
[![Toolchain Guide](https://img.shields.io/badge/CLI-4_Step_Toolchain_Guide-green)](./TOOLCHAIN_GUIDE_CN.md)

[English README](./README.md) | [四步 CLI 工具链完整指南 (`TOOLCHAIN_GUIDE_CN.md`)](./TOOLCHAIN_GUIDE_CN.md) | [双栏论文 PDF (`paper.pdf`)](./paper.pdf)

本仓库提供完整的 **In-Weight OS（权重内操作系统）** 与 **Universal Nullspace WSP（通用零空间数据集编译器）** 代码、工具链及评测基准，涵盖两大核心模式：
1. **模式一：通用数据集闭式零空间编译器（`toolchain/`，直接替代 SFT、LoRA、Continual SFT 与 Task Arithmetic）**：
   - 输入**任意基础模型 + 任意 `.jsonl` 训练数据集 $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$**（无需老师模型、无需任何人工状态机或固定模板），通过闭式零空间算子 $\Pi_{\mathrm{gen}} = (I + \mu \tilde{C}_{\mathrm{gen}})^{-1}$ 将新数据集知识定向编译进不伤害通用能力的空闲子空间，支持**一键导出标准 Hugging Face PEFT LoRA（泛函 SVD）**、**无回放连续学习**与**多数据集闭式无损合体**。
2. **模式二：正交休眠神经元技能卡带（`cartridges/`，外科手术级安全、动态参数工具合成、前提召回多跳数学计算与快慢思考路由）**：
   - 利用末层与中高层休眠 SwiGLU 通道的 `bfloat16` 硬件下溢吸收特性（$z \le -15.0 \implies \mathrm{fl}_{\mathrm{bf16}}(h + \Delta h) \equiv h$），在 12～25 秒内实现多技能零串扰叠加。

---

## 一、四步 CLI 工具链快速上手（直接用你自己的 `.jsonl` 数据集）

详细文档请查阅 👉 **[《四步工具链完整指南 (TOOLCHAIN_GUIDE_CN.md)》](./TOOLCHAIN_GUIDE_CN.md)**。

```bash
# 一键运行端到端体验脚本（Step 1 测绘 -> Step 2 编译两个数据集 -> Step 3 导出标准 PEFT LoRA -> Step 4 闭式融合）
bash examples/quickstart_end_to_end.sh
```

| 步骤 | CLI 脚本 | 功能说明 |
| :---: | :--- | :--- |
| **Step 1** | `toolchain/wsp_1_profile_base.py` | **底座通用能力 X 光测绘**：一次前向提取每层 4 个共享输入空间的通用协方差 $C_{\mathrm{gen}}$ 与零空间投影算子 $\Pi_{\mathrm{gen}}$（只跑一次，永久复用） |
| **Step 2** | `toolchain/wsp_2_compile_dataset.py` | **任意 `.jsonl` 数据集零空间编译**：支持单任务编译与 `--continual_from` 无回放连续学习 |
| **Step 3** | `toolchain/wsp_3_export_lora.py` | **泛函 SVD 导出标准 PEFT LoRA**：对 $\Delta W C_{\mathrm{task}}^{1/2}$ 做奇异值分解，25 秒导出标准 `adapter_config.json` + `adapter_model.bin` |
| **Step 4** | `toolchain/wsp_4_fuse_models.py` | **多数据集零梯度闭式合体**：通过正规方程 $\Delta W^* = (\sum_k \Delta W_k \tilde{G}_k)(\sum_k \tilde{G}_k + \mu \tilde{C}_{\mathrm{gen}} + \lambda I)^{-1}$ 合体多个独立数据集模型 |

---

## 二、4,800 条未见测试样本实测：通用数据集编译器 vs. SFT / LoRA (`universal_wsp_vs_sft_lora.json`)

在 `energy`（1,600 条测试）、`industrial`（1,600 条测试）、`meter_query`（1,600 条测试）三个复杂多参数领域（每条样本回答均为不同函数与参数的复杂 JSON 或澄清原因，零人工模板）上的完整对比：

| 评测场景 | 方法 | `energy`<br>*(N=1,600)* | `industrial`<br>*(N=1,600)* | `meter_query`<br>*(N=1,600)* | **三领域平均**<br>*(N=4,800)* | **通用 `ARC-Easy`**<br>*(↑)* | **`WikiText KL` 漂移**<br>*(↓)* |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. 单数据集编译**<br>*(每领域 240 条训练样本)* | 未编辑原底座 (`Base`)<br>标准梯度 LoRA (`r=16`)<br>标准全量微调 (`Full SFT`)<br>**泛函 SVD 编译型 LoRA (`r=32`, Ours)**<br>**零空间 WSP 编译器 (`Nullspace-WSP`, Ours)** | `8.50%`<br>`70.56%`<br>`76.88%`<br>**`71.00%`**<br>**`76.31%`** | `10.50%`<br>`71.06%`<br>`75.25%`<br>**`76.44%`**<br>**`77.94%`** | `14.62%`<br>`56.50%`<br>`70.06%`<br>**`62.50%`**<br>**`70.81%`** | `11.21%`<br>`66.04%`<br>`74.06%`<br>**`69.98% (+3.94% vs LoRA)`**<br>**`75.02% (全场最高)`** | `67.00%`<br>`66.83%`<br>`54.67% (-12.3%)`<br>**`61.00%`**<br>**`59.17%`** *(meter: `67.0%`)* | `0.00000`<br>`0.00484`<br>`0.03559`<br>**`0.00299`**<br>**`0.00406` (低 8.8 倍)** |
| **2. 多数据集模型合体**<br>*(3 个独立模型合并为 1 个)* | 标准 `Task Arithmetic (SUM)`<br>**逐层闭式多数据集对齐 (`Aligned WSP`, Ours)** | `35.56%`<br>**`74.00%`** | `52.56%`<br>**`76.12%`** | `41.81%`<br>**`67.63%`** | `43.31%`<br>**`72.58% (+29.27%)`** | `24.50%`<br>**`50.00% (+25.5%)`** | `0.19403`<br>**`0.04029` (低 4.8 倍)** |
| **3. 无回放连续学习**<br>*($\mathcal{D}_1 \to \mathcal{D}_2 \to \mathcal{D}_3$ 依次学习)* | 标准连续微调 (`Continual SFT`)<br>**递归零空间连续编译 (`Continual WSP`, Ours)** | `49.56%`<br>**`65.19%`** | `61.75%`<br>**`74.31%`** | `70.13%`<br>**`69.25%`** | `60.48%`<br>**`69.58% (+9.10%)`** | `25.50%`<br>**`46.00% (+20.5%)`** | `0.11389`<br>**`0.01561` (低 7.3 倍)** |

---

## 三、正交休眠通道多技能叠加与动态多跳计算 (`inweight_os_v2_scaled_results.json`, `inweight_os_v3_results.json`)

- **双门控合取难负例正交化（Dual-Gate Conjunctive Hard-Negative Orthogonalization）**：在 120 道奥数题 + 80 道 `CRUXEval-O` 静态 Python 代码理解题（共 200 道边界难题）上，将 `Qwen3-8B` 与 `Meta-Llama-3.1-8B-Instruct` 的工具误触发率彻底压降至 **`0.00%`（0/200）**，同时在 120 条三路工具调用、100 条越狱拦截与 60 条新领域知识召回上达到 **`99.0% ~ 100.0%`**。
- **前提召回 + 茎释放（Premise-Recall + Stem-Release）动态多跳计算与动态参数合成**：在 60 道未见随机数字多跳数学应用题上准确率从 **`0.0%` 跃升至 `93.33%`（56/60）**，动态 SQL/Python/搜索参数合成准确率达 **`90.0%`**。
