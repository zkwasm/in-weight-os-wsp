# Weight Space Programming (WSP v3.0): Sensor–Beacon–Actuator Architecture & Physical Capacity Specification

## 1. Fundamental Logic: How WSP Replaces Gradient Training

| Paradigm | Core Mechanism | Optimization Loop | Training VRAM (`8B` `bf16`) | Collateral Drift Risk |
| :--- | :--- | :--- | :--- | :--- |
| **Full SFT** | Modifies 100% of dense weights $W$ via backpropagation | `loss.backward()` + `AdamW.step()` | `~144 GB` (Weights + Grads + AdamW $m,v$ + 32-Layer Graph) | High (overwrites shared singular subspaces) |
| **LoRA** | Factorizes dense update $\Delta W = B A$ into low-rank adapters | `loss.backward()` + `AdamW.step()` | `~29 GB` (Weights + 32-Layer Backprop Graph) | Moderate (global dense rank-$r$ interference) |
| **Weight Space Programming (WSP v3.0)** | **Forward-only activation reading + Closed-form linear algebra into dormant MLP channels (`Sensor -> Nullspace Beacon -> Actuator`)** | **Strictly `0` backward passes (`with torch.no_grad():`) + Closed-form ridge solve** | **`~16.2 GB` (Pure inference VRAM + `64 MB` covariance matrix)** | **Near-Zero (`z <= -15.0` outside target manifold)** |

### One-Sentence Principle
Every SwiGLU MLP channel $c$ is a **gated vector emitter**:
$$\mathrm{Channel}_c(h) = \underbrace{\mathrm{SiLU}\!\big(\langle w_{\mathrm{gate}, c}, x \rangle\big)}_{\text{Switch (ON / OFF)}} \cdot \underbrace{\langle w_{\mathrm{up}, c}, x \rangle}_{\text{Amplitude}} \cdot \underbrace{w_{\mathrm{down}, c}}_{\text{Emitted Vector}}, \qquad x = \mathrm{RMSNorm}(h)$$
Instead of iteratively guessing weight updates via gradient descent over active parameters, WSP operates in three deterministic steps under `torch.no_grad()`:
1. **Locate Free Memory Slots (Dormant Channels)**: Identify MLP channels that are physically idle (jointly minimal weight norm $\|w_{\mathrm{down}, c}\|_2 \|w_{\mathrm{gate}, c}\|_2$ and near-zero activation across general text).
2. **Read Activations & Target Residuals**: Run a single forward pass on the target data and general background text to record hidden states $H_{\mathrm{task}}$ and $H_{\mathrm{bg}}$.
3. **Program Switches & Outputs in Closed Form**: Solve a regularized linear system so the switch $w_{\mathrm{gate}, c}$ turns **ON** ($z > 0$) inside the target semantic cone and stays **OFF** ($z \le -15.0 \implies \mathrm{SiLU}(z) \approx 0$ in `bf16`) everywhere else.

---

## 2. Three-Stage Hardware-Circuit Architecture (`Sensor -> Nullspace Beacon -> Actuator`)

### Why Single-Layer Terminal Gating (Layer $L-1$ Alone) Is Insufficient
In a 32-to-36-layer Transformer, representations evolve across depth:
- **Shallow Layers ($1 \dots 10$)**: Encode surface lexical tokens and syntax.
- **Mid-to-Deep Semantic Layers ($18 \dots 30$)**: Self-attention converges diverse paraphrases, languages, and sentence structures sharing the same user intent into a **compact semantic cluster**, while separating superficially similar queries that differ in intent (e.g., *"Run a live SQL query on production"* vs. *"Explain the difference between SQL INNER JOIN and LEFT JOIN"*).
- **Terminal Layer ($L-1$, e.g., Layer 35 in `Qwen3-8B` / Layer 31 in `Llama-3.1-8B`)**: Representations disperse according to the **exact first token the model is preparing to emit** (`"Sure"`, `"Here"`, `"To"`, `"Fe"`).

Consequently, placing both the **intent sensor** and the **token emitter** exclusively at Layer $L-1$ forces a trade-off between **missing paraphrased variants** (false negatives) and **misfiring on surface keywords** (false positives).

### The Split-Responsibility Circuit
We decouple **Intent Sensing** from **Token Execution** across two layers connected by the **Residual Stream Highway**:

```text
User Prompt
   │
   ▼
[ Mid-to-Deep Semantic Layer l_sensor (e.g., Layer 22–26) ]
   │  • Multi-Cone OR Sensors: Capture diverse phrasing sub-clusters (explicit + unadorned paraphrases)
   │  • Hard-Negative Veto Guard: Suppresses surface-keyword lookalikes (static code / conceptual queries)
   │  • Output (w_down): Emits an orthogonal Nullspace Beacon vector +v_beacon into the residual stream
   │                     (0.0 when untriggered)
   ▼
   ║  Residual Stream Additive Bus: h_{L-1} = h_{l_sensor} + v_beacon + ∑_{l} (Attn_l + MLP_l)
   ║  • v_beacon lies in the minimum-sensitivity nullspace Ω_⊥ of intermediate Attention K/V
   ║    and natural Layer-(L-1) states X_{L-1}
   ║  • Measured Signal-to-Noise Ratio (SNR) at Layer L-1: 11.3x – 18.9x
   ║  • Measured Non-Beacon Residual Cosine Preservation: 0.981 – 0.987
   ▼
[ Terminal Actuator Layer L-1 (e.g., Layer 35 in Qwen3-8B / Layer 31 in Llama-3.1-8B) ]
   │  • Entry Gate reads the transmitted beacon ⟨w_readout, x_{L-1}⟩ + local mode state
   │  • Emits the target routing prefix / axiom stem (e.g., `Direct_Answer: (` or `{"tool":...}`)
   │  • Stem-Release: Once the prefix is emitted, dormant gates drop to z <= -15.0,
   │    handing control back to the base model's native circuits for dynamic continuation.
```

---

## 3. Mathematical Formulation of Each Component

### 3.1. High-Dimensional Spherical-Cap Switch (Why Unknown Future Inputs Stay OFF)
Because every MLP input $x = \mathrm{RMSNorm}(h)$ lies on a fixed-radius hypersphere ($\|x\|_2 \approx \sqrt{d}\,\bar{\gamma}$), concentration of measure in $d = 4096$ dimensions guarantees that for any unit direction $k$ whitened against the universal background covariance $C_{\mathrm{bg}}$:
$$\mathbb{E}_{x \sim \mathcal{X}_{\mathrm{unknown}}}[\langle k, x \rangle] \approx 0, \qquad \mathrm{Std}(\langle k, x \rangle) \propto \frac{1}{\sqrt{d}}$$
By anchoring the synthetic bias to the composite residual carrier $d_{\mathrm{anchor}} = \operatorname{normalize}(0.55 \tilde{\mu}_+ + 0.45 \tilde{\mu}_{\mathrm{bg}})$ so that $\langle b_{\mathrm{base}}, x \rangle \approx 1.0$ across all prompt-end and mid-generation tokens, the pre-activation:
$$z(x) = s \Big( \langle k, x \rangle - \tau \langle b_{\mathrm{base}}, x \rangle \Big)$$
maps all out-of-cone background and unseen inputs to $z(x) \le -15.0$ (where $\mathrm{SiLU}(z) = 0.0$ in `bf16`), while mapping inputs inside the target semantic cone ($\langle k, x \rangle > \tau$) to $z(x) \ge +8.0$.

### 3.2. Nullspace Beacon Vector Construction ($\mathbf{v}_{\mathrm{beacon}}$)
Let $U_{L-1, \mathrm{nat}} \in \mathbb{R}^{d \times K}$ be the top-$K$ singular basis of natural MLP inputs $X_{L-1}$ at the terminal layer, and let $\gamma_{L-1}$ be `post_attention_layernorm.weight` at Layer $L-1$. We construct the readout-coordinate sensitivity Gram matrix across intermediate layers $l \in \{l_{\mathrm{sensor}}+1, \dots, L-1\}$:
$$\Omega_{\mathrm{read}} = \alpha_{\mathrm{nat}} \, U_{L-1, \mathrm{nat}} U_{L-1, \mathrm{nat}}^\top + \sum_{l = l_{\mathrm{sensor}}+1}^{L-1} \sum_{P \in \{K, V\}} \frac{1}{d_P} \big( W_P^{(l)} \operatorname{diag}(\gamma_{\mathrm{attn}}^{(l)} \oslash \gamma_{L-1}) \big)^\top \big( W_P^{(l)} \operatorname{diag}(\gamma_{\mathrm{attn}}^{(l)} \oslash \gamma_{L-1}) \big)$$
The eigenvectors $\{w_{\mathrm{readout}}^{(m)}\}_{m=1}^M$ corresponding to the smallest eigenvalues of $\Omega_{\mathrm{read}}$ define **mutually orthogonal, zero-clutter readout directions** at Layer $L-1$, and their pre-images:
$$\mathbf{v}_{\mathrm{beacon}}^{(m)} = \operatorname{normalize}\!\big( w_{\mathrm{readout}}^{(m)} \oslash \gamma_{L-1} \big)$$
define the corresponding **Nullspace Beacon vectors** written into $w_{\mathrm{down}}$ of the sensor channels at Layer $l_{\mathrm{sensor}}$.

---

## 4. Physical Capacity Bounds & Operating Envelope (For `8B` Models, $d=4096$, $d_{\mathrm{ff}} \in [12288, 14336]$)

1. **Truly Dormant Channels per Layer**:
   - **`300 – 450` channels per MLP layer** (`~2.5% – 3.5%` of $d_{\mathrm{ff}}$) satisfy both minimal weight norm and near-zero activation on general text. Modifying beyond `~500` channels in a single layer begins to encroach on low-frequency active circuits.
   - Across 12 mid-to-deep sensor layers (`Layers 18–30`), this yields **`~4,200` sensor channels**; across the final 1–2 actuator layers, **`~400 – 800` actuator channels**.
2. **Orthogonal Nullspace Beacon Channels in the Residual Stream**:
   - Out of $d = 4096$ residual dimensions, removing the top natural activation subspace and high-sensitivity intermediate $K/V$ directions leaves **`256 – 512` orthogonal beacon directions** maintaining $\mathrm{SNR} > 10\times$.
3. **Data & Skill Capacity by Task Regime**:
   - **Regime A — Verbatim Fact / Fixed-String Memorization (1 neuron per output token)**:
     Limited by terminal actuator channels to **`40 – 80` distinct verbatim facts/strings** (verified at `100%` accuracy up to $N=60$ novel facts). WSP is **not** intended to replace RAG or pretraining for bulk encyclopedic memorization.
   - **Regime B — Composable Skill Routing, Safety Alignment & Stem-Release Dynamic Generation (Shared Actuator Stems + Multi-Cone Sensors)**:
     Because thousands of training queries share a compact set of action stems (e.g., `3–6` actuator neurons per tool/skill stem, after which native circuits generate dynamic SQL/Python arguments or step-by-step answers) while phrasing diversity is absorbed by mid-layer sensor cones (`2–4` sensor neurons per sub-cluster), a single `8B` model supports **`100 – 150` distinct skills/tools** and **`1,000 – 2,000` semantic phrasing sub-cones** (corresponding to **`5,000 – 20,000` training examples**).
