# Engineering & Research Plan: RACC (Role-Aware Conditional Context Retrieval) for StereoQueerEval Task B

## 1. Executive Summary & Review of the Proposal

### 1.1 Scientific Soundness
The **RACC (Role-Aware Conditional Context Retrieval)** proposal directly addresses the fundamental empirical bottleneck observed in Task B:
$$\text{Macro-F1} \approx 0.590 \quad \text{with} \quad F1(\text{no}) \approx 0.717, \; F1(\text{explicit}) \approx 0.616, \; \mathbf{F1(\text{implicit}) \approx 0.438}$$

Standard dense cross-attention queries the entire token sequence uniformly ($Q \times K^T$), which answers *"Which tokens correlate with this class?"*. This fails on implicit hate because implicit toxicity is fundamentally **asymmetric and contextual**: the comment contains the pragmatic anchor (often sarcastic, mock-polite, or rhetorical), while the video title and description contain the target identity and discourse frame.

RACC decomposes this into a 3-stage asymmetric inference pipeline:
$$\mathbf{Q_0} \xrightarrow[\text{on } \tilde{H}_{20}]{\text{Comment Grounding}} \mathbf{Q_C} \xrightarrow[\text{on } \tilde{H}_{21}]{\text{Conditional Context Retrieval}} \mathbf{Q_{CTX}} \xrightarrow[\text{on } \tilde{H}_{22}]{\text{Global Verification}} \mathbf{Q_G} \xrightarrow{\text{MHSA + FFN}} \mathbf{Z}$$

### 1.2 Two Orthogonal Axes of Ablation
1. **Axis 1 (Conditioning / Visibility Axis)**: $\text{Comment} \to [\text{Title}, \text{Desc}] \to \text{Full Sequence}$
2. **Axis 2 (Backbone Depth Axis)**: $H_{20} \to H_{21} \to H_{22}$ (where $H_{20}, H_{21}$ are sliding-window local transformed layers and $H_{22}$ is global transformed in ModernBERT).

---

## 2. Mathematical Formalization & Exact Tensor Flow

### Notation & Shapes
- $B$: Batch size
- $S$: Sequence length ($S \le 256$)
- $D$: Hidden dimension ($D = 768$)
- $C$: Number of class queries ($C = 3$: `q_no`, `q_implicit`, `q_explicit`)
- $H_{20}, H_{21}, H_{22} \in \mathbb{R}^{B \times S \times D}$: Backbone representations
- $E_{\text{role}} \in \mathbb{R}^{B \times S \times D}$: Shared post-encoder role embeddings

$$\tilde{H}_l = \text{LayerNorm}(H_l + E_{\text{role}}), \quad l \in \{20, 21, 22\}$$

### Stage 1: Comment Grounding (Hop 1 on $\tilde{H}_{20}$)
Let $Q_0 \in \mathbb{R}^{B \times C \times D}$ be the learned base class queries.
$$Q_C = Q_0 + \text{Dropout}\left(\text{CA}\left(\text{LN}(Q_0), \tilde{H}_{20}, \tilde{H}_{20}; M_{\text{comment}}\right)\right)$$
*Output:* $Q_C \in \mathbb{R}^{B \times C \times D}$ (Comment-anchored query states).

### Stage 2: Conditional Context Retrieval (Hop 2 on $\tilde{H}_{21}$)
Using $Q_C$ as the search query:
$$r_T = \text{CA}\left(\text{LN}(Q_C), \tilde{H}_{21}, \tilde{H}_{21}; M_{\text{title}}\right) \in \mathbb{R}^{B \times C \times D}$$
$$r_D = \text{CA}\left(\text{LN}(Q_C), \tilde{H}_{21}, \tilde{H}_{21}; M_{\text{desc}}\right) \in \mathbb{R}^{B \times C \times D}$$

**Learned Source Gate ($\alpha \in \mathbb{R}^{B \times C \times 2}$):**
$$e_T = W_T [\text{LN}(Q_C) \,\|\, r_T] + b_T \in \mathbb{R}^{B \times C \times 1}$$
$$e_D = W_D [\text{LN}(Q_C) \,\|\, r_D] + b_D \in \mathbb{R}^{B \times C \times 1}$$
$$[\alpha_T, \alpha_D] = \text{Softmax}([e_T, e_D], \text{dim}=-1)$$
$$r_{\text{ctx}} = \alpha_T r_T + \alpha_D r_D \in \mathbb{R}^{B \times C \times D}$$

**Relational Gated Fusion (ESIM / BiDAF Matching Vector):**
$$u = [Q_C \,\|\, r_{\text{ctx}} \,\|\, Q_C \odot r_{\text{ctx}} \,\|\, |Q_C - r_{\text{ctx}}|] \in \mathbb{R}^{B \times C \times 4D}$$
$$g = \sigma(W_g u + b_g) \in \mathbb{R}^{B \times C \times D}$$
$$Q_{\text{ctx}} = Q_C + g \odot (W_r r_{\text{ctx}}) \in \mathbb{R}^{B \times C \times D}$$

### Stage 3: Global Verification (Hop 3 on $\tilde{H}_{22}$)
$$Q_G = Q_{\text{ctx}} + \text{Dropout}\left(\text{CA}\left(\text{LN}(Q_{\text{ctx}}), \tilde{H}_{22}, \tilde{H}_{22}; M_{\text{all}}\right)\right) \in \mathbb{R}^{B \times C \times D}$$

### Stage 4: Query Self-Attention (MHSA), FFN & Output
$$Q_{\text{SA}} = Q_G + \text{Dropout}\left(\text{MHSA}\left(\text{LN}(Q_G)\right)\right) \in \mathbb{R}^{B \times C \times D}$$
$$Z = \text{LN}\left(Q_{\text{SA}} + \text{FFN}(\text{LN}(Q_{\text{SA}}))\right) \in \mathbb{R}^{B \times C \times D}$$
$$z_{\text{flat}} = [z_{\text{no}} \,\|\, z_{\text{implicit}} \,\|\, z_{\text{explicit}}] \in \mathbb{R}^{B \times 3D}$$
$$\mathbf{s} = W_{\text{cls}} z_{\text{flat}} + b_{\text{cls}} \in \mathbb{R}^{B \times 3}$$

---

## 3. Engineering Safeguards & Edge Cases

1. **Empty Context Mask Fallback (Zero-division Protection)**:
   If a sample has an empty description or title ($M_{\text{desc}}$ has no active tokens), key-padding masks in PyTorch Cross-Attention would cause $-\infty$ everywhere $\to \text{NaN}$.
   *Safeguard:* If $\sum M_{\text{target}} == 0$ for a sequence in batch, route attention to a designated dummy null token or set $r_{\text{target}} = \mathbf{0}$ with gate weight $\alpha_{\text{target}} = 0$.

2. **AMP FP16 Underflow/Overflow Protection**:
   The vector difference $|Q_C - r_{\text{ctx}}|$ and source gating scores must compute in `float32` prior to softmax/sigmoid to avoid FP16 numerical degradation under mixed precision.

3. **Pre-Norm Consistency**:
   All cross-attention and self-attention operations use Pre-LayerNorm with residual skip connections to preserve gradient highways during backpropagation.

---

## 4. Controlled Baseline Matrix (B0 to B5)

| Model ID | Architecture Name | Decoder Memory | Visibility Schedule | Core Hypothesis Tested |
|---|---|---|---|---|
| **B0** | Current Reference | $H_{22}$ (1 hop) | Full Sequence ($M_{\text{all}}$) | Legacy baseline |
| **B1** | Fixed 3-Hop | $H_{22} \to H_{22} \to H_{22}$ | Full Sequence ($M_{\text{all}} \times 3$) | Effect of decoder capacity alone |
| **B2** | Progressive Memory Only | $H_{20} \to H_{21} \to H_{22}$ | Full Sequence ($M_{\text{all}} \times 3$) | Axis 2 alone (depth hierarchy) |
| **B3** | Role Mask Only (Fixed $H_{22}$) | $H_{22} \to H_{22} \to H_{22}$ | $M_{\text{comment}} \to [M_T, M_D] \to M_{\text{all}}$ | Axis 1 alone (visibility/role schedule) |
| **B4** | Conditional Context (Fixed $H_{22}$) | $H_{22} \to H_{22} \to H_{22}$ | $Q_C \to \text{Context Gate} \to \text{Fusion}$ | Value of conditioned retrieval vs independent |
| **B5** | **Full RACC** | $H_{20} \to H_{21} \to H_{22}$ | **$M_{\text{comment}} \to [M_T, M_D] \to M_{\text{all}}$** | **Full Dual-Axis Synergistic Architecture** |

---

## 5. Diagnostic Suite & Analysis Tools

1. **Context Corruption Test ($\Delta_{\text{context}}$)**:
   Evaluate validation set with true context vs. randomly shuffled YouTube title/description pairs.
   $$\Delta_{\text{context}} = \text{Macro-F1}_{\text{true}} - \text{Macro-F1}_{\text{shuffled}}$$
   A high $\Delta_{\text{context}} > 0$ validates true semantic context reliance.

2. **Hierarchical Diagnostic Decomposition**:
   - **Root Decision**: $F1(\text{no} \text{ vs } \text{hate})$, where $\text{hate} = \{\text{implicit}, \text{explicit}\}$.
   - **Fine-Grained Decision**: $F1(\text{implicit} \text{ vs } \text{explicit})$ on gold hateful subset only.

3. **Multi-Hop Attention Extraction**:
   Export tensor $\mathbf{A} \in [B, 3, 3, S]$ logging per-hop query-to-token attention maps and gate coefficients $[\alpha_T, \alpha_D] \in [B, 3, 2]$ for qualitative inspection.
