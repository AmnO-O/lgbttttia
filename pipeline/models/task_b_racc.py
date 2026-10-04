"""
Task B: Role-Aware Conditional Context Retrieval (RACC) Architecture.
Designed for StereoQueerEval 2027 Subtask B (3-way Hate Speech Classification: no, yes_implicit, yes_explicit).

Inference Flow:
  1. Comment Grounding (Hop 1 on H20): q_0 queries comment tokens under M_comment -> Q_C
  2. Conditional Context Retrieval (Hop 2 on H21): Q_C queries Title and Description independently -> r_T, r_D
  3. Learned Context Source Gating: [alpha_T, alpha_D] = Softmax(f([Q_C, r_T]), f([Q_C, r_D])) -> r_ctx
  4. Relational Gated Fusion: u = [Q_C; r_ctx; Q_C * r_ctx; |Q_C - r_ctx|] -> g = sigmoid(W_g u) -> Q_ctx = Q_C + g * W_r(r_ctx)
  5. Global Verification (Hop 3 on H22): Q_ctx queries entire sequence under M_all -> Q_G
  6. Query Interaction (MHSA) & FFN Refinement -> Z = [z_no, z_implicit, z_explicit]
  7. Classification Head & Task C Representation Bridge
"""

import math
from typing import Dict, Any, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data import ROLE_PAD, ROLE_TITLE, ROLE_DESC, ROLE_COMMENT, NUM_ROLES


class ContextSourceGate(nn.Module):
    """
    Learned scalar gate per class query computing adaptive attention between Title and Description:
      e_T = W_T [LN(Q_C) ; r_T] + b_T
      e_D = W_D [LN(Q_C) ; r_D] + b_D
      [alpha_T, alpha_D] = Softmax([e_T, e_D], dim=-1)
      r_ctx = alpha_T * r_T + alpha_D * r_D
    """
    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.score_title = nn.Linear(hidden_dim * 2, 1)
        self.score_desc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, q_c_norm: torch.Tensor, r_t: torch.Tensor, r_d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # q_c_norm: [B, C, D], r_t: [B, C, D], r_d: [B, C, D]
        feat_t = torch.cat([q_c_norm, r_t], dim=-1) # [B, C, 2D]
        feat_d = torch.cat([q_c_norm, r_d], dim=-1) # [B, C, 2D]

        e_t = self.score_title(feat_t) # [B, C, 1]
        e_d = self.score_desc(feat_d)   # [B, C, 1]

        scores = torch.cat([e_t, e_d], dim=-1) # [B, C, 2]
        alpha = F.softmax(scores, dim=-1)      # [B, C, 2]

        alpha_t = alpha[..., 0:1] # [B, C, 1]
        alpha_d = alpha[..., 1:2] # [B, C, 1]

        r_ctx = alpha_t * r_t + alpha_d * r_d  # [B, C, D]
        return r_ctx, alpha


class RelationalGatedFusion(nn.Module):
    """
    ESIM / BiDAF style relational interaction matching between Comment Query and Retrieved Context:
      u = [Q_C ; r_ctx ; Q_C * r_ctx ; |Q_C - r_ctx|]  in R^{B x C x 4D}
      g = sigmoid(W_g u + b_g)                         in R^{B x C x D}
      Q_ctx = Q_C + g * (W_r r_ctx)                    in R^{B x C x D}
    """
    def __init__(self, hidden_dim: int = 768, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gate_proj = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.context_proj = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, q_c: torch.Tensor, r_ctx: torch.Tensor) -> torch.Tensor:
        # q_c: [B, C, D], r_ctx: [B, C, D]
        diff = torch.abs(q_c - r_ctx)
        prod = q_c * r_ctx
        u = torch.cat([q_c, r_ctx, prod, diff], dim=-1) # [B, C, 4D]

        g = self.gate_proj(u) # [B, C, D]
        r_proj = self.context_proj(r_ctx) # [B, C, D]

        q_ctx = self.layer_norm(q_c + g * r_proj)
        return q_ctx


class TaskBRACCModel(nn.Module):
    """
    RACC: Role-Aware Conditional Context Retrieval for StereoQueerEval Task B.
    
    Supports strict controlled baseline ablations:
      - baseline_mode='racc' (Default): Full Dual-Axis RACC (H20->H21->H22, Comment->Context->Global)
      - baseline_mode='fixed_h22_full': B1 (H22->H22->H22, Full Sequence visibility)
      - baseline_mode='prog_h20_full': B2 (H20->H21->H22, Full Sequence visibility)
      - baseline_mode='fixed_h22_role': B3 (H22->H22->H22, Role Masking visibility)
      - baseline_mode='fixed_h22_cond': B4 (H22->H22->H22, Conditional Context Retrieval)
    """
    def __init__(
        self,
        mmbert_backbone: nn.Module,
        num_classes: int = 3,
        hidden_dim: Optional[int] = None,
        num_heads: int = 8,
        ffn_dim: int = 1536,
        dropout: float = 0.1,
        use_query_interaction: bool = True,
        baseline_mode: str = "racc",
        multiscale_layer_indices: Optional[Tuple[int, ...]] = None,
    ):
        super().__init__()
        self.mmbert = mmbert_backbone
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.use_query_interaction = use_query_interaction
        self.baseline_mode = baseline_mode

        # Detect backbone hidden dim (768 for mmBERT-base)
        if hidden_dim is None:
            if hasattr(self.mmbert, 'config') and hasattr(self.mmbert.config, 'hidden_size'):
                self.hidden_dim = self.mmbert.config.hidden_size
            elif hasattr(self.mmbert, 'config') and hasattr(self.mmbert.config, 'd_model'):
                self.hidden_dim = self.mmbert.config.d_model
            else:
                self.hidden_dim = 768
        else:
            self.hidden_dim = hidden_dim

        self.multiscale_layer_indices = multiscale_layer_indices

        # Post-Encoder Shared Role Embeddings
        self.role_embeddings = nn.Embedding(
            NUM_ROLES,
            self.hidden_dim,
            padding_idx=ROLE_PAD
        )
        nn.init.normal_(self.role_embeddings.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.role_embeddings.weight[ROLE_PAD].zero_()

        self.role_norm = nn.LayerNorm(self.hidden_dim)

        # Base Learned Class Queries: [q_no, q_implicit, q_explicit]
        self.class_queries = nn.Parameter(torch.empty(num_classes, self.hidden_dim))
        nn.init.trunc_normal_(self.class_queries, std=0.02)

        # Stage 1: Comment Grounding Cross-Attention
        self.ca_comment_norm_q = nn.LayerNorm(self.hidden_dim)
        self.ca_comment = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.ca_comment_dropout = nn.Dropout(dropout)

        # Stage 2: Conditional Context Retrieval Cross-Attentions
        self.ca_title_norm_q = nn.LayerNorm(self.hidden_dim)
        self.ca_title = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.ca_desc_norm_q = nn.LayerNorm(self.hidden_dim)
        self.ca_desc = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Stage 2: Context Source Gating & Gated Relational Fusion
        self.context_gate_norm_q = nn.LayerNorm(self.hidden_dim)
        self.context_gate = ContextSourceGate(hidden_dim=self.hidden_dim)
        self.relational_fusion = RelationalGatedFusion(hidden_dim=self.hidden_dim, dropout=dropout)

        # Stage 3: Global Verification Cross-Attention
        self.ca_global_norm_q = nn.LayerNorm(self.hidden_dim)
        self.ca_global = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.ca_global_dropout = nn.Dropout(dropout)

        # Query Interaction: Multi-Head Self-Attention (MHSA between q_no, q_imp, q_exp)
        if self.use_query_interaction:
            self.sa_norm = nn.LayerNorm(self.hidden_dim)
            self.query_sa = nn.MultiheadAttention(
                embed_dim=self.hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            self.sa_dropout = nn.Dropout(dropout)

        # Task-Specific FFN Refinement
        self.ffn_norm = nn.LayerNorm(self.hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, self.hidden_dim),
            nn.Dropout(dropout)
        )

        # Final Normalization & Joint 3-Way Classifier
        self.final_norm = nn.LayerNorm(self.hidden_dim)
        self.classifier = nn.Linear(num_classes * self.hidden_dim, num_classes)

    def _safe_cross_attention(
        self,
        mha_layer: nn.MultiheadAttention,
        q: torch.Tensor,
        kv: torch.Tensor,
        valid_mask: torch.Tensor,
        role_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Executes cross-attention with fail-safe zero-padding protection for empty role masks.
        key_padding_mask: True indicates positions to IGNORE.
        When need_weights=False, PyTorch enables fast fused Scaled Dot-Product Attention (SDPA).
        """
        batch_size, seq_len, _ = kv.shape
        # Base ignore mask: invalid padding positions
        ignore_mask = ~valid_mask # [B, S]

        if role_mask is not None:
            # Additionally ignore tokens that do not belong to the target role
            ignore_mask = ignore_mask | (~role_mask)

        # Check for sequences where ALL tokens are masked out (e.g. empty description)
        all_masked = ignore_mask.all(dim=-1) # [B]
        if all_masked.any():
            # Temporarily unmask first position to avoid NaN in Softmax, will zero out representation afterwards
            safe_ignore_mask = ignore_mask.clone()
            safe_ignore_mask[all_masked, 0] = False
        else:
            safe_ignore_mask = ignore_mask

        # PyTorch MHA expects key_padding_mask of shape [B, S]
        out, attn_weights = mha_layer(
            query=q,
            key=kv,
            value=kv,
            key_padding_mask=safe_ignore_mask,
            need_weights=need_weights
        )

        if all_masked.any():
            out[all_masked] = 0.0
            if attn_weights is not None:
                attn_weights[all_masked] = 0.0

        return out, attn_weights

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        role_ids: torch.Tensor,
        return_diagnostics: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, Any]]]:
        batch_size = input_ids.size(0)

        # 1. Forward through mmBERT backbone with hidden states
        outputs = self.mmbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        has_hidden_states = hasattr(outputs, "hidden_states") and outputs.hidden_states is not None
        if has_hidden_states:
            num_hs = len(outputs.hidden_states)
            indices = self.multiscale_layer_indices
            if indices is None:
                indices = (-3, -2, -1)

            if len(indices) != 3:
                raise ValueError(
                    f"Expected exactly 3 hidden-state indices for RACC 3-hop retrieval, got {indices}"
                )

            h_layers = []
            for idx in indices:
                actual_idx = idx if idx >= 0 else num_hs + idx
                if not (0 <= actual_idx < num_hs):
                    raise IndexError(
                        f"Invalid hidden-state index {idx} (resolved to {actual_idx}) for {num_hs} hidden states."
                    )
                h_layers.append(outputs.hidden_states[actual_idx].to(torch.float32))

            if self.baseline_mode in ["racc", "prog_h20_full"]:
                h_20, h_21, h_22 = h_layers[0], h_layers[1], h_layers[2]
            else:
                # Fixed H22 for B0, B1, B3, B4 baselines (top layer across all 3 hops)
                top_layer = outputs.hidden_states[-1].to(torch.float32)
                h_20, h_21, h_22 = top_layer, top_layer, top_layer
        else:
            h_22 = outputs.last_hidden_state.to(torch.float32)
            h_20, h_21 = h_22, h_22

        # 2. Inject Role Embeddings across multi-scale hidden states
        e_role = self.role_embeddings(role_ids) # [B, S, D]
        h_20_role = self.role_norm(h_20 + e_role)
        h_21_role = self.role_norm(h_21 + e_role)
        h_22_role = self.role_norm(h_22 + e_role)

        # 3. Construct Boolean Visibility Masks
        valid_mask = attention_mask.bool()
        comment_mask = (role_ids == ROLE_COMMENT) & valid_mask
        title_mask = (role_ids == ROLE_TITLE) & valid_mask
        desc_mask = (role_ids == ROLE_DESC) & valid_mask

        # 4. Expand Base Class Queries: [B, 3, D]
        q_0 = self.class_queries.unsqueeze(0).expand(batch_size, -1, -1)

        # =========================================================================
        # STAGE 1: COMMENT GROUNDING (Hop 1)
        # =========================================================================
        use_role_mask_s1 = self.baseline_mode in ["racc", "fixed_h22_role", "fixed_h22_cond"]
        mask_s1 = comment_mask if use_role_mask_s1 else None

        q_0_norm = self.ca_comment_norm_q(q_0)
        attn_out_s1, attn_w_s1 = self._safe_cross_attention(
            self.ca_comment, q_0_norm, h_20_role, valid_mask, role_mask=mask_s1, need_weights=return_diagnostics
        )
        q_c = q_0 + self.ca_comment_dropout(attn_out_s1) # [B, 3, D]

        # =========================================================================
        # STAGE 2: CONDITIONAL CONTEXT RETRIEVAL & RELATIONAL FUSION (Hop 2)
        # =========================================================================
        if self.baseline_mode in ["racc", "fixed_h22_cond"]:
            # Retrieve Title and Description independently conditioned on Q_C
            q_c_norm_t = self.ca_title_norm_q(q_c)
            r_t, attn_w_title = self._safe_cross_attention(
                self.ca_title, q_c_norm_t, h_21_role, valid_mask, role_mask=title_mask, need_weights=return_diagnostics
            )

            q_c_norm_d = self.ca_desc_norm_q(q_c)
            r_d, attn_w_desc = self._safe_cross_attention(
                self.ca_desc, q_c_norm_d, h_21_role, valid_mask, role_mask=desc_mask, need_weights=return_diagnostics
            )

            # Adaptive Source Gating with shared symmetric query normalization
            q_gate = self.context_gate_norm_q(q_c)
            r_ctx, alpha = self.context_gate(q_gate, r_t, r_d)

            # ESIM / BiDAF Relational Matching Gated Fusion
            q_ctx = self.relational_fusion(q_c, r_ctx)
            attn_w_s2 = torch.stack([attn_w_title, attn_w_desc], dim=1) if return_diagnostics else None # [B, 2, 3, S]
        else:
            # Baseline without conditional split: standard full/role cross-attention
            mask_s2 = (title_mask | desc_mask) if self.baseline_mode == "fixed_h22_role" else None
            q_c_norm = self.ca_title_norm_q(q_c)
            attn_out_s2, attn_w_s2 = self._safe_cross_attention(
                self.ca_title, q_c_norm, h_21_role, valid_mask, role_mask=mask_s2, need_weights=return_diagnostics
            )
            q_ctx = q_c + attn_out_s2
            alpha = torch.full((batch_size, self.num_classes, 2), 0.5, device=q_c.device)

        # =========================================================================
        # STAGE 3: GLOBAL VERIFICATION (Hop 3 on H22)
        # =========================================================================
        q_ctx_norm = self.ca_global_norm_q(q_ctx)
        attn_out_s3, attn_w_s3 = self._safe_cross_attention(
            self.ca_global, q_ctx_norm, h_22_role, valid_mask, role_mask=None, need_weights=return_diagnostics # Full sequence visibility
        )
        q_g = q_ctx + self.ca_global_dropout(attn_out_s3) # [B, 3, D]

        # =========================================================================
        # STAGE 4: QUERY INTERACTION (MHSA) & FFN
        # =========================================================================
        if self.use_query_interaction:
            q_sa_norm = self.sa_norm(q_g)
            sa_out, _ = self.query_sa(q_sa_norm, q_sa_norm, q_sa_norm, need_weights=False)
            q_sa = q_g + self.sa_dropout(sa_out)
        else:
            q_sa = q_g

        # Task-specific FFN
        z = self.final_norm(q_sa + self.ffn(self.ffn_norm(q_sa))) # [B, 3, D]

        # =========================================================================
        # STAGE 5: CLASSIFICATION HEAD & TASK C BRIDGE
        # =========================================================================
        z_flat = z.reshape(batch_size, -1) # [B, 3 * D] = [B, 2304]
        logits = self.classifier(z_flat)   # [B, 3]

        # Task C Bridge Representation
        probs = F.softmax(logits, dim=-1)  # [B, 3]
        h_b = torch.bmm(probs.unsqueeze(1), z).squeeze(1) # [B, D]

        diagnostics = None
        if return_diagnostics:
            diagnostics = {
                'alpha_gate': alpha,                      # [B, 3, 2]
                'attn_hop1_comment': attn_w_s1,           # [B, 3, S]
                'attn_hop2_context': attn_w_s2,           # [B, 2, 3, S] or [B, 3, S]
                'attn_hop3_global': attn_w_s3,            # [B, 3, S]
                'refined_queries': z                      # [B, 3, D]
            }

        return logits, h_b, diagnostics
