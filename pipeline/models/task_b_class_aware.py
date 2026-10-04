import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

# Role IDs for explicit role injection
# 0: Pad / Special tokens
# 1: Title (<T>...</T>)
# 2: Description (<D>...</D>)
# 3: Comment (<C>...</C>)
ROLE_PAD = 0
ROLE_TITLE = 1
ROLE_DESC = 2
ROLE_COMMENT = 3
NUM_ROLES = 4

# Class index ordering for Task B queries:
# 0: non_hate ('no')
# 1: implicit ('yes_implicit')
# 2: explicit ('yes_explicit')
CLASS_NON_HATE = 0
CLASS_IMPLICIT = 1
CLASS_EXPLICIT = 2
NUM_CLASSES = 3


class TaskBDecoderLayer(nn.Module):
    """
    A single Class-Aware Decoder Block composed of:
      1. Multi-Head Cross-Attention (MHCA):
         - Q = queries from previous layer (or initial learned class query bank)
         - K, V = H_final (post-encoder token representations + role embeddings)
      2. Multi-Head Self-Attention (MHSA) Query Interaction:
         - Cross-class boundary calibration (NonHate <-> Implicit <-> Explicit)
    """
    def __init__(
        self,
        d_model: int = 768,
        num_heads: int = 8,
        dropout: float = 0.25,
        use_query_interaction: bool = True
    ):
        super(TaskBDecoderLayer, self).__init__()
        self.use_query_interaction = use_query_interaction

        # Cross-Attention over text sequence tokens
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.layer_norm_cross = nn.LayerNorm(d_model)
        self.dropout_cross = nn.Dropout(dropout)

        # Self-Attention across class queries (inter-class calibration)
        if self.use_query_interaction:
            self.self_attention = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            self.layer_norm_self = nn.LayerNorm(d_model)
            self.dropout_self = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # 1. Multi-Head Cross-Attention
        z_attn, attn_weights = self.cross_attention(
            query=query,
            key=key_value,
            value=key_value,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            average_attn_weights=True  # [B, 3, S]
        )
        z = self.layer_norm_cross(query + self.dropout_cross(z_attn))

        # 2. Multi-Head Self-Attention between class queries
        if self.use_query_interaction:
            z_self, _ = self.self_attention(
                query=z,
                key=z,
                value=z,
                need_weights=False
            )
            z = self.layer_norm_self(z + self.dropout_self(z_self))

        return z, attn_weights


class TaskBClassAwareAttentionModel(nn.Module):
    """
    Task B (Hate Speech) Class-Aware Attention Architecture with Multi-Layer Query Refinement:
      - Backbone: mmBERT Encoder (22 layers) producing H_mmBERT ∈ [B, S, d_model]
      - Post-Encoder Contextual Role Injection: H_final = LayerNorm(H_mmBERT + E_role)
        (Explicit role embeddings for Title vs Description vs Comment injected post-encoder)
      - Consecutive Decoder Stack (num_decoder_layers = 2 default):
        * Layer 1: Coarse grounding of learned class queries over full sequence
        * Layer 2: Targeted contextual re-querying over H_final conditioned on Layer 1 query outputs
      - Query Interaction (MHSA): Inter-label self-attention across 3 class queries
      - Joint Cross-Class Classification Head f_θ -> logits s ∈ [B, 3]
      - Task C Bridge: Hate-Type-Aware Representation h_B = ∑_c (p_c · z'_c) ∈ [B, d_model]
    """
    def __init__(
        self,
        mmbert_model: nn.Module,
        d_model: int = 768,
        num_heads: int = 8,
        dropout: float = 0.25,
        use_query_interaction: bool = True,   # Ablation hypothesis H2 toggle
        num_decoder_layers: int = 2,          # Multi-layer consecutive cross-attention depth
        hidden_dim: Optional[int] = None,
        num_queries: int = NUM_CLASSES,
    ):
        super(TaskBClassAwareAttentionModel, self).__init__()
        self.mmbert = mmbert_model
        self.d_model = d_model
        self.num_heads = num_heads
        self.use_query_interaction = use_query_interaction
        self.num_decoder_layers = max(1, num_decoder_layers)
        self.num_queries = num_queries
        hidden_dim = hidden_dim or d_model // 2

        # ---------------------------------------------------------------------
        # Post-Encoder Contextual Role Injection
        # ---------------------------------------------------------------------
        # E_role for Title, Description, Comment, and Pad (added to H_mmBERT post-encoder)
        self.role_embeddings = nn.Embedding(NUM_ROLES, d_model, padding_idx=ROLE_PAD)
        nn.init.normal_(self.role_embeddings.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.role_embeddings.weight[ROLE_PAD].zero_()
        self.layer_norm_input = nn.LayerNorm(d_model)
        self.dropout_input = nn.Dropout(dropout)

        # ---------------------------------------------------------------------
        # 3 Learned Base Class Queries: [q_NonHate, q_Implicit, q_Explicit]
        # ---------------------------------------------------------------------
        self.query_embeddings = nn.Parameter(torch.empty(self.num_queries, d_model))
        nn.init.normal_(self.query_embeddings, mean=0.0, std=0.02)

        # ---------------------------------------------------------------------
        # Consecutive Cross-Attention Decoder Stack (num_decoder_layers >= 1)
        # ---------------------------------------------------------------------
        self.decoder_layers = nn.ModuleList([
            TaskBDecoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout,
                use_query_interaction=use_query_interaction
            )
            for _ in range(self.num_decoder_layers)
        ])

        # ---------------------------------------------------------------------
        # Joint Cross-Class Classification Head
        # Projects concatenated query representations [B, 3 * d_model] -> [B, NUM_CLASSES]
        # Enables joint comparative reasoning across (NonHate vs. Implicit vs. Explicit)
        # ---------------------------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Linear(self.num_queries * d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_queries)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        role_ids: torch.Tensor,
        return_attention_map: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            input_ids: [B, S]
            attention_mask: [B, S] (1 for valid tokens, 0 for pad)
            role_ids: [B, S] (0: pad, 1: title, 2: desc, 3: comment)
            return_attention_map: Whether to return the [B, 3, S] attention interpretability map
        Returns:
            logits: [B, 3] (raw classification scores s for [no, yes_implicit, yes_explicit])
            h_B: [B, d_model] (Hate-Type-Aware Representation for Task C bridge)
            attn_weights: [B, 3, S] if return_attention_map else None
        """
        B, S = input_ids.shape

        # ---------------------------------------------------------------------
        # Backbone Forward & Post-Encoder Contextual Role Injection
        # ---------------------------------------------------------------------
        backbone_trainable = any(p.requires_grad for p in self.mmbert.parameters())
        with torch.set_grad_enabled(backbone_trainable):
            h_mmbert = self.mmbert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        h_mmbert = h_mmbert.to(torch.float32)  # [B, S, d_model]

        # Post-encoder role injection: add E_role to H_mmBERT, followed by LayerNorm
        e_role = self.role_embeddings(role_ids)  # [B, S, d_model]
        h_final = self.layer_norm_input(h_mmbert + e_role)
        h_final = self.dropout_input(h_final)    # [B, S, d_model]

        # ---------------------------------------------------------------------
        # Consecutive Cross-Attention Decoder Stack (Multi-Hop Query Refinement)
        # ---------------------------------------------------------------------
        # Initialize queries: Q_base [3, d_model] expanded to [B, 3, d_model]
        q = self.query_embeddings.unsqueeze(0).expand(B, -1, -1)  # [B, 3, d_model]
        key_padding_mask = (attention_mask == 0)

        all_attn_weights = []
        for layer in self.decoder_layers:
            # Each consecutive layer takes queries from previous layer (q)
            # and attends over the same contextualized H_final as key/value
            q, layer_attn = layer(
                query=q,
                key_value=h_final,
                key_padding_mask=key_padding_mask,
                need_weights=return_attention_map
            )
            if return_attention_map and layer_attn is not None:
                all_attn_weights.append(layer_attn)

        # Refined query representation: Z' ∈ [B, 3, d_model]
        z_prime = q

        # ---------------------------------------------------------------------
        # Joint Cross-Class Classification Head
        # Concatenate 3 class representations: [B, 3, d_model] -> [B, 3 * d_model]
        # ---------------------------------------------------------------------
        z_flat = z_prime.reshape(B, self.num_queries * self.d_model)  # [B, 3 * d_model]
        s = self.classifier(z_flat)                              # [B, 3] (raw logits)

        # ---------------------------------------------------------------------
        # TASK C BRIDGE: Hate-Type-Aware Representation h_B
        # h_B = ∑_c (p_c · z'_c) ∈ [B, d_model]
        # ---------------------------------------------------------------------
        probs = F.softmax(s, dim=-1).unsqueeze(-1)  # [B, 3, 1]
        h_B = (probs * z_prime).sum(dim=1)          # [B, d_model]

        last_attn = all_attn_weights[-1] if all_attn_weights else None
        return s, h_B, (last_attn if return_attention_map else None)

    @staticmethod
    def decode_predictions(
        logits: torch.Tensor,
        use_hierarchical: bool = True,
        threshold: float = 0.50
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decodes raw logits s ∈ [B, 3] into discrete class indices [B] and probabilities [B, 3].
        
        If use_hierarchical=True:
          Resolves Probability Mass Splitting (where non-hate wins unfairly because hate
          is divided across implicit and explicit).
          Stage 1 (Binary Gate): P(Hate) = P(implicit) + P(explicit) >= threshold
          Stage 2 (Sub-type):    If Hate, argmax(P(implicit), P(explicit)), else 0 ('no')
        """
        probs = F.softmax(logits.float(), dim=-1)  # [B, 3]
        if not use_hierarchical:
            preds = torch.argmax(probs, dim=-1)
            return preds, probs

        # Stage 1: Hate presence
        p_hate = probs[:, 1] + probs[:, 2]
        is_hate = p_hate >= threshold

        # Stage 2: Hate subtype
        fine_pred = torch.where(
            probs[:, 1] >= probs[:, 2],
            torch.ones_like(p_hate, dtype=torch.long),
            torch.full_like(p_hate, 2, dtype=torch.long)
        )
        preds = torch.where(is_hate, fine_pred, torch.zeros_like(p_hate, dtype=torch.long))
        return preds, probs
