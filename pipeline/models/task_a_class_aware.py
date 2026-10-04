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

# Class index ordering for Task A queries (Binary Stereotype Classification):
# 0: non_stereotype ('no')
# 1: stereotype ('yes')
CLASS_NON_STEREOTYPE = 0
CLASS_STEREOTYPE = 1
NUM_STEREOTYPE_CLASSES = 2


class TaskADecoderLayer(nn.Module):
    """
    Pre-LayerNorm (Pre-LN) Class-Aware Transformer Decoder Block for Task A (Stereotype):
      1. Pre-Norm Cross-Attention (MHCA):
         q_norm = norm_cross(query)
         z = query + Dropout(MHCA(q_norm, H_final, H_final))
      2. Pre-Norm Self-Attention (MHSA - Inter-Query Interaction):
         z_norm = norm_self(z)
         z = z + Dropout(MHSA(z_norm, z_norm, z_norm))
      3. Pre-Norm Position-wise Feed-Forward Network (FFN):
         z_norm_ffn = norm_ffn(z)
         z = z + FFN(z_norm_ffn)
    """
    def __init__(
        self,
        d_model: int = 768,
        num_heads: int = 8,
        d_ffn: Optional[int] = None,
        dropout: float = 0.25,
        use_query_interaction: bool = True
    ):
        super(TaskADecoderLayer, self).__init__()
        self.use_query_interaction = use_query_interaction
        d_ffn = d_ffn or (2 * d_model)

        # 1. Pre-Norm Cross-Attention
        self.norm_cross = nn.LayerNorm(d_model)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.dropout_cross = nn.Dropout(dropout)

        # 2. Pre-Norm Self-Attention across class queries (MHSA)
        if self.use_query_interaction:
            self.norm_self = nn.LayerNorm(d_model)
            self.self_attention = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            self.dropout_self = nn.Dropout(dropout)

        # 3. Pre-Norm Position-wise Feed-Forward Network
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # 1. Multi-Head Cross-Attention (Pre-LN)
        q_norm = self.norm_cross(query)
        z_attn, attn_weights = self.cross_attention(
            query=q_norm,
            key=key_value,
            value=key_value,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            average_attn_weights=True  # [B, 2, S]
        )
        z = query + self.dropout_cross(z_attn)

        # 2. Multi-Head Self-Attention between class queries (Pre-LN)
        if self.use_query_interaction:
            z_norm = self.norm_self(z)
            z_self, _ = self.self_attention(
                query=z_norm,
                key=z_norm,
                value=z_norm,
                need_weights=False
            )
            z = z + self.dropout_self(z_self)

        # 3. Position-wise Feed-Forward Network (Pre-LN)
        z_norm_ffn = self.norm_ffn(z)
        z_ffn = self.ffn(z_norm_ffn)
        z = z + z_ffn

        return z, attn_weights


class TaskAClassAwareAttentionModel(nn.Module):
    """
    Task A (Stereotype Classification) Class-Aware Attention Architecture:
      - Backbone: mmBERT Encoder producing H_mmBERT ∈ [B, S, d_model]
      - Post-Encoder Contextual Role Injection: H_final = LayerNorm(H_mmBERT + E_role)
        (Explicit role embeddings for Title vs Description vs Comment injected post-encoder)
      - Consecutive Pre-LN Decoder Stack (num_decoder_layers >= 1, default: 3 for H20 -> H21 -> H22):
        * Layer 1: Coarse grounding of learned stereotype class queries over full sequence
        * Layer 2-3: Targeted contextual re-querying over H_final conditioned on prior query outputs
      - Query Interaction (MHSA): Inter-query interaction between [q_NonStereotype, q_Stereotype]
      - Final LayerNorm (Pre-LN termination): z' = final_norm(q)
      - Joint Cross-Class Classification Head f_θ -> logits s ∈ [B, 2]
      - Task A Bridge: Stereotype-Aware Representation h_A = ∑_c (p_c · z'_c) ∈ [B, d_model]
    """
    def __init__(
        self,
        mmbert_model: nn.Module,
        d_model: int = 768,
        num_heads: int = 8,
        d_ffn: Optional[int] = None,
        dropout: float = 0.25,
        use_query_interaction: bool = True,
        num_decoder_layers: int = 3,
        use_multiscale_layers: bool = True,
        multiscale_layer_indices: Tuple[int, ...] = (-3, -2, -1),
        hidden_dim: Optional[int] = None,
        num_queries: int = NUM_STEREOTYPE_CLASSES,
    ):
        super(TaskAClassAwareAttentionModel, self).__init__()
        self.mmbert = mmbert_model
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ffn = d_ffn or (2 * d_model)
        self.use_query_interaction = use_query_interaction
        self.num_decoder_layers = max(1, num_decoder_layers)
        self.use_multiscale_layers = use_multiscale_layers
        self.multiscale_layer_indices = multiscale_layer_indices
        self.num_queries = num_queries
        hidden_dim = hidden_dim or d_model // 2

        # ---------------------------------------------------------------------
        # Post-Encoder Contextual Role Injection
        # ---------------------------------------------------------------------
        self.role_embeddings = nn.Embedding(NUM_ROLES, d_model, padding_idx=ROLE_PAD)
        nn.init.normal_(self.role_embeddings.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.role_embeddings.weight[ROLE_PAD].zero_()
        self.layer_norm_input = nn.LayerNorm(d_model)
        self.dropout_input = nn.Dropout(dropout)

        # ---------------------------------------------------------------------
        # 2 Learned Base Class Queries: [q_NonStereotype, q_Stereotype]
        # ---------------------------------------------------------------------
        self.query_embeddings = nn.Parameter(torch.empty(self.num_queries, d_model))
        nn.init.normal_(self.query_embeddings, mean=0.0, std=0.02)

        # ---------------------------------------------------------------------
        # Consecutive Pre-LN Decoder Stack with FFN
        # ---------------------------------------------------------------------
        self.decoder_layers = nn.ModuleList([
            TaskADecoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ffn=self.d_ffn,
                dropout=dropout,
                use_query_interaction=use_query_interaction
            )
            for _ in range(self.num_decoder_layers)
        ])

        # ---------------------------------------------------------------------
        # Final LayerNorm for Pre-LN stack
        # ---------------------------------------------------------------------
        self.final_norm = nn.LayerNorm(d_model)

        # ---------------------------------------------------------------------
        # Joint Cross-Class Classification Head
        # Projects concatenated query representations [B, 2 * d_model] -> [B, 2]
        # ---------------------------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Linear(self.num_queries * d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_queries)
        )

    def train(self, mode: bool = True):
        super(TaskAClassAwareAttentionModel, self).train(mode)
        if mode and not any(p.requires_grad for p in self.mmbert.parameters()):
            self.mmbert.eval()
        return self

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
            return_attention_map: Whether to return the [B, 2, S] attention interpretability map
        Returns:
            logits: [B, 2] (raw classification scores s for [no, yes])
            h_A: [B, d_model] (Stereotype-Aware Representation for Task A bridge)
            attn_weights: [B, 2, S] if return_attention_map else None
        """
        B, S = input_ids.shape

        # ---------------------------------------------------------------------
        # Backbone Forward & Multi-Scale Hidden State Extraction
        # ---------------------------------------------------------------------
        backbone_trainable = any(p.requires_grad for p in self.mmbert.parameters())
        if not backbone_trainable and self.mmbert.training:
            self.mmbert.eval()

        with torch.set_grad_enabled(backbone_trainable):
            try:
                outputs = self.mmbert(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
            except TypeError:
                outputs = self.mmbert(input_ids=input_ids, attention_mask=attention_mask)

        e_role = self.role_embeddings(role_ids)  # [B, S, d_model]

        has_hidden_states = hasattr(outputs, "hidden_states") and outputs.hidden_states is not None
        if has_hidden_states:
            num_hs = len(outputs.hidden_states)
            h_22_pre_norm = outputs.hidden_states[-1].to(torch.float32)

            if self.use_multiscale_layers:
                h_layers = []
                for idx in self.multiscale_layer_indices:
                    actual_idx = idx if idx >= 0 else num_hs + idx
                    actual_idx = max(0, min(actual_idx, num_hs - 1))
                    h_layers.append(outputs.hidden_states[actual_idx].to(torch.float32))

                if len(h_layers) < self.num_decoder_layers:
                    h_layers.extend([h_layers[-1]] * (self.num_decoder_layers - len(h_layers)))
                elif len(h_layers) > self.num_decoder_layers:
                    h_layers = h_layers[:self.num_decoder_layers]
            else:
                h_layers = [h_22_pre_norm] * self.num_decoder_layers
        else:
            last_hidden = getattr(outputs, "last_hidden_state", outputs).to(torch.float32)
            h_layers = [last_hidden] * self.num_decoder_layers

        h_final_stack = [
            self.dropout_input(self.layer_norm_input(h + e_role))
            for h in h_layers
        ]

        # ---------------------------------------------------------------------
        # Consecutive Cross-Attention Decoder Stack
        # ---------------------------------------------------------------------
        q = self.query_embeddings.unsqueeze(0).expand(B, -1, -1)  # [B, 2, d_model]
        key_padding_mask = (attention_mask == 0)

        all_attn_weights = []
        for layer, h_layer_final in zip(self.decoder_layers, h_final_stack):
            q, layer_attn = layer(
                query=q,
                key_value=h_layer_final,
                key_padding_mask=key_padding_mask,
                need_weights=return_attention_map
            )
            if return_attention_map and layer_attn is not None:
                all_attn_weights.append(layer_attn)

        # ---------------------------------------------------------------------
        # Final Pre-LN Normalization & Joint Classification Head
        # ---------------------------------------------------------------------
        z_prime = self.final_norm(q)                                   # [B, 2, d_model]
        z_flat = z_prime.reshape(B, self.num_queries * self.d_model)   # [B, 2 * d_model]
        s = self.classifier(z_flat)                                     # [B, 2] (raw logits)

        # ---------------------------------------------------------------------
        # TASK A BRIDGE: Stereotype-Aware Representation h_A
        # h_A = ∑_c (p_c · z'_c) ∈ [B, d_model]
        # ---------------------------------------------------------------------
        probs = F.softmax(s, dim=-1).unsqueeze(-1)  # [B, 2, 1]
        h_A = (probs * z_prime).sum(dim=1)          # [B, d_model]

        last_attn = all_attn_weights[-1] if all_attn_weights else None
        return s, h_A, (last_attn if return_attention_map else None)

    @staticmethod
    def decode_predictions(
        logits: torch.Tensor,
        threshold: float = 0.50
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decodes raw logits s ∈ [B, 2] into binary discrete predictions [B] and probabilities [B, 2].
        """
        probs = F.softmax(logits.float(), dim=-1)  # [B, 2]
        preds = torch.where(
            probs[:, 1] >= threshold,
            torch.ones(probs.size(0), dtype=torch.long, device=logits.device),
            torch.zeros(probs.size(0), dtype=torch.long, device=logits.device)
        )
        return preds, probs
