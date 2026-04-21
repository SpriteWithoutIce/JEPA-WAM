"""
action_heads.py

Action Head (GR00T-style Flow Matching) and Auxiliary Head (Cross-Attention Decoder)
for JEPA-VLA.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Action Head: GR00T-style Flow Matching
# =============================================================================


class StateEncoder(nn.Module):
    """Encode proprioceptive state into a single token."""

    def __init__(self, d_proprio: int, d_a: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_proprio, d_a),
            nn.SiLU(),
            nn.Linear(d_a, d_a),
        )

    def forward(self, proprio: torch.Tensor) -> torch.Tensor:
        # proprio: [B, D_proprio]
        x = self.encoder(proprio)  # [B, D_a]
        return x.unsqueeze(1)  # [B, 1, D_a]


class NoisyActionEmbed(nn.Module):
    """Embed noisy actions with time-step conditioning and positional embeddings."""

    def __init__(self, d_action: int, d_a: int, horizon: int):
        super().__init__()
        self.action_proj = nn.Linear(d_action, d_a)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, d_a),
            nn.SiLU(),
            nn.Linear(d_a, d_a),
        )
        self.pos_embed = nn.Parameter(torch.randn(horizon, d_a) * 0.02)

    def forward(self, action_noisy: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # action_noisy: [B, H_a, D_action]
        # t: [B]
        x = self.action_proj(action_noisy)  # [B, H_a, D_a]
        t_emb = self.time_mlp(t.unsqueeze(-1))  # [B, D_a]
        x = x + t_emb.unsqueeze(1)  # broadcast time
        x = x + self.pos_embed.unsqueeze(0)  # broadcast position
        return x  # [B, H_a, D_a]


class SelfAttnBlock(nn.Module):
    def __init__(self, d_a: int, n_heads: int, ffn_ratio: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_a)
        self.attn = nn.MultiheadAttention(d_a, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_a)
        self.ffn = nn.Sequential(
            nn.Linear(d_a, d_a * ffn_ratio),
            nn.GELU(),
            nn.Linear(d_a * ffn_ratio, d_a),
        )

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        h = self.norm2(x)
        x = x + self.ffn(h)
        return x


class CrossAttnBlock(nn.Module):
    def __init__(self, d_a: int, d_llm: int, n_heads: int, ffn_ratio: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_a)
        self.cond_proj = nn.Linear(d_llm, d_a)
        self.attn = nn.MultiheadAttention(d_a, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_a)
        self.ffn = nn.Sequential(
            nn.Linear(d_a, d_a * ffn_ratio),
            nn.GELU(),
            nn.Linear(d_a * ffn_ratio, d_a),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # cond: [B, D_llm]
        h = self.norm1(x)
        k_v = self.cond_proj(cond).unsqueeze(1)  # [B, 1, D_a]
        h, _ = self.attn(h, k_v, k_v)  # cross-attn
        x = x + h
        h = self.norm2(x)
        x = x + self.ffn(h)
        return x


class ActionHeadBackbone(nn.Module):
    def __init__(self, d_a: int = 1024, d_llm: int = 896, n_heads: int = 16, num_layers: int = 16, ffn_ratio: int = 4):
        super().__init__()
        blocks = []
        for i in range(num_layers):
            if i % 2 == 0:  # 0, 2, 4, ... (human-indexed odd layers)
                blocks.append(SelfAttnBlock(d_a, n_heads, ffn_ratio))
            else:  # 1, 3, 5, ... (human-indexed even layers)
                blocks.append(CrossAttnBlock(d_a, d_llm, n_heads, ffn_ratio))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, cond)
        return x


class ActionOutputHead(nn.Module):
    def __init__(self, d_a: int, d_action: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_a)
        self.proj = nn.Linear(d_a, d_action)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1 + H_a, D_a]
        x = x[:, 1:, :]  # drop state token
        x = self.norm(x)
        return self.proj(x)  # [B, H_a, D_action]


class ActionHead(nn.Module):
    """
    GR00T-style Flow Matching Action Head.

    Args:
        d_proprio: Dimensionality of proprioceptive input.
        d_action: Dimensionality of action output.
        d_a: Internal dimension of the action head.
        d_llm: Dimensionality of LLM hidden states (for cross-attention conditioning).
        horizon: Number of action steps to predict.
        n_heads: Number of attention heads.
        num_layers: Number of transformer layers (alternating self/cross-attention).
        ffn_ratio: Expansion ratio for FFN hidden dim.
        beta_alpha, beta_beta: Beta distribution parameters for flow-matching time sampling.
    """

    def __init__(
        self,
        d_proprio: int,
        d_action: int,
        d_a: int = 1024,
        d_llm: int = 896,
        horizon: int = 8,
        n_heads: int = 16,
        num_layers: int = 16,
        ffn_ratio: int = 4,
        beta_alpha: float = 1.5,
        beta_beta: float = 1.0,
    ):
        super().__init__()
        self.state_enc = StateEncoder(d_proprio, d_a)
        self.noisy_emb = NoisyActionEmbed(d_action, d_a, horizon)
        self.backbone = ActionHeadBackbone(d_a, d_llm, n_heads, num_layers, ffn_ratio)
        self.out_head = ActionOutputHead(d_a, d_action)
        self.beta_alpha = beta_alpha
        self.beta_beta = beta_beta
        self.horizon = horizon
        self.d_action = d_action

    def forward(
        self,
        z_action: torch.Tensor,
        proprio: torch.Tensor,
        action_gt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Training forward.

        Args:
            z_action: [B, D_llm]  LLM last hidden state.
            proprio:  [B, D_proprio]
            action_gt: [B, H_a, D_action]

        Returns:
            loss, v_pred
        """
        B = proprio.shape[0]
        device = proprio.device

        # Sample flow-matching time step
        t = torch.distributions.Beta(self.beta_alpha, self.beta_beta).sample((B,)).to(device)

        # Interpolate between noise and ground-truth action
        noise = torch.randn_like(action_gt)
        action_noisy = (1 - t[:, None, None]) * noise + t[:, None, None] * action_gt
        velocity_gt = action_gt - noise

        # Tokenize
        state_tok = self.state_enc(proprio)  # [B, 1, D_a]
        act_tok = self.noisy_emb(action_noisy, t)  # [B, H_a, D_a]
        x = torch.cat([state_tok, act_tok], dim=1)  # [B, 1 + H_a, D_a]

        # Backbone
        x = self.backbone(x, cond=z_action)  # [B, 1 + H_a, D_a]

        # Output
        v_pred = self.out_head(x)  # [B, H_a, D_action]

        # Flow-matching loss
        loss = F.mse_loss(v_pred, velocity_gt)
        return loss, v_pred

    @torch.no_grad()
    def sample_action(
        self,
        z_action: torch.Tensor,
        proprio: torch.Tensor,
        num_steps: int = 10,
    ) -> torch.Tensor:
        """
        Inference: Euler integration from pure noise to action.

        Args:
            z_action: [B, D_llm]
            proprio:  [B, D_proprio]
            num_steps: Number of integration steps.

        Returns:
            action: [B, H_a, D_action]
        """
        B = proprio.shape[0]
        device = proprio.device
        H_a, D_action = self.horizon, self.d_action

        x = torch.randn(B, H_a, D_action, device=device)
        dt = 1.0 / num_steps

        for step in range(num_steps):
            t = torch.full((B,), step * dt, device=device)
            state_tok = self.state_enc(proprio)
            act_tok = self.noisy_emb(x, t)
            tokens = torch.cat([state_tok, act_tok], dim=1)
            tokens = self.backbone(tokens, cond=z_action)
            v_pred = self.out_head(tokens)
            x = x + v_pred * dt

        return x


# =============================================================================
# Auxiliary Head: Cross-Attention Decoder for Future JEPA Embedding Prediction
# =============================================================================


class AuxQueries(nn.Module):
    """Learnable queries constructed from view + time + spatial position embeddings."""

    def __init__(self, num_views_max: int = 3, T: int = 4, H: int = 14, W: int = 14, d: int = 768):
        super().__init__()
        self.view_pe = nn.Parameter(torch.randn(num_views_max, d) * 0.02)
        self.time_pe = nn.Parameter(torch.randn(T, d) * 0.02)
        self.spatial_pe = nn.Parameter(torch.randn(H, W, d) * 0.02)
        self.T, self.H, self.W = T, H, W

    def forward(self, B: int, V: int) -> torch.Tensor:
        v = self.view_pe[:V][:, None, None, None, :]  # [V, 1, 1, 1, d]
        t = self.time_pe[None, :, None, None, :]  # [1, T, 1, 1, d]
        s = self.spatial_pe[None, None, :, :, :]  # [1, 1, H, W, d]
        q = v + t + s  # [V, T, H, W, d]
        q = q.reshape(1, V * self.T * self.H * self.W, -1)
        q = q.expand(B, -1, -1).contiguous()  # [B, N_q, d]
        return q


class AuxDecoderBlock(nn.Module):
    """Standard pre-norm transformer decoder block: CrossAttn -> SelfAttn -> FFN."""

    def __init__(self, d: int, n_heads: int, ffn_ratio: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.cross_attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.self_attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.norm3 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, d * ffn_ratio),
            nn.GELU(),
            nn.Linear(d * ffn_ratio, d),
        )

    def forward(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        # Cross-attention: query -> memory
        h = self.norm1(queries)
        h, _ = self.cross_attn(h, memory, memory)
        queries = queries + h

        # Self-attention: query <-> query
        h = self.norm2(queries)
        h, _ = self.self_attn(h, h, h)
        queries = queries + h

        # FFN
        h = self.norm3(queries)
        queries = queries + self.ffn(h)
        return queries


class AuxHead(nn.Module):
    """
    Auxiliary head that predicts future V-JEPA embeddings from LLM hidden states.

    Args:
        d_llm: Dimensionality of LLM hidden states.
        d_jepa: Dimensionality of V-JEPA encoder output (target dimension).
        num_views_max: Maximum number of camera views (for PE reservation).
        d_aux: Internal dimension of the decoder.
        n_heads: Number of attention heads.
        num_layers: Number of decoder layers.
        ffn_ratio: Expansion ratio for FFN.
        T, H, W: Temporal and spatial grid sizes for target embedding.
    """

    def __init__(
        self,
        d_llm: int,
        d_jepa: int,
        num_views_max: int = 3,
        d_aux: int = 768,
        n_heads: int = 12,
        num_layers: int = 12,
        ffn_ratio: int = 4,
        T: int = 4,
        H: int = 14,
        W: int = 14,
    ):
        super().__init__()
        self.queries = AuxQueries(num_views_max, T, H, W, d_aux)
        self.query_proj = nn.Linear(d_aux, d_aux)
        self.memory_proj = nn.Linear(d_llm, d_aux)

        self.blocks = nn.ModuleList([
            AuxDecoderBlock(d_aux, n_heads, ffn_ratio)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(d_aux)
        self.output_proj = nn.Linear(d_aux, d_jepa)

        self.T, self.H, self.W = T, H, W

    def forward(self, llm_hidden: torch.Tensor, V: int) -> torch.Tensor:
        """
        Args:
            llm_hidden: [B, L, D_llm]
            V: Number of views in the current batch.

        Returns:
            out: [B, V, T, H, W, D_jepa]
        """
        B = llm_hidden.shape[0]

        queries = self.queries(B, V)  # [B, N_q, d_aux]
        queries = self.query_proj(queries)
        memory = self.memory_proj(llm_hidden)  # [B, L, d_aux]

        for block in self.blocks:
            queries = block(queries, memory)

        queries = self.final_norm(queries)
        out = self.output_proj(queries)  # [B, N_q, D_jepa]
        out = out.reshape(B, V, self.T, self.H, self.W, -1)
        return out
