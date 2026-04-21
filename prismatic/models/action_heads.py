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
        **_: object,
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
        state = proprio[:,0,:]
        state_tok = self.state_enc(state)  # [B, 1, D_a]
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


def learnable_random_perturbations(seq_len: int, dim: int, device: torch.device, dtype: torch.dtype) -> nn.Parameter:
    random_perturbations = nn.Parameter(torch.zeros(seq_len, dim, device=device, dtype=dtype))
    nn.init.normal_(random_perturbations, mean=0.0, std=0.02)
    return random_perturbations


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).reshape_as(x)

    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        assert dim % 2 == 0, "RoPE head_dim must be even."
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


class MLPResNetBlock(nn.Module):
    """VLA-Adapter residual block with self/action-proprio/task attention."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.ffn = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.ReLU())
        self.num_heads = 8
        self.head_dim = dim // self.num_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        self.gating_factor = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, h_t: torch.Tensor, h_a: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        ratio_g = torch.tanh(self.gating_factor)
        h = torch.cat([h_a, p], dim=1)

        B, T, C = x.shape
        K_a = h.shape[1]
        K_t = h_t.shape[1]

        q = self.q_proj(x)
        k_tokens = self.k_proj(x)
        v_tokens = self.v_proj(x)
        k_action = self.k_proj(h)
        v_action = self.v_proj(h)
        k_task = self.k_proj(h_t)
        v_task = self.v_proj(h_t)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k_tokens = k_tokens.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v_tokens = v_tokens.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k_action = k_action.view(B, K_a, self.num_heads, self.head_dim).transpose(1, 2)
        v_action = v_action.view(B, K_a, self.num_heads, self.head_dim).transpose(1, 2)
        k_task = k_task.view(B, K_t, self.num_heads, self.head_dim).transpose(1, 2)
        v_task = v_task.view(B, K_t, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.cat(
            [
                torch.matmul(q, k_tokens.transpose(-2, -1)),
                torch.matmul(q, k_action.transpose(-2, -1)),
                torch.matmul(q, k_task.transpose(-2, -1)) * ratio_g,
            ],
            dim=-1,
        )
        attn_scores = attn_scores / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_scores, dim=-1)

        v_combined = torch.cat([v_tokens, v_action, v_task], dim=2)
        output = torch.matmul(attn_weights, v_combined)
        output = output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.o_proj(output)
        return self.ffn(output + x)


class MLPResNetBlockPro(nn.Module):
    """VLA-Adapter Pro block: separate self/adapter/task projections with RoPE."""

    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.ffn = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.ReLU())
        self.q_proj = nn.Linear(dim, dim)
        self.k_self = nn.Linear(dim, dim)
        self.v_self = nn.Linear(dim, dim)
        self.k_adapter = nn.Linear(dim, dim)
        self.v_adapter = nn.Linear(dim, dim)
        self.k_task = nn.Linear(dim, dim)
        self.v_task = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        self.gating_factor = nn.Parameter(torch.zeros(1))
        self.rope = RotaryPositionEmbedding(self.head_dim)
        self.film_gen = nn.Sequential(nn.Linear(dim, dim * 2))

    def forward(self, x: torch.Tensor, h_t: torch.Tensor, h_a: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        ratio_g = torch.tanh(self.gating_factor)
        h_adapter = torch.cat((h_a, p), dim=1)
        h_task = h_t
        B, T, C = x.shape
        K_a = h_adapter.size(1)
        K_t = h_task.size(1)

        def reshape_heads(t: torch.Tensor, length: int) -> torch.Tensor:
            return t.view(B, length, self.num_heads, self.head_dim).transpose(1, 2)

        q = reshape_heads(self.q_proj(x), T)
        k_tokens = reshape_heads(self.k_self(x), T)
        v_tokens = reshape_heads(self.v_self(x), T)
        k_adapter = reshape_heads(self.k_adapter(h_adapter), K_a)
        v_adapter = reshape_heads(self.v_adapter(h_adapter), K_a)
        k_task = reshape_heads(self.k_task(h_task), K_t)
        v_task = reshape_heads(self.v_task(h_task), K_t)

        cos_main, sin_main = self.rope(seq_len=T, device=x.device, dtype=x.dtype)
        q, k_tokens = apply_rope(q, k_tokens, cos_main, sin_main)
        cos_a, sin_a = self.rope(seq_len=K_a, device=x.device, dtype=x.dtype)
        _, k_adapter = apply_rope(k_adapter, k_adapter, cos_a, sin_a)
        cos_t, sin_t = self.rope(seq_len=K_t, device=x.device, dtype=x.dtype)
        _, k_task = apply_rope(k_task, k_task, cos_t, sin_t)

        attn_scores = torch.cat(
            [
                torch.matmul(q, k_tokens.transpose(-2, -1)),
                torch.matmul(q, k_adapter.transpose(-2, -1)),
                torch.matmul(q, k_task.transpose(-2, -1)) * ratio_g,
            ],
            dim=-1,
        )
        attn_scores = attn_scores / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_scores, dim=-1)

        v_combined = torch.cat([v_tokens, v_adapter, v_task], dim=2)
        output = torch.matmul(attn_weights, v_combined)
        output = output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.o_proj(output)
        return self.ffn(output + x)


class MLPResNet(nn.Module):
    """VLA-Adapter MLPResNet used by the L1 regression action head."""

    def __init__(
        self,
        num_blocks: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        use_pro_version: bool = False,
    ):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        block_cls = MLPResNetBlockPro if use_pro_version else MLPResNetBlock
        self.mlp_resnet_blocks = nn.ModuleList([block_cls(dim=hidden_dim) for _ in range(num_blocks)])
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, h_a: torch.Tensor, h_t: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(self.layer_norm1(x)))
        max_layer = h_t.shape[1] - 1
        for i, block in enumerate(self.mlp_resnet_blocks):
            layer_idx = min(i + 1, max_layer)
            x = block(x, h_t=h_t[:, layer_idx, :], h_a=h_a[:, layer_idx, :], p=p)
        x = self.layer_norm2(x)
        return self.fc2(x)


class L1RegressionActionHead(nn.Module):
    """VLA-Adapter-style MLPResNet action head adapted to the JEPA-VLA interface."""

    def __init__(
        self,
        d_proprio: int,
        d_action: int,
        d_llm: int = 896,
        hidden_dim: int = 896,
        horizon: int = 8,
        num_blocks: int = 24,
        use_pro_version: bool = True,
    ):
        super().__init__()
        self.action_dim = d_action
        self.hidden_dim = hidden_dim
        self.horizon = horizon
        self.proprio_projector = nn.Linear(d_proprio, d_llm)
        self.model = MLPResNet(
            num_blocks=num_blocks,
            input_dim=d_llm * d_action,
            hidden_dim=hidden_dim,
            output_dim=d_action,
            use_pro_version=use_pro_version,
        )

    def _prepare_hidden_states(
        self,
        z_action: torch.Tensor,
        hidden_states: Optional[tuple[torch.Tensor, ...]],
        task_token_count: Optional[int],
        action_token_count: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden_states is None:
            action_states = z_action[:, None, None, :]
            task_states = z_action[:, None, None, :]
            return action_states, task_states

        stacked = torch.stack(tuple(hidden_states), dim=1)
        action_token_count = max(0, min(action_token_count, stacked.shape[2] - 1))
        if task_token_count is None:
            task_token_count = max(1, stacked.shape[2] - action_token_count - 2)
        task_start = 1
        non_action_end = stacked.shape[2] - action_token_count if action_token_count > 0 else stacked.shape[2]
        task_end = min(non_action_end - 1, task_start + task_token_count)
        task_states = stacked[:, :, task_start:task_end, :]
        if action_token_count > 0:
            action_states = stacked[:, :, -action_token_count:, :]
        else:
            action_states = stacked[:, :, -1:, :]
        return action_states, task_states

    def predict_action(
        self,
        z_action: torch.Tensor,
        proprio: torch.Tensor,
        hidden_states: Optional[tuple[torch.Tensor, ...]] = None,
        task_token_count: Optional[int] = None,
        action_token_count: int = 0,
        phase: str = "Inference",
    ) -> torch.Tensor:
        batch_size = z_action.shape[0]
        device = z_action.device

        if proprio.dim() == 3:
            proprio = proprio[:, 0, :]

        proprio_features = self.proprio_projector(proprio.to(dtype=z_action.dtype))
        proprio_features = proprio_features.unsqueeze(dim=1)
        action_states, task_states = self._prepare_hidden_states(
            z_action, hidden_states, task_token_count, action_token_count=action_token_count
        )

        cond_actions = torch.zeros(
            (batch_size, self.action_dim * self.horizon, self.hidden_dim),
            device=device,
            dtype=z_action.dtype,
        ).detach()
        x = cond_actions.reshape(batch_size, self.horizon, self.action_dim * self.hidden_dim)

        if phase == "Training":
            _, seq_len, dim = x.shape
            perturbations = learnable_random_perturbations(seq_len, dim, device=x.device, dtype=x.dtype)
            x = x + perturbations

        return self.model(x, h_a=action_states, p=proprio_features, h_t=task_states)

    def forward(
        self,
        z_action: torch.Tensor,
        proprio: torch.Tensor,
        action_gt: torch.Tensor,
        hidden_states: Optional[tuple[torch.Tensor, ...]] = None,
        task_token_count: Optional[int] = None,
        action_token_count: int = 0,
        phase: str = "Training",
        **_: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pred = self.predict_action(
            z_action,
            proprio,
            hidden_states=hidden_states,
            task_token_count=task_token_count,
            action_token_count=action_token_count,
            phase=phase,
        )
        return F.l1_loss(pred, action_gt), pred

    @torch.no_grad()
    def sample_action(
        self,
        z_action: torch.Tensor,
        proprio: torch.Tensor,
        num_steps: Optional[int] = None,
        hidden_states: Optional[tuple[torch.Tensor, ...]] = None,
        task_token_count: Optional[int] = None,
        action_token_count: int = 0,
    ) -> torch.Tensor:
        return self.predict_action(
            z_action,
            proprio,
            hidden_states=hidden_states,
            task_token_count=task_token_count,
            action_token_count=action_token_count,
            phase="Inference",
        )


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
