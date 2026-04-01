"""
CrossModalFusion — CONSTRUCTION.md Sections 5.2 & 5.4.3.

Architecture (two-layer hierarchy):

Layer 1 — Memory-guided visual aggregation (Section 5.2, first layer):
    Z̃_VM = MultiHeadCrossAttn(Q=Z^M, K=Z^V, V=Z^V)  ∈ R^{N_M × d}
    Memory tokens query visual tokens; only task-relevant visual context
    is pulled into the memory representation.

Layer 2 — Proprioceptive integration (Section 5.2, second layer):
    e^Q_exp = e^Q · 1_{N_M}^T  ∈ R^{N_M × d}
    Z_fused = LayerNorm(Z̃_VM + MLP([Z̃_VM ; e^Q_exp]))  ∈ R^{N_M × d}

Context for action head (Section 5.4.3):
    e^Q_enc = MLP_state(e^Q)  ∈ R^d
    C = [Z_fused ; e^Q_enc]  ∈ R^{(N_M + 1) × d}

Output shape: (B, N_M + 1, d).
"""
import torch
import torch.nn as nn

from models.attn import FlashMHA


# ==============================================================
#  Cross-attention block  (Q ≠ K/V sources)
# ==============================================================

class CrossAttentionBlock(nn.Module):
    """
    Pre-norm cross-attention block.
    Q comes from `z_q`, K/V come from `z_kv`.
    Both are expected to be in the same embedding space (dim d).
    """

    def __init__(self, d: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm_q  = nn.LayerNorm(d)
        self.norm_kv = nn.LayerNorm(d)
        self.attn    = FlashMHA(d, n_heads, dropout=dropout)
        self.norm_ff = nn.LayerNorm(d)
        self.ff      = nn.Sequential(
            nn.Linear(d, d * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d * 4, d),
        )

    def forward(
        self,
        z_q:  torch.Tensor,   # (B, N_q,  d)
        z_kv: torch.Tensor,   # (B, N_kv, d)
    ) -> torch.Tensor:
        # Cross-attention: Q=z_q, K=V=z_kv  (residual on z_q)
        q  = self.norm_q(z_q)
        kv = self.norm_kv(z_kv)
        z_q = z_q + self.attn(q, kv, kv)
        # Feed-forward
        z_q = z_q + self.ff(self.norm_ff(z_q))
        return z_q


# ==============================================================
#  CrossModalFusion
# ==============================================================

class CrossModalFusion(nn.Module):
    """
    Parameters
    ----------
    d_ssm    : memory SSM output dim
    d_visual : SGMTS output dim
    d_lang   : LLM embedding dim  (accepted but not separately projected —
               language context is absorbed by Z^M via tree dynamics)
    d_q      : proprioceptive state dim
    d        : unified fusion dim (= output dim)
    n_heads  : number of attention heads
    n_layers : depth of cross-attention stack (Layer 1)
    dropout  : dropout probability
    """

    def __init__(
        self,
        d_ssm: int = 256,
        d_visual: int = 256,
        d_lang: int = 896,
        d_q: int = 84,
        d: int = 256,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d = d

        # ── Input projections ────────────────────────────────────────
        self.mem_proj = nn.Linear(d_ssm,    d)
        self.vis_proj = nn.Linear(d_visual, d)

        # ── Layer 1: memory-guided visual aggregation ─────────────────
        # Q = Z^M (memory tokens), K/V = Z^V (visual tokens)
        self.cross_blocks = nn.ModuleList([
            CrossAttentionBlock(d, n_heads, dropout) for _ in range(n_layers)
        ])

        # ── Layer 2: proprioceptive integration (Section 5.2) ─────────
        # MLP([Z̃_VM ; e^Q_exp]) → residual on Z̃_VM, then LayerNorm
        self.prop_proj = nn.Linear(d_q, d)            # e^Q → d
        self.prop_mlp  = nn.Sequential(               # MLP([Z̃_VM ; e^Q_exp])
            nn.Linear(d * 2, d * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d * 4, d),
        )
        self.out_norm = nn.LayerNorm(d)

        # ── Action-head context token (Section 5.4.3) ─────────────────
        # e^Q_enc = MLP_state(e^Q)  appended as extra token to Z_fused
        self.state_mlp = nn.Sequential(
            nn.Linear(d_q, d),
            nn.GELU(),
            nn.Linear(d, d),
        )

    def forward(
        self,
        Z_M:    torch.Tensor,   # (B, N_M, d_ssm)
        Z_V:    torch.Tensor,   # (B, P,   d_visual)
        lang_g: torch.Tensor,   # (B, d_lang)  — kept for API compat
        q:      torch.Tensor,   # (B, d_q)
    ) -> torch.Tensor:
        """
        Returns C : (B, N_M + 1, d) — context for the action head.

        Section 5.2:
            Layer 1  : Z̃_VM = CrossAttn(Q=Z^M, K=V=Z^V)
            Layer 2  : Z_fused = LN(Z̃_VM + MLP([Z̃_VM ; e^Q_exp]))
        Section 5.4.3:
            C = [Z_fused ; e^Q_enc]
        """
        # Project to shared dim d
        z_m = self.mem_proj(Z_M)   # (B, N_M, d)
        z_v = self.vis_proj(Z_V)   # (B, P,   d)

        # ── Layer 1: cross-attention  Q=Z^M, K/V=Z^V ─────────────────
        Z_tilde = z_m
        for blk in self.cross_blocks:
            Z_tilde = blk(Z_tilde, z_v)   # (B, N_M, d)

        # ── Layer 2: proprioceptive integration ───────────────────────
        e_q = self.prop_proj(q)                                      # (B, d)
        e_q_exp = e_q.unsqueeze(1).expand(-1, Z_tilde.shape[1], -1) # (B, N_M, d)
        Z_fused = self.out_norm(
            Z_tilde + self.prop_mlp(torch.cat([Z_tilde, e_q_exp], dim=-1))
        )                                                            # (B, N_M, d)

        # ── Append proprioceptive token (Section 5.4.3) ───────────────
        e_q_enc = self.state_mlp(q).unsqueeze(1)                     # (B, 1, d)
        C = torch.cat([Z_fused, e_q_enc], dim=1)                     # (B, N_M+1, d)

        return C

