"""
GateFusion — 门控视觉特征融合 (CONSTRUCTION.md §3.2)

将 SGMTS 输出的语义增强视觉特征 Z_v 与骨架 ViT 原始 patch 特征 V_t
逐元素门控融合，输出增强后的视觉 token V_t'：

    α = σ(W_gate · [Z_v; V_t])  ∈ [0,1]^{N_p × d}
    V_t' = α ⊙ Z_v + (1-α) ⊙ V_t

初始化策略：W_gate 的偏置设为 -5，使 α ≈ 0，保证训练初期双树模块
不扰动骨架行为；随训练推进，α 逐渐对语义关键区域赋予更高权重。
"""
import torch
import torch.nn as nn


class GateFusion(nn.Module):
    """
    门控融合：SGMTS 增强特征 + 骨架 ViT patch 特征。

    Args:
        d_vit : ViT 原始 patch 特征维度（InternVL3-1B ViT 为 1024）
    """

    def __init__(self, d_vit: int = 896):
        super().__init__()
        self.W_gate = nn.Linear(d_vit * 2, d_vit, bias=True)
        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.W_gate.weight)
        nn.init.constant_(self.W_gate.bias, -5.0)

    def forward(
        self,
        Z_v: torch.Tensor,
        V_t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            Z_v : (B, N_p, d_vit)  SGMTS 语义增强视觉特征
            V_t : (B, N_p, d_vit)  骨架 ViT 原始 patch 特征
        Returns:
            V_t' : (B, N_p, d_vit)  门控融合后的增强视觉 token
        """
        gate_input = torch.cat([Z_v, V_t], dim=-1)           # (B, N_p, 2*d_vit)
        alpha = torch.sigmoid(self.W_gate(gate_input))        # (B, N_p, d_vit)
        return alpha * Z_v + (1.0 - alpha) * V_t
