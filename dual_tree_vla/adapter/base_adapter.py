"""
DualTreeVLA — 适配器基类 (CONSTRUCTION.md §3.1)

规定与 VLA 骨架交互的两处最小接口：
  1. 视觉特征 Hook  — 提取 ViT patch 特征 P_t（只读）
  2. 融合注入接口  — GateFusion(Z_v, P_t) → V_t'，mem_proj(m_ctx) → 拼接 LLM 序列
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn


class BaseDualTreeAdapter(ABC, nn.Module):
    """
    双树增强适配器抽象基类。

    子类需实现：
      - forward(images, image_mask, prompt, state, actions_gt, mode)
      - inference(images, image_mask, prompt, state)
      - reset(batch_size)  — 重置所有 HMT 实例

    骨架权重加载与适配器完全解耦：
      子类在 __init__ 中调用 backbone.__class__.from_pretrained(...)，
      BaseDualTreeAdapter 不干预任何骨架参数。
    """

    def __init__(self):
        nn.Module.__init__(self)

    @abstractmethod
    def forward(
        self,
        images: list,
        image_mask: torch.Tensor,
        prompt: str,
        state: torch.Tensor,
        actions_gt: Optional[torch.Tensor] = None,
        subtask_ids: Optional[torch.Tensor] = None,
        mode: str = "phase1",
    ) -> Dict[str, torch.Tensor]:
        """训练用 forward，返回损失字典。"""

    @abstractmethod
    def inference(
        self,
        images: list,
        image_mask: torch.Tensor,
        prompt: str,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """推理用 forward，返回预测动作 (B, horizon, d_a)。"""

    @abstractmethod
    def reset(self, batch_size: int = 1) -> None:
        """重置记忆树（跨 episode 调用）。"""

    def freeze_backbone(self, freeze_llm: bool = True, freeze_vit: bool = True):
        """冻结骨架参数（子类可复用此方法）。"""
        backbone = getattr(self, "backbone", None)
        if backbone is None:
            return
        embedder = getattr(backbone, "embedder", None)
        if embedder is None:
            return
        model = getattr(embedder, "model", None)
        if model is None:
            return
        if freeze_vit and hasattr(model, "vision_model"):
            for p in model.vision_model.parameters():
                p.requires_grad = False
        if freeze_vit and hasattr(model, "mlp1"):
            for p in model.mlp1.parameters():
                p.requires_grad = False
        if freeze_llm and hasattr(model, "language_model"):
            for p in model.language_model.parameters():
                p.requires_grad = False
