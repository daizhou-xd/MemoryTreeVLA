"""
MemoryTreeVLA: Vision-Language-Action model augmented with hierarchical memory tree.
Uses a VLM backbone for perception/language understanding and a policy head for action output.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from .memory_tree import MemoryTree


class MemoryTreeVLA(nn.Module):
    """
    MemoryTreeVLA integrates:
      - A VLM backbone (e.g., LLaVA / OpenVLA) for visual and language encoding
      - A MemoryTree module for hierarchical sub-goal management
      - An action policy head for continuous robot control
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # ---- Memory Tree ----
        self.memory_tree = MemoryTree(
            embed_dim=cfg.model.embed_dim,
            max_depth=cfg.model.max_depth,
        )

        # ---- Visual Encoder (placeholder) ----
        self.visual_encoder = nn.Identity()

        # ---- Language Encoder (placeholder) ----
        self.language_encoder = nn.Identity()

        # ---- Fusion Layer ----
        self.fusion = nn.Sequential(
            nn.Linear(cfg.model.embed_dim * 2, cfg.model.embed_dim),
            nn.GELU(),
        )

        # ---- Action Head ----
        self.action_head = nn.Sequential(
            nn.Linear(cfg.model.embed_dim, cfg.model.action_dim),
        )

    def forward(
        self,
        images: torch.Tensor,
        language_tokens: torch.Tensor,
        proprio: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        visual_feat = self.visual_encoder(images)
        lang_feat = self.language_encoder(language_tokens)
        fused = self.fusion(torch.cat([visual_feat, lang_feat], dim=-1))
        actions = self.action_head(fused)
        return {"actions": actions, "fused_features": fused}

    def reset(self):
        """Reset episode-level state."""
        self.memory_tree.reset()
