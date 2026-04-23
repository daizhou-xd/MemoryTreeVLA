"""
dual_tree_vla.model.common — 通用模型组件

包含:
  FlashMHA              — 自动选择后端的多头注意力
  CrossModalFusion      — 门控跨模态融合
  JumpAwareHead         — 动作突变检测（语义无关）
"""
from .attn import FlashMHA, flash_attn_available, sdpa_available
from .fusion import CrossModalFusion
from .semantic_jump_head import JumpAwareHead, SemanticJumpHead

__all__ = [
    "FlashMHA",
    "flash_attn_available",
    "sdpa_available",
    "CrossModalFusion",
    "JumpAwareHead",
    "SemanticJumpHead",
]
