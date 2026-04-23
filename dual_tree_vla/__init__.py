"""
DualTreeVLA — top-level package.

即插即用双树增强模块（CONSTRUCTION.md §1）：
  DualTreeAdapter_Evo1  — 适配 Evo-1/InternVL3-1B 骨架
  SGMTS                 — 语义引导 Mamba 树扫描（视觉树）
  GateFusion            — 门控视觉特征融合
  HierarchicalMemoryTree — 层级语义记忆树
  JumpAwareHead         — 跳变感知头
"""
from .adapter import BaseDualTreeAdapter, DualTreeAdapter_Evo1
from .model import SGMTS, GateFusion, JumpAwareHead, HierarchicalMemoryTree

__all__ = [
    "BaseDualTreeAdapter",
    "DualTreeAdapter_Evo1",
    "SGMTS",
    "GateFusion",
    "JumpAwareHead",
    "HierarchicalMemoryTree",
]
