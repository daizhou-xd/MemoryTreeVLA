"""
DualTreeVLA 策略基类 — 对标 FlowPolicy 的 base_policy.py

所有策略均继承此基类，实现统一的 predict_action / reset / set_normalizer 接口。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional

import torch
import torch.nn as nn


class BasePolicy(ABC, nn.Module):
    """
    抽象策略基类，定义所有策略必须实现的最小接口。

    接口要求
    --------
    predict_action(obs_dict) -> action_dict
        obs_dict : str → Tensor
        返回      : {"action": (B, H_a, d_a)} 字典
    reset()
        重置任何内部状态（如记忆树、历史缓冲区）
    """

    @abstractmethod
    def predict_action(
        self,
        image: torch.Tensor,        # (B, C, H, W)
        instruction: str,
        state: torch.Tensor,        # (B, d_q)
        a_prev: Optional[torch.Tensor] = None,  # (B, d_a) or None
    ) -> Dict[str, torch.Tensor]:
        """
        单步推理。
        Returns: {"action": (B, H_a, d_a)}
        """
        raise NotImplementedError

    def reset(self, batch_size: int = 1) -> None:
        """重置所有跨步持久化状态（记忆树、动作历史等）。"""
        pass

    def get_parameter_groups(self, base_lr: float) -> list:
        """
        返回用于优化器的参数分组列表（可选实现）。
        默认返回单组（全参数，使用 base_lr）。
        """
        return [{"params": list(self.parameters()), "lr": base_lr}]
