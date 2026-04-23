"""
LinearNormalizer — 线性（均值/方差）归一化工具

对标 FlowPolicy 的 normalizer.py，用于动作和状态的归一化与反归一化。
支持逐通道 fit / normalize / unnormalize。
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


class LinearNormalizer(nn.Module):
    """
    线性归一化器：x_norm = (x - mean) / (std + eps)

    用法
    ----
    normalizer = LinearNormalizer()
    normalizer.fit(data_dict)          # data_dict: {"actions": Tensor, "states": Tensor}
    x_norm = normalizer.normalize(x, key="actions")
    x_orig = normalizer.unnormalize(x_norm, key="actions")
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self._params: Dict[str, nn.ParameterDict] = {}

    def fit(self, data: Dict[str, torch.Tensor]) -> "LinearNormalizer":
        """
        从数据字典中计算每个 key 的均值和标准差并存储。

        data: {key: Tensor of shape (N, *dims) or (N, T, *dims)}
        """
        for key, x in data.items():
            flat = x.reshape(-1, x.shape[-1]).float()
            mean = flat.mean(0)
            std  = flat.std(0).clamp(min=self.eps)
            # 注册为 buffer 以便 state_dict 持久化
            self.register_buffer(f"{key}_mean", mean)
            self.register_buffer(f"{key}_std",  std)
        return self

    def normalize(self, x: torch.Tensor, key: str) -> torch.Tensor:
        mean = getattr(self, f"{key}_mean", None)
        std  = getattr(self, f"{key}_std",  None)
        if mean is None or std is None:
            return x
        return (x - mean.to(x.device)) / (std.to(x.device) + self.eps)

    def unnormalize(self, x: torch.Tensor, key: str) -> torch.Tensor:
        mean = getattr(self, f"{key}_mean", None)
        std  = getattr(self, f"{key}_std",  None)
        if mean is None or std is None:
            return x
        return x * (std.to(x.device) + self.eps) + mean.to(x.device)

    def state_dict_params(self) -> Dict[str, torch.Tensor]:
        """返回所有已 fit 的均值/方差（用于 checkpoint 存储）。"""
        return {k: v for k, v in self._buffers.items()}

    @classmethod
    def from_state_dict(cls, d: Dict[str, torch.Tensor], eps: float = 1e-6) -> "LinearNormalizer":
        inst = cls(eps=eps)
        for k, v in d.items():
            inst.register_buffer(k, v)
        return inst
