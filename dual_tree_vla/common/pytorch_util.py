"""
pytorch_util — PyTorch 通用工具函数

对标 FlowPolicy 的 pytorch_util.py，提供:
  - 随机种子设置
  - 设备获取
  - 参数量统计
  - 混合精度上下文
"""
from __future__ import annotations

import os
import random
from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn as nn


def set_seed(seed: int) -> None:
    """设置全局随机种子（Python / NumPy / PyTorch / CUDA）。"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_str: Optional[str] = None) -> torch.device:
    """
    返回 torch.device。

    device_str=None 时自动选择：cuda > mps > cpu。
    """
    if device_str is not None:
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(
    model: nn.Module,
    trainable_only: bool = False,
) -> int:
    """返回模型参数数量。trainable_only=True 时只计算可训练参数。"""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def print_parameter_summary(model: nn.Module, named_modules: Optional[dict] = None) -> None:
    """
    打印各子模块参数统计表。

    named_modules: {name: module}，默认打印 model 下所有直接子模块。
    """
    if named_modules is None:
        named_modules = dict(model.named_children())
    total_all, train_all = 0, 0
    print("\n── 参数统计 ──────────────────────────────")
    for name, mod in named_modules.items():
        total = sum(p.numel() for p in mod.parameters())
        train = sum(p.numel() for p in mod.parameters() if p.requires_grad)
        status = "TRAIN" if train > 0 else "frozen"
        print(f"  {name:<26}: {total/1e6:6.2f}M  trainable={train/1e6:6.2f}M  [{status}]")
        total_all += total
        train_all += train
    print(f"  {'─'*60}")
    print(f"  {'TOTAL':<26}: {total_all/1e6:6.2f}M  trainable={train_all/1e6:6.2f}M\n")


@contextmanager
def autocast_ctx(dtype: str = "bf16", enabled: bool = True):
    """
    混合精度上下文管理器。

    dtype: "bf16" | "fp16" | "fp32" | "no"
    """
    if not enabled or dtype in ("fp32", "no"):
        yield
        return
    _dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
    with torch.autocast(device_type="cuda", dtype=_dtype):
        yield
