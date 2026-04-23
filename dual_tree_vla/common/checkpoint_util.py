"""
CheckpointManager — 检查点管理工具

对标 FlowPolicy 的 checkpoint_util.py，支持 best-k 管理、断点续训。
"""
from __future__ import annotations

import heapq
import os
from pathlib import Path
from typing import Dict, Optional

import torch


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    extra: Optional[Dict] = None,
    accel=None,
) -> None:
    """
    保存训练检查点。

    Parameters
    ----------
    path      : 保存路径（.pt 文件）
    model     : 模型（支持 DDP / Accelerate 包装）
    optimizer : 优化器
    epoch     : 当前 epoch
    step      : 当前全局步数
    extra     : 额外信息（loss、配置等）
    accel     : accelerate.Accelerator 实例（可选）
    """
    if accel is not None and hasattr(accel, "unwrap_model"):
        state = accel.unwrap_model(model).state_dict()
    elif hasattr(model, "module"):
        state = model.module.state_dict()
    else:
        state = model.state_dict()

    ckpt = {
        "model":     state,
        "optimizer": optimizer.state_dict(),
        "epoch":     epoch,
        "step":      step,
    }
    if extra:
        ckpt.update(extra)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    strict: bool = True,
) -> Dict:
    """
    加载检查点，返回 ckpt 字典（含 epoch / step）。

    若 strict=False，自动跳过 shape 不匹配的键（partial load）。
    """
    ckpt = torch.load(path, map_location="cpu")

    if "model" in ckpt:
        state = ckpt["model"]
    else:
        # 兼容旧格式（直接保存 state_dict）
        state = ckpt

    if strict:
        model.load_state_dict(state, strict=True)
    else:
        model_sd = model.state_dict()
        compatible = {
            k: v for k, v in state.items()
            if k in model_sd and v.shape == model_sd[k].shape
        }
        model.load_state_dict(compatible, strict=False)

    if optimizer is not None and "optimizer" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except Exception:
            pass  # 优化器状态不兼容时忽略

    return ckpt


class CheckpointManager:
    """
    Best-K 检查点管理器。

    跟踪指标（如 val_loss），保留 top-k 最优检查点，自动删除较差的检查点。

    用法
    ----
    mgr = CheckpointManager(ckpt_dir="data/outputs/phase1", k=3, mode="min")
    mgr.save(model, optimizer, epoch=5, step=1000, metric=0.123)
    """

    def __init__(
        self,
        ckpt_dir: str | Path,
        k: int = 3,
        mode: str = "min",     # "min" 保留最小值，"max" 保留最大值
        prefix: str = "ckpt",
    ):
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.k     = k
        self.mode  = mode
        self.prefix = prefix
        # heap: (score, path)，score 越小越好
        self._heap: list = []

    def _score(self, metric: float) -> float:
        return metric if self.mode == "min" else -metric

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        step: int,
        metric: float,
        accel=None,
        extra: Optional[Dict] = None,
    ) -> Path:
        """保存检查点并管理 top-k。返回保存路径。"""
        fname = self.ckpt_dir / f"{self.prefix}_ep{epoch:04d}_step{step:07d}.pt"
        save_checkpoint(fname, model, optimizer, epoch, step,
                        extra={"metric": metric, **(extra or {})}, accel=accel)

        score = self._score(metric)
        heapq.heappush(self._heap, (score, str(fname)))

        # 超过 k 个时删除最差的
        while len(self._heap) > self.k:
            _, worst_path = heapq.heappop(self._heap)
            try:
                os.remove(worst_path)
            except FileNotFoundError:
                pass

        return fname

    def best_path(self) -> Optional[Path]:
        """返回当前最优检查点路径（min score）。"""
        if not self._heap:
            return None
        return Path(self._heap[0][1])
