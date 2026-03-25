"""Evaluation metrics for VLA benchmarks."""

from typing import List
import numpy as np


class AverageMeter:
    """Tracks a running average."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_success_rate(successes: List[bool]) -> float:
    """Compute task success rate from a list of episode outcomes."""
    if not successes:
        return 0.0
    return float(np.mean(successes))
