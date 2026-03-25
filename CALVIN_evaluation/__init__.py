"""
CALVIN Evaluation — evaluates MemoryTreeVLA on the CALVIN benchmark.

CALVIN measures long-horizon performance as chains of N consecutive tasks
(N = 1, 2, 3, 4, 5) without environment reset.
"""

from .calvin_evaluator import CALVINEvaluator

__all__ = ["CALVINEvaluator"]
