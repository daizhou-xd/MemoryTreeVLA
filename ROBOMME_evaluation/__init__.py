"""
ROBOMME Evaluation — evaluates MemoryTreeVLA on the RoboMME benchmark.

RoboMME is a multi-modal robot manipulation evaluation suite covering
diverse tabletop manipulation tasks with rich language instructions.
"""

from .robomme_evaluator import ROBOMMEEvaluator

__all__ = ["ROBOMMEEvaluator"]
