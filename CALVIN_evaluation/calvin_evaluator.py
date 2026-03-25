"""
CALVINEvaluator: Runs MemoryTreeVLA on the CALVIN benchmark and computes
the average sequence length completed (ASLC) metric.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

from MTVLA.utils import setup_logger

logger = setup_logger("CALVIN-Eval")


class CALVINEvaluator:
    """
    Evaluates a policy on the CALVIN benchmark.

    Key metric: average number of consecutive sub-tasks completed (out of 5).
    Reported as SR1 … SR5 (success rate for chains of length 1–5).

    Args:
        model:      Policy to evaluate.
        cfg:        Evaluation config.
        split:      Dataset split to evaluate on (e.g., "D->D", "ABC->D").
        result_dir: Directory to save evaluation logs.
    """

    SPLITS = ["D->D", "ABC->D", "ABCD->D"]

    def __init__(self, model, cfg, split: str = "D->D",
                 result_dir: str = "results/calvin"):
        self.model = model
        self.cfg = cfg
        self.split = split
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(self, num_sequences: int = 1000) -> Dict:
        """
        Evaluate on CALVIN.

        Returns:
            dict with SR1..SR5 and avg_len.
        """
        logger.info(f"Evaluating on CALVIN split={self.split} "
                    f"({num_sequences} sequences) ...")

        chain_lengths: List[int] = []
        for _ in range(num_sequences):
            length = self._run_sequence()
            chain_lengths.append(length)

        results = self._compute_metrics(chain_lengths)
        logger.info(f"CALVIN results: {results}")
        self._save_results(results)
        return results

    def _run_sequence(self) -> int:
        """
        Run one evaluation sequence (up to 5 chained sub-tasks).
        Returns the number of sub-tasks completed before failure.
        """
        # TODO: build CALVIN env, obtain task sequence, step model
        completed = 0
        for task_idx in range(5):
            success = self._run_subtask(task_idx)
            if success:
                completed += 1
            else:
                break
        return completed

    def _run_subtask(self, task_idx: int) -> bool:
        """Execute a single sub-task within a sequence."""
        # TODO: integrate with CALVIN gym environment
        return False

    def _compute_metrics(self, chain_lengths: List[int]) -> Dict:
        import numpy as np
        results: Dict = {}
        for k in range(1, 6):
            results[f"SR{k}"] = float(np.mean([l >= k for l in chain_lengths]))
        results["avg_len"] = float(np.mean(chain_lengths))
        return results

    def _save_results(self, results: Dict):
        out_path = self.result_dir / f"results_{self.split.replace('->', '_')}.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {out_path}")
