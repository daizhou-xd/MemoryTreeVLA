"""
LIBEROEvaluator: Runs MemoryTreeVLA on LIBERO benchmark tasks and records results.
"""

import json
from pathlib import Path
from typing import Dict, List

from MTVLA.utils import setup_logger, compute_success_rate

logger = setup_logger("LIBERO-Eval")


class LIBEROEvaluator:
    """
    Evaluates a policy on LIBERO task suites.

    Args:
        model:      MemoryTreeVLA policy (or any callable policy).
        cfg:        Evaluation config dict / object.
        result_dir: Directory to save evaluation results.
    """

    TASK_SUITES = ["LIBERO-Spatial", "LIBERO-Object", "LIBERO-Goal", "LIBERO-Long"]

    def __init__(self, model, cfg, result_dir: str = "results/libero"):
        self.model = model
        self.cfg = cfg
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_suite(self, suite_name: str, num_episodes: int = 50) -> Dict:
        """Run evaluation on a single LIBERO task suite."""
        assert suite_name in self.TASK_SUITES, f"Unknown suite: {suite_name}"
        logger.info(f"Evaluating on {suite_name} ({num_episodes} episodes) ...")

        successes: List[bool] = []
        # TODO: replace with actual LIBERO env loop
        for ep in range(num_episodes):
            success = self._run_episode(suite_name, ep)
            successes.append(success)

        success_rate = compute_success_rate(successes)
        result = {"suite": suite_name, "num_episodes": num_episodes,
                  "success_rate": success_rate}
        logger.info(f"{suite_name} success rate: {success_rate:.3f}")
        return result

    def evaluate_all(self, num_episodes: int = 50) -> Dict:
        """Evaluate on all 4 LIBERO task suites."""
        all_results = {}
        for suite in self.TASK_SUITES:
            all_results[suite] = self.evaluate_suite(suite, num_episodes)
        self._save_results(all_results)
        return all_results

    def _run_episode(self, suite_name: str, episode_idx: int) -> bool:
        """Run a single episode. Returns True if successful."""
        # TODO: build env, reset, step through actions from self.model
        return False

    def _save_results(self, results: Dict):
        out_path = self.result_dir / "results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {out_path}")
