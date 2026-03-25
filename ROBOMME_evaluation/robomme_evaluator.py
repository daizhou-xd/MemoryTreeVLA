"""
ROBOMMEEvaluator: Evaluates MemoryTreeVLA on the RoboMME benchmark.

RoboMME covers multiple categories of manipulation tasks with
multi-modal (vision + language) instructions and structured metrics.
"""

import json
from pathlib import Path
from typing import Dict, List

from MTVLA.utils import setup_logger, compute_success_rate

logger = setup_logger("ROBOMME-Eval")


class ROBOMMEEvaluator:
    """
    Evaluates a policy on the RoboMME benchmark.

    Task categories:
      - Pick-and-Place
      - Stacking
      - Assembly
      - Rearrangement
      - Instruction-Following

    Args:
        model:      Policy to evaluate.
        cfg:        Evaluation config.
        result_dir: Directory to save evaluation results.
    """

    TASK_CATEGORIES = [
        "Pick-and-Place",
        "Stacking",
        "Assembly",
        "Rearrangement",
        "Instruction-Following",
    ]

    def __init__(self, model, cfg, result_dir: str = "results/robomme"):
        self.model = model
        self.cfg = cfg
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_category(self, category: str, num_episodes: int = 100) -> Dict:
        """Evaluate on a single RoboMME task category."""
        assert category in self.TASK_CATEGORIES, f"Unknown category: {category}"
        logger.info(f"Evaluating on {category} ({num_episodes} episodes) ...")

        successes: List[bool] = []
        for ep in range(num_episodes):
            success = self._run_episode(category, ep)
            successes.append(success)

        success_rate = compute_success_rate(successes)
        result = {"category": category, "num_episodes": num_episodes,
                  "success_rate": success_rate}
        logger.info(f"{category} success rate: {success_rate:.3f}")
        return result

    def evaluate_all(self, num_episodes: int = 100) -> Dict:
        """Evaluate on all RoboMME task categories."""
        all_results = {}
        for category in self.TASK_CATEGORIES:
            all_results[category] = self.evaluate_category(category, num_episodes)

        # Compute overall average
        overall = sum(r["success_rate"] for r in all_results.values()) / len(all_results)
        all_results["overall"] = overall
        logger.info(f"Overall success rate: {overall:.3f}")
        self._save_results(all_results)
        return all_results

    def _run_episode(self, category: str, episode_idx: int) -> bool:
        """Run one episode for the given category. Returns True if successful."""
        # TODO: build RoboMME env, step with self.model
        return False

    def _save_results(self, results: Dict):
        out_path = self.result_dir / "results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {out_path}")
