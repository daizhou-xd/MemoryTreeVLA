"""
eval_robomme.py — Entry point for RoboMME benchmark evaluation.

Usage:
    python ROBOMME_evaluation/eval_robomme.py \
        --model_ckpt outputs/checkpoint.pth \
        --category all \
        --num_episodes 100
"""

import argparse
from robomme_evaluator import ROBOMMEEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MemoryTreeVLA on RoboMME")
    parser.add_argument("--model_ckpt", type=str, required=True)
    parser.add_argument("--config", type=str,
                        default="ROBOMME_evaluation/configs/robomme_eval.yaml")
    parser.add_argument("--category", type=str, default="all",
                        choices=["all"] + ROBOMMEEvaluator.TASK_CATEGORIES)
    parser.add_argument("--num_episodes", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()

    # TODO: load model from checkpoint
    model = None

    evaluator = ROBOMMEEvaluator(model=model, cfg=args, result_dir="results/robomme")

    if args.category == "all":
        results = evaluator.evaluate_all(num_episodes=args.num_episodes)
    else:
        results = evaluator.evaluate_category(args.category,
                                              num_episodes=args.num_episodes)
    print(results)


if __name__ == "__main__":
    main()
