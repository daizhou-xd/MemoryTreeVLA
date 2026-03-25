"""
eval_libero.py — Entry point for LIBERO benchmark evaluation.

Usage:
    python LIBERO_evaluation/eval_libero.py \
        --model_ckpt outputs/checkpoint.pth \
        --config LIBERO_evaluation/configs/libero_eval.yaml
"""

import argparse
from libero_evaluator import LIBEROEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MemoryTreeVLA on LIBERO")
    parser.add_argument("--model_ckpt", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--config", type=str,
                        default="LIBERO_evaluation/configs/libero_eval.yaml")
    parser.add_argument("--suite", type=str, default="all",
                        choices=["all"] + LIBEROEvaluator.TASK_SUITES)
    parser.add_argument("--num_episodes", type=int, default=50)
    return parser.parse_args()


def main():
    args = parse_args()

    # TODO: load model from checkpoint
    model = None

    evaluator = LIBEROEvaluator(model=model, cfg=args, result_dir="results/libero")

    if args.suite == "all":
        results = evaluator.evaluate_all(num_episodes=args.num_episodes)
    else:
        results = evaluator.evaluate_suite(args.suite, num_episodes=args.num_episodes)

    print(results)


if __name__ == "__main__":
    main()
