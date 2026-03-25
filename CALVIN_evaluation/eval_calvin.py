"""
eval_calvin.py — Entry point for CALVIN benchmark evaluation.

Usage:
    python CALVIN_evaluation/eval_calvin.py \
        --model_ckpt outputs/checkpoint.pth \
        --split D->D \
        --num_sequences 1000
"""

import argparse
from calvin_evaluator import CALVINEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MemoryTreeVLA on CALVIN")
    parser.add_argument("--model_ckpt", type=str, required=True)
    parser.add_argument("--config", type=str,
                        default="CALVIN_evaluation/configs/calvin_eval.yaml")
    parser.add_argument("--split", type=str, default="D->D",
                        choices=CALVINEvaluator.SPLITS)
    parser.add_argument("--num_sequences", type=int, default=1000)
    return parser.parse_args()


def main():
    args = parse_args()

    # TODO: load model from checkpoint
    model = None

    evaluator = CALVINEvaluator(
        model=model,
        cfg=args,
        split=args.split,
        result_dir="results/calvin",
    )
    results = evaluator.evaluate(num_sequences=args.num_sequences)
    print(results)


if __name__ == "__main__":
    main()
