"""
train.py — Entry point for training MemoryTreeVLA.

Usage:
    python MTVLA/train.py --config MTVLA/configs/default.yaml
"""

import argparse
from configs import load_config
from models import MemoryTreeVLA
from utils import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Train MemoryTreeVLA")
    parser.add_argument("--config", type=str, default="MTVLA/configs/default.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    logger = setup_logger("MTVLA-Train")

    logger.info(f"Loaded config from {args.config}")
    model = MemoryTreeVLA(cfg)
    logger.info(f"Model: {cfg.model.name} | Backbone: {cfg.model.backbone}")

    # TODO: build dataloader, optimizer, trainer and launch training loop


if __name__ == "__main__":
    main()
