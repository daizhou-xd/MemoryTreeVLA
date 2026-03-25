"""Config dataclasses and YAML loader for MTVLA."""

import yaml
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    name: str = "MemoryTreeVLA"
    embed_dim: int = 512
    action_dim: int = 7          # e.g., 6-DOF + gripper
    max_depth: int = 4
    backbone: str = "openvla-7b"


@dataclass
class TrainConfig:
    batch_size: int = 32
    lr: float = 1e-4
    epochs: int = 50
    warmup_steps: int = 1000
    grad_clip: float = 1.0
    output_dir: str = "outputs/"


@dataclass
class DataConfig:
    dataset: str = "libero"
    data_root: str = "data/"
    num_workers: int = 4
    image_size: int = 224


@dataclass
class MTVLAConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)


def load_config(path: str) -> MTVLAConfig:
    """Load a YAML config file and return an MTVLAConfig instance."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    cfg = MTVLAConfig()
    if "model" in raw:
        cfg.model = ModelConfig(**raw["model"])
    if "train" in raw:
        cfg.train = TrainConfig(**raw["train"])
    if "data" in raw:
        cfg.data = DataConfig(**raw["data"])
    return cfg
