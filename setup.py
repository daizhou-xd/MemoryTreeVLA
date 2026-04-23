"""
DualTreeVLA — setup.py

对标 FlowPolicy 的包安装配置，允许通过 `pip install -e .` 安装开发版。
"""
from setuptools import setup, find_packages

setup(
    name="dual_tree_vla",
    version="0.1.0",
    description="DualTreeVLA: Long-horizon Robot Manipulation via Dual-Tree VLA",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "transformers>=4.40.0,<4.45.0",
        "accelerate>=0.28.0",
        "h5py>=3.9.0",
        "opencv-python>=4.8.0",
        "pyyaml>=6.0",
        "einops>=0.7.0",
        "tqdm>=4.66.0",
        "numpy>=1.24.0,<2.0",
    ],
    extras_require={
        "train": [
            "deepspeed>=0.14.0,<0.15.0",
            "wandb>=0.17.0",
            "ninja>=1.11.0",
        ],
        "eval": [
            "websockets>=12.0",
        ],
    },
)
