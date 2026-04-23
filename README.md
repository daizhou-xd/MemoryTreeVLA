# DualTreeVLA

> **「双树」**：Visual Tree（视觉树）+ Memory Tree（记忆树），两棵树并行驱动长时程机器人操作。

**DualTreeVLA** 是一个即插即用（Plug-and-Play）的双树适配器模块，以零侵入方式叠加在 Evo-1 风格骨架（InternVL3-1B + FlowMatching 动作头）上。外部 Evo-1 仓库仅用于开发阶段的实现对照；当前训练、离线评测和在线推理均使用本仓内迁移后的 `InternVL3Backbone + DualTreeAdapter_Evo1` 主链路。两棵树协同运作：

| 树 | 全称 | 延迟目标 | 职责 |
|----|------|----------|------|
| 🌿 **视觉树** | SGMTS（语义引导 Mamba 树扫描编码器） | **< 5 ms / 帧** | 利用 InternVL3 ViT patch 特征构建语义加权 MST，Tree-SSM O(N_p) 扫描 |
| 🌳 **记忆树** | HMT（层级记忆树） | **< 1 ms / 帧** | 跨帧增量维护层级语义记忆，O(depth) 分支/合并/提升 |

两棵树的输出通过 GateFusion 与 LLM 隐藏状态融合，驱动 Flow Matching 动作头完成连续动作预测。

---

## 目录

1. [项目结构](#项目结构)
2. [环境配置](#环境配置)
3. [数据准备](#数据准备)
4. [模型权重](#模型权重)
5. [配置文件说明](#配置文件说明)
6. [训练](#训练)
   - [三阶段训练总览](#三阶段训练总览)
   - [阶段 0 — 预训练（RoboCerebra）](#阶段-0--预训练robocerebra)
   - [阶段 1 — FlowMatching 热身（LIBERO）](#阶段-1--flowmatching-热身libero)
   - [阶段 2 — 全量微调（LIBERO）](#阶段-2--全量微调libero)
   - [断点续训](#断点续训)
7. [评估](#评估)
   - [预训练评估（树结构可视化）](#预训练评估树结构可视化)
   - [离线轨迹评估](#离线轨迹评估)
   - [LIBERO 仿真成功率评估](#libero-仿真成功率评估)
   - [LIBERO Server/Client 评估](#libero-serverclient-评估)
8. [输出结构](#输出结构)
9. [常见问题](#常见问题)

---

## 项目结构

```
DualTreeVLA/
├── dual_tree_vla/
│   ├── config/
│   │   ├── pretrain.yaml         # 阶段 0：预训练超参
│   │   ├── train_phase1.yaml     # 阶段 1：FlowMatching 热身
│   │   ├── train_phase2.yaml     # 阶段 2：全量微调
│   │   └── default.yaml          # 评估默认超参
│   ├── adapter/
│   │   ├── base_adapter.py       # 双树适配器基类
│   │   └── evo1_adapter.py       # DualTreeAdapter_Evo1（主模型）
│   ├── dataset/
│   │   ├── libero.py             # LIBERO LeRobot 格式加载器
│   │   ├── robocerebra.py        # RoboCerebra 训练集加载器
│   │   └── robocerebra_bench.py  # RoboCerebraBench 评测加载器
│   ├── losses/
│   │   └── tree_losses.py        # l_boundary / l_sem / l_elev
│   └── model/
│       ├── action_head/
│       │   └── flow_matching.py  # FlowMatchingActionHead
│       ├── common/
│       │   ├── gate_fusion.py    # GateFusion（门控视觉-语言融合）
│       │   └── semantic_jump_head.py  # JumpAwareHead
│       ├── memory_tree/          # 🌳 记忆树
│       │   ├── node.py
│       │   ├── tree.py           # HierarchicalMemoryTree（含 to_json_dict）
│       │   ├── operations.py     # MLPElevation / semantic_elevation
│       │   └── tree_ssm.py       # TreeSSMReadout
│       └── sgmts/                # 🌿 视觉树
│           └── sgmts.py          # SGMTSEncoder（含 return_attn / sigma_maps）
├── pretrain.py                   # 阶段 0 训练入口
├── train.py                      # 阶段 1 / 2 训练入口
├── eval.py                       # 离线轨迹评估入口（与训练同主链路）
├── scripts/
│   ├── pretrain.sh               # 阶段 0 多卡启动脚本
│   ├── train_phase1.sh           # 阶段 1 多卡启动脚本
│   ├── train_phase2.sh           # 阶段 2 多卡启动脚本
│   ├── pretrain_eval.py          # 预训练评估（树 JSON + heatmap）
│   ├── eval_server.py            # LIBERO Server/Client：多视角推理服务端
│   └── eval_client.py            # LIBERO Server/Client：Evo1 风格多视角仿真客户端
├── data/
│   ├── libero/                   # LIBERO LeRobot 格式数据集
│   └── RoboCerebra/
│       ├── RoboCerebra_trainset/ # 预训练用（3 场景）
│       └── RoboCerebraBench/     # 评测用（6 子集）
├── model_weights/
│   ├── Qwen2.5-0.5B/             # Evo-1 LLM 骨架（InternVL3-1B）
│   └── Qwen2.5-1.5B-Instruct/
├── results/                      # 评估输出（自动创建）
│   └── pretrain_eval/
│       └── ep005/
│           ├── trees/            # 记忆树 JSON
│           └── heatmaps/         # 视觉树 σ 权重叠加图
├── requirements.txt
└── CONSTRUCTION.md               # 双树架构详细设计文档
```

---

## 环境配置

### 硬件要求

| 组件 | 最低 | 推荐 |
|---|---|---|
| GPU | 1× RTX 3090 24G | **4-8× RTX A6000 48G** |
| CPU | 16 核 | 64 核 |
| 内存 | 64 GB | 256 GB |
| 存储 | 200 GB SSD | 1 TB NVMe |

### 安装

```bash
conda create -n dualtree python=3.10 -y
conda activate dualtree

git clone <YOUR_REPO_URL> DualTreeVLA
cd DualTreeVLA

# PyTorch 2.2 + CUDA 12.1（先装 torch，再装 flash-attn）
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121

# 项目依赖
pip install -r requirements.txt

# Flash Attention（可选，自动回退到 SDPA）
pip install flash-attn --no-build-isolation

# DeepSpeed（Phase 2 多卡推荐）
DS_BUILD_OPS=1 pip install deepspeed --no-build-isolation
```

验证：

```bash
python -c "
import torch
print('CUDA:', torch.cuda.is_available())
print('GPU count:', torch.cuda.device_count())
print('BF16:', torch.cuda.is_bf16_supported())
"
```

### LIBERO 仿真环境（仅仿真评估需要）

```bash
pip install robosuite==1.4.0
pip install bddl==1.0.1 easydict einops thop cloudpickle "gym==0.25.2" future matplotlib
pip install -e data/libero/LIBERO

# 无头服务器渲染（无显示器时必须设置）
export MUJOCO_GL=osmesa
# 若 osmesa 未安装：conda install -c conda-forge mesalib -y
```

---

## 数据准备

### RoboCerebra 训练集（阶段 0 预训练）

```
data/RoboCerebra/RoboCerebra_trainset/
├── coffee_table/
│   ├── case1/
│   │   ├── demo.hdf5              # 动作(T,7) + 状态(T,7)
│   │   ├── case1.mp4              # RGB 视频
│   │   └── task_description.json  # 子任务标注
│   └── ...
├── kitchen_table/
└── study_table/
```

下载：

```bash
pip install -U huggingface_hub
huggingface-cli download qiukingballball/RoboCerebra \
    --repo-type dataset \
    --include "RoboCerebra_trainset/**" \
    --local-dir data/RoboCerebra
```

验证：

```bash
python -c "
from dual_tree_vla.dataset import RoboCerebraDataset
ds = RoboCerebraDataset('data/RoboCerebra/RoboCerebra_trainset', subsample=4)
print(f'Trajectories: {len(ds)}')
s = ds[0]
print('frames:', s['frames'].shape, 'actions:', s['actions'].shape)
"
```

### RoboCerebraBench（评估用）

```bash
huggingface-cli download qiukingballball/RoboCerebra \
    --repo-type dataset \
    --include "RoboCerebraBench/**" \
    --local-dir data/RoboCerebra
```

包含六个测试子集：`Ideal` / `Memory_Execution` / `Memory_Exploration` / `Mix` / `Observation_Mismatching` / `Random_Disturbance`。

### LIBERO（阶段 1 / 2 训练）

```bash
python -c "
from huggingface_hub import snapshot_download
for name in ['libero_10_image', 'libero_spatial_image', 'libero_object_image', 'libero_goal_image']:
    snapshot_download(f'lerobot/{name}', repo_type='dataset', local_dir=f'data/libero/{name.replace(\"_image\",\"\")}')
print('Done')
"
```

---

## 模型权重

DualTreeVLA 基于 **Evo-1** 骨架，需要 InternVL3-1B 权重（通过 `Qwen2.5-0.5B` 路径加载）：

```bash
# 方式一：ModelScope（国内推荐）
pip install modelscope
python -c "
from modelscope import snapshot_download
snapshot_download('Qwen/Qwen2.5-0.5B', cache_dir='model_weights')
"

# 方式二：HuggingFace CLI
huggingface-cli download Qwen/Qwen2.5-0.5B \
    --local-dir model_weights/Qwen2.5-0.5B

# 国内镜像加速
export HF_ENDPOINT=https://hf-mirror.com
```

> Evo-1 的 CLIP ViT 通过 InternVL3 内置的 `extract_feature()` 提取 patch 特征，无需单独下载 CLIP 权重。

---

## 配置文件说明

所有配置文件位于 `dual_tree_vla/config/`：

| 文件 | 用途 | 可训练模块 | 损失函数 |
|---|---|---|---|
| `pretrain.yaml` | 阶段 0：预训练 | SGMTS, GateFusion, JumpAwareHead, TreeSSM, MLPElevation, sem_proj, mem_proj | $L_\text{boundary} + 0.5L_\text{sem} + 0.2L_\text{elev}$ |
| `train_phase1.yaml` | 阶段 1：FlowMatching 热身 | GateFusion, FlowMatchingActionHead | $L_\text{flow}$ |
| `train_phase2.yaml` | 阶段 2：全量微调 | 全部模块（LLM 0.1× LR） | $L_\text{flow}$ |
| `default.yaml` | 评估默认 | — | — |

关键参数（`pretrain.yaml` 模型块）：

```yaml
model:
  vlm_path:       "model_weights/Qwen2.5-0.5B"  # Evo-1 VLM 路径
  d_vit:          896      # InternVL3-1B ViT 隐藏维度
  d_a:            7        # 动作维度
  mount_tau:      0.4      # HMT 语义挂载阈值
  max_tree_depth: 4        # 记忆树最大深度
  alpha:          0.5      # SGMTS MST 边权语义偏置
  delta_w:        0.1      # HMT 叶节点权重增量
```

---

## 训练

### 三阶段训练总览

```
阶段 0：预训练（RoboCerebra）
  ├─ 骨架全冻结（ViT + LLM + FlowHead）
  ├─ 训练：SGMTS, GateFusion, JumpAwareHead, TreeSSMReadout,
  │        MLPElevation, sem_proj, mem_proj
  ├─ 损失：L_boundary + 0.5×L_sem + 0.2×L_elev
  └─ 输出：data/outputs/pretrain/pretrain_best.pt
          results/pretrain_eval/  (树 JSON + heatmap)
          ↓
阶段 1：FlowMatching 热身（LIBERO）
  ├─ 加载 pretrain_best.pt
  ├─ 冻结：ViT, LLM, 双树语义模块
  ├─ 训练：GateFusion, FlowMatchingActionHead
  ├─ 损失：L_flow (Flow Matching CFM)
  └─ 输出：data/outputs/phase1/phase1_best.pt
          ↓
阶段 2：全量微调（LIBERO）
  ├─ 加载 phase1_best.pt
  ├─ 全部模块可训练（LLM 以 0.1× LR）
  ├─ 损失：L_flow
  └─ 输出：data/outputs/phase2/phase2_best.pt
```

---

### 阶段 0 — 预训练（RoboCerebra）

让视觉树学习语义引导的 patch 扫描，记忆树学习子任务边界检测与语义层级提升。

**单卡调试**：

```bash
conda activate dualtree
cd /path/to/DualTreeVLA

python pretrain.py --config dual_tree_vla/config/pretrain.yaml
```

**多卡训练（推荐，8 GPU）**：

```bash
bash scripts/pretrain.sh

# 指定卡数
bash scripts/pretrain.sh 4
# 等价于：
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/pretrain.sh 4
```

**关键配置参数**（`dual_tree_vla/config/pretrain.yaml`）：

```yaml
train:
  batch_size:    2        # 每卡 batch size
  epochs:        30
  lr:            3.0e-4
  ckpt_dir:      "data/outputs/pretrain"
  save_every:    5        # 每 N epoch 保存 checkpoint
  eval_every:    5        # 每 N epoch 运行一次评估（0 = 禁用）
  eval_samples:  3        # 评估轨迹数
  result_dir:    "results/pretrain_eval"  # 树 JSON + heatmap 输出目录
```

Checkpoint 保存至 `data/outputs/pretrain/`：

```
data/outputs/pretrain/
├── pretrain_ep005.pt
├── pretrain_ep010.pt
└── pretrain_best.pt       ← 最优，供阶段 1 加载
```

---

### 阶段 1 — FlowMatching 热身（LIBERO）

加载预训练权重，冻结双树语义模块，仅训练 GateFusion 和 FlowMatchingActionHead。

> **前置条件**：`data/outputs/pretrain/pretrain_best.pt` 必须存在。

**单卡调试**：

```bash
python train.py --config dual_tree_vla/config/train_phase1.yaml --phase 1
```

**多卡训练（推荐）**：

```bash
bash scripts/train_phase1.sh

# 指定卡数
bash scripts/train_phase1.sh 4
```

Checkpoint 保存至 `data/outputs/phase1/`：

```
data/outputs/phase1/
├── phase1_ep005.pt
└── phase1_best.pt         ← 最优，供阶段 2 加载
```

---

### 阶段 2 — 全量微调（LIBERO）

解冻全部模块（LLM 以 0.1× 学习率微调），使用 DeepSpeed ZeRO-3（大显存场景）。

> **前置条件**：`data/outputs/phase1/phase1_best.pt` 必须存在。

**单卡调试**：

```bash
python train.py --config dual_tree_vla/config/train_phase2.yaml --phase 2
```

**多卡训练（推荐）**：

```bash
bash scripts/train_phase2.sh

# 指定卡数
bash scripts/train_phase2.sh 4
```

Checkpoint 保存至 `data/outputs/phase2/`。

---

### 断点续训

在各阶段的 yaml 中设置 `resume_from`，或通过命令行传入（train.py 支持 `--resume`）：

```bash
# 阶段 0 断点续训
python pretrain.py \
    --config dual_tree_vla/config/pretrain.yaml \
    # 在 pretrain.yaml 中设置 train.resume_from: "data/outputs/pretrain/pretrain_ep015.pt"

# 阶段 1 断点续训
python train.py \
    --config dual_tree_vla/config/train_phase1.yaml \
    --phase 1 \
    --resume data/outputs/phase1/phase1_ep010.pt
```

---

## 评估

### 预训练评估（树结构可视化）

`scripts/pretrain_eval.py` 是独立评估脚本，运行后输出：
- **记忆树 JSON**：每条轨迹的树拓扑、节点统计、边界帧序号
- **视觉树 heatmap**：每帧 SGMTS 的语义重要性 σ 以 Jet 色图叠加在原图上（50% 透明度）

```bash
# 使用已有 checkpoint 评估（推荐先跑完 30 epoch 预训练再评估）
python scripts/pretrain_eval.py \
    --config dual_tree_vla/config/pretrain.yaml \
    --ckpt   data/outputs/pretrain/pretrain_best.pt \
    --out    results/eval_pretrain \
    --device cuda

# 限制轨迹数（快速验证）
python scripts/pretrain_eval.py \
    --config dual_tree_vla/config/pretrain.yaml \
    --ckpt   data/outputs/pretrain/pretrain_best.pt \
    --out    results/eval_pretrain_quick \
    --max_traj 5

# 禁用 heatmap 保存（只保存树 JSON，节省时间）
python scripts/pretrain_eval.py \
    --ckpt   data/outputs/pretrain/pretrain_best.pt \
    --out    results/eval_pretrain \
    --no_heatmap

# 控制 heatmap 采样密度（每 N 帧保存一张）
python scripts/pretrain_eval.py \
    --ckpt   data/outputs/pretrain/pretrain_best.pt \
    --out    results/eval_pretrain \
    --heatmap_step 1    # 每帧都保存
```

**输出结构**：

```
results/eval_pretrain/
├── trees/
│   ├── traj_0000.json     # 完整记忆树拓扑 + 节点统计
│   ├── traj_0001.json
│   └── ...
├── heatmaps/
│   ├── traj_0000/
│   │   ├── frame_0000.png  # SGMTS σ 权重叠加原图
│   │   ├── frame_0004.png
│   │   └── ...
│   └── ...
└── summary.json            # 所有轨迹汇总指标（boundary_F1 等）
```

树 JSON 格式示例：

```json
{
  "traj_id": "traj_0000",
  "instruction": "Pick the mug and place it on the tray.",
  "T": 64,
  "branch_frames": [15, 38],
  "n_nodes": 7,
  "n_leaves": 4,
  "n_abstract": 3,
  "nodes": {
    "0": {
      "type": "abstract",
      "depth": 0,
      "children_ids": [1, 2],
      "w": 0.85,
      "s_norm": 1.024
    },
    "1": {
      "type": "leaf",
      "depth": 1,
      "n_actions": 15,
      "a_mean": [0.12, -0.03, 0.08, 0.0, 0.0, 0.0, 1.0],
      "z_v_norm": 0.973
    }
  }
}
```

**pretrain_eval 参数表**：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--config` | `dual_tree_vla/config/pretrain.yaml` | 配置 YAML |
| `--ckpt` | 必填 | 预训练 checkpoint 路径 |
| `--out` | `results/eval_pretrain` | 输出根目录 |
| `--max_traj` | 全部 | 限制评估轨迹数 |
| `--device` | `cuda` | 推理设备 |
| `--heatmap_step` | `4` | 每隔几帧保存一张 heatmap |
| `--no_heatmap` | False | 禁用 heatmap 保存 |

> 预训练循环本身也会在每 `eval_every` epoch 自动调用此逻辑，
> 输出保存至 `results/pretrain_eval/ep{N}/`。

---

### 离线轨迹评估

`eval.py` 以 teacher-forcing 方式对预录轨迹进行离线评估，无需仿真环境。当前实现直接加载仓内 `InternVL3Backbone + DualTreeAdapter_Evo1`，不再实例化旧的 `dual_tree_vla.policy.DualTreeVLA`；对 LIBERO 评估时也会按完整 episode 加载，而不是 step-level 样本。

**评估 RoboCerebraBench**（推荐，6 子集全覆盖）：

```bash
conda activate dualtree
cd /path/to/DualTreeVLA

python eval.py \
    --ckpt      data/outputs/phase2/phase2_best.pt \
    --config    dual_tree_vla/config/default.yaml \
    --dataset   robocerebra_bench \
    --bench_root data/RoboCerebra/RoboCerebraBench \
    --out       results/bench_eval.json

# 只评估部分子集
python eval.py \
    --ckpt      data/outputs/phase2/phase2_best.pt \
    --dataset   robocerebra_bench \
    --bench_root data/RoboCerebra/RoboCerebraBench \
    --task_types Ideal Random_Disturbance \
    --out       results/bench_partial.json

# 快速调试
python eval.py \
    --ckpt      data/outputs/phase2/phase2_best.pt \
    --dataset   robocerebra_bench \
    --bench_root data/RoboCerebra/RoboCerebraBench \
    --max_traj  6 \
    --device    cpu
```

**评估 LIBERO 离线轨迹**：

```bash
# LIBERO-10（长时程任务）
python eval.py \
    --ckpt      data/outputs/phase2/phase2_best.pt \
    --dataset   libero \
    --data_root data/libero \
    --libero_split long \
    --out       results/libero_long_eval.json

# 批量评估所有子集
for SPLIT in spatial object goal long; do
    python eval.py \
        --ckpt      data/outputs/phase2/phase2_best.pt \
        --dataset   libero \
        --data_root data/libero \
        --libero_split ${SPLIT} \
        --out       results/libero_${SPLIT}_eval.json
done
```

**典型输出**：

```
PER TASK-TYPE RESULTS  (RoboCerebraBench)
============================================================
  Ideal                    n=10  L1=0.081  L2=0.153  F1=0.724
  Memory_Execution         n=10  L1=0.089  L2=0.167  F1=0.682
  Memory_Exploration       n=10  L1=0.093  L2=0.176  F1=0.651
  Mix                      n=10  L1=0.102  L2=0.192  F1=0.610
  Observation_Mismatching  n=10  L1=0.097  L2=0.181  F1=0.632
  Random_Disturbance       n=10  L1=0.099  L2=0.186  F1=0.623

OVERALL  L1=0.094  L2=0.176  F1=0.654  mono=0.861
```

**常用参数表**：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--ckpt` | 必填 | `.pt` checkpoint 路径 |
| `--config` | `dual_tree_vla/config/default.yaml` | 模型配置 YAML |
| `--dataset` | 必填 | `robocerebra_bench` / `robocerebra` / `libero` |
| `--bench_root` | — | RoboCerebraBench 目录 |
| `--task_types` | 全 6 类 | 指定子集（空格分隔） |
| `--data_root` | — | robocerebra / libero 数据根目录 |
| `--libero_split` | `long` | `spatial` / `object` / `goal` / `long` |
| `--max_traj` | 全部 | 限制轨迹数（调试用） |
| `--boundary_tol` | `5` | 边界 F1 容忍窗口（步数） |
| `--print_tree` | 关闭 | 打印轨迹结束时的记忆树 ASCII 结构 |
| `--out` | 不保存 | 结果保存为 JSON |

---

### LIBERO 仿真成功率评估

> **前置条件**：按照[安装](#libero-仿真环境仅仿真评估需要)章节完成 LIBERO 仿真依赖安装。

当前仓库没有单文件“单进程 LIBERO 成功率评估”脚本；在线仿真评测使用下面的 WebSocket Server/Client 两段式流程。

---

### LIBERO Server/Client 评估

GPU 推理节点与 MuJoCo 仿真节点通过 WebSocket 分离，适合推理/仿真异机部署。当前在线协议已对齐 Evo1 风格多视角 JSON：客户端发送 `image` 视角列表、`image_mask` 和 `action_mask`，服务端使用仓内 backbone + adapter 主链路解码并推理。

说明：

- `scripts/eval_server.py` 目前支持 argparse 参数：`--ckpt`、`--config`、`--stats`、`--host`、`--port`
- `scripts/eval_client.py` 目前**不支持 argparse**，需要直接编辑脚本顶部的 `Args` 类来设置 `SERVER_URL`、`task_suites`、`num_episodes`、`horizon`、`max_steps`、`ckpt_name` 等参数
- `scripts/eval_client.py` 当前会始终保存视频到 `video_log_file/<ckpt_name>/<suite>/`

**步骤一：启动推理服务端**（GPU 节点）

```bash
conda activate dualtree
cd /path/to/DualTreeVLA

python scripts/eval_server.py \
  --ckpt   <path/to/phase_or_exported_checkpoint.pt> \
    --config dual_tree_vla/config/train_phase2.yaml \
    --port   9000
# 输出：DualTreeVLA server running at ws://0.0.0.0:9000
```

如果工作区中没有 `data/outputs/phase*/` 训练产物，可以先使用你在云端训练得到的 `.pt` checkpoint，或显式传入导出的权重路径。

**步骤二：配置并启动仿真客户端**（另开终端或远程节点）

先编辑 `scripts/eval_client.py` 顶部的 `Args` 类，例如：

```python
class Args():
  horizon      = 16
  max_steps    = [600]
  SERVER_URL   = "ws://127.0.0.1:9000"
  ckpt_name    = "DualTreeVLA_phase2"
  task_suites  = ["libero_10"]
  log_file     = f"./log_file/{ckpt_name}.txt"
  num_episodes = 10
  SEED         = 42
```

然后运行：

```bash
conda activate dualtree
cd /path/to/DualTreeVLA
export MUJOCO_GL=osmesa   # 无头服务器必须设置

python scripts/eval_client.py
```

**服务端参数**：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--ckpt` | 必填 | Phase 2 checkpoint |
| `--config` | `dual_tree_vla/config/train_phase2.yaml` | 模型配置 YAML |
| `--stats` | 自动探测 | 可选的 stats.json 路径 |
| `--port` | `9000` | WebSocket 监听端口 |
| `--host` | `0.0.0.0` | 绑定地址 |

**客户端配置项**（`scripts/eval_client.py` 顶部 `Args` 类） ：

| 字段 | 默认值 | 说明 |
|---|---|---|
| `SERVER_URL` | `ws://127.0.0.1:9000` | 服务端地址 |
| `task_suites` | `['libero_10']` | 要评估的 suite 列表 |
| `num_episodes` | `10` | 每任务 rollout 次数 |
| `horizon` | `16` | 每次推理后执行的动作步数 |
| `max_steps` | `[600]` | 每个 suite 的最大环境步数列表 |
| `ckpt_name` | `DualTreeVLA_phase2` | 日志与视频目录名前缀 |
| `log_file` | `./log_file/<ckpt_name>.txt` | 文本日志输出路径 |
| `SEED` | `42` | 随机种子 |

---

## 输出结构

```
results/
├── pretrain_eval/
│   ├── ep005/
│   │   ├── trees/
│   │   │   ├── traj_0000.json  # 记忆树拓扑 JSON
│   │   │   └── traj_0001.json
│   │   ├── heatmaps/
│   │   │   ├── traj_0000/
│   │   │   │   ├── frame_0000.png  # SGMTS σ 叠加图
│   │   │   │   └── frame_0004.png
│   │   │   └── traj_0001/
│   │   └── summary.json        # 所有轨迹汇总指标
│   └── ep010/
│       └── ...
├── bench_eval.json             # RoboCerebraBench 离线评估结果
├── libero_long_eval.json       # LIBERO 离线评估结果
└── libero10_sim.json           # LIBERO 仿真成功率结果
```

**指标说明**：

| 指标 | 含义 | 目标 |
|---|---|---|
| `action_l1 / l2` | 预测动作与真值逐步偏差 | 越小越好 |
| `tree_nodes` | 轨迹末尾记忆树节点数 | 与任务复杂度匹配 |
| `tree_branches` | 每条轨迹分支次数 | 等于子任务数 |
| `subtask_boundary_f1` | 分支事件与 GT 边界的 F1 | 趋近 1.0 |
| `prog_monotone_rate` | 树中语义进度单调比例 | 趋近 1.0 |
| `boundary_F1` | 预训练评估专用（边界检测 F1） | 趋近 1.0 |

---

## 常见问题

### 1. `NCCL` 通信超时

```bash
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0   # ip link 查看实际网卡名
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=1800
```

### 2. Flash Attention 编译失败

```bash
python -c "import torch; print(torch.version.cuda)"   # 确认 CUDA 版本
nvcc --version
pip uninstall flash-attn -y && pip cache purge
pip install flash-attn --no-build-isolation
```

> Flash Attention 可选，项目会自动回退到 PyTorch SDPA，性能影响极小。

### 3. 显存不足（OOM）

```bash
# 减小 batch_size
train:
  batch_size: 1
  grad_accum: 8   # 保持等效 batch = 8

# Phase 2 已启用 ZeRO-3，仍 OOM 时在 ds_zero3.json 中开启 CPU offload:
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {"device": "cpu"},
    "offload_param": {"device": "cpu"}
  }
}
```

### 4. 模型权重路径错误

config 中 `model.vlm_path` 必须指向已下载的 Qwen2.5-0.5B 目录：

```yaml
model:
  vlm_path: "model_weights/Qwen2.5-0.5B"
```

路径为相对于项目根目录的相对路径，或绝对路径。

### 5. 国内网络下载慢

```bash
export HF_ENDPOINT=https://hf-mirror.com
# 或使用 ModelScope
pip install modelscope
```

### 6. `robosuite` 导入失败

```bash
pip uninstall robosuite -y
pip install robosuite==1.4.0   # 必须 1.4.0，新版移除了 single_arm_env
```

### 7. LIBERO 仿真黑屏 / 无显示

```bash
export MUJOCO_GL=osmesa
# 若 osmesa 未安装：
conda install -c conda-forge mesalib -y
```

### 8. `websockets` 未安装

```bash
pip install "websockets>=12.0"
```

---

## 引用

如果本项目对您的研究有帮助，请考虑引用相关工作。

---
