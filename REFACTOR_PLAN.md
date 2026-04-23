# DualTreeVLA 项目结构重构计划

> 参考: FlowPolicy（AAAI 2025 Oral）项目结构
> 目标: 使 DualTreeVLA 结构与 FlowPolicy 同等清晰、分层明确
> 状态: **待审阅** — 确认后方可执行

---

## 一、现状问题分析

### 1.1 与 FlowPolicy 的结构对比

| 方面 | FlowPolicy（参考标准）| DualTreeVLA（当前状态）| 问题 |
|------|----------------------|----------------------|------|
| **主入口位置** | `FlowPolicy/train.py` `FlowPolicy/eval.py`（包同级） | `scripts/train.py` `scripts/pretrain.py` `scripts/eval.py`（脚本目录内）| 训练入口混在脚本目录，职责不清 |
| **配置系统** | `flow_policy_3d/config/`（包内 Hydra 配置）| `configs/`（根目录 plain YAML）| 配置与代码包分离，路径引用混乱 |
| **策略抽象层** | `policy/`（独立策略目录，有基类）| 无 policy/ 目录，主模型直接放 `model/` | 缺少策略抽象，不易扩展 |
| **通用工具层** | `model/common/` + `common/`（双层工具）| 无 `common/` 目录 | 通用组件散落，无法复用 |
| **模型内部组织** | `model/flow/`, `model/vision/`, `model/common/`（按功能分组）| `model/attn.py`, `model/fusion.py` 平铺在 `model/` 根目录 | 通用组件与核心组件混放 |
| **损失函数** | `consistencyfm/`（专门模块）| `losses/`（目录存在但**完全为空**）| 损失逻辑散落在训练脚本中 |
| **数据集抽象** | `dataset/base_dataset.py`（抽象基类）| 无基类，三个数据集各自独立 | 缺少统一接口 |
| **Shell 脚本** | `scripts/` 仅放 `.sh`，入口 `.py` 不在 scripts/ | `scripts/` 混放 `.py` 入口和 `.sh` 启动脚本 | 职责混乱 |
| **预训练权重** | `third_party/`（第三方）；模型输出到 `data/outputs/`| `checkpoints/` 既放预训练 backbone 又混用训练输出路径 | 命名语义不明确 |
| **数据目录** | `data/`（统一数据目录，在输出中）| `dataset/` 根目录（与代码包 `dual_tree_vla/dataset/` 命名易混淆）| 代码和数据目录命名冲突 |

### 1.2 核心结构缺陷总结

1. **职责混乱**：`scripts/` 既有入口 Python 文件又有 Shell 脚本
2. **分层缺失**：无 `policy/` 抽象层，无 `common/` 工具层
3. **配置游离**：`configs/` 在根目录，与包代码脱节
4. **命名冲突**：代码包 `dual_tree_vla/dataset/` 与数据目录 `dataset/` 同名
5. **空目录**：`losses/` 存在但无实际代码
6. **组件平铺**：`model/attn.py`, `model/fusion.py` 应归入 `model/common/`

---

## 二、目标结构（对标 FlowPolicy）

```
DualTreeVLA/                              # 项目根目录
│
├── README.md                             # 项目说明
├── CONSTRUCTION.md                       # 双树架构设计文档（保留根目录）
├── requirements.txt                      # Python 依赖
├── setup.py                              # ← 新增：包安装配置
│
├── pretrain.py                           # ← 从 scripts/ 迁移：预训练主入口
├── train.py                              # ← 从 scripts/ 迁移：训练主入口
├── eval.py                               # ← 从 scripts/ 迁移：离线评估主入口
│
├── dual_tree_vla/                        # 核心代码包（内部结构重组）
│   ├── __init__.py
│   │
│   ├── config/                           # ← 从根目录 configs/ 整体迁移
│   │   ├── pretrain.yaml
│   │   ├── train_phase1.yaml
│   │   ├── train_phase2.yaml
│   │   ├── default.yaml
│   │   └── deepspeed/                    # ← 新增子目录（整理 JSON 配置）
│   │       ├── ds_zero2.json
│   │       └── ds_zero3.json
│   │
│   ├── policy/                           # ← 新增：统一策略抽象层
│   │   ├── __init__.py
│   │   ├── base_policy.py                # ← 新增：抽象策略基类
│   │   └── dual_tree_policy.py           # ← model/dual_tree_vla.py 迁移重命名
│   │
│   ├── model/                            # 模型组件（内部重组）
│   │   ├── __init__.py
│   │   ├── action_head/                  # 动作头（保持不变）
│   │   │   ├── __init__.py
│   │   │   └── flow_matching.py
│   │   ├── memory_tree/                  # 记忆树（保持不变）
│   │   │   ├── __init__.py
│   │   │   ├── node.py
│   │   │   ├── tree.py
│   │   │   ├── operations.py
│   │   │   └── tree_ssm.py
│   │   ├── sgmts/                        # 视觉树（保持不变）
│   │   │   ├── __init__.py
│   │   │   └── sgmts.py
│   │   └── common/                       # ← 新增：通用模型组件
│   │       ├── __init__.py
│   │       ├── attn.py                   # ← 从 model/attn.py 迁移
│   │       ├── fusion.py                 # ← 从 model/fusion.py 迁移
│   │       └── semantic_jump_head.py     # ← 从 model/semantic_jump_head.py 迁移
│   │
│   ├── dataset/                          # 数据集（补充基类）
│   │   ├── __init__.py
│   │   ├── base_dataset.py               # ← 新增：抽象数据集基类
│   │   ├── libero.py
│   │   ├── robocerebra.py
│   │   └── robocerebra_bench.py
│   │
│   ├── losses/                           # 损失函数（补充实现）
│   │   ├── __init__.py
│   │   ├── flow_loss.py                  # ← 新增：FlowMatching 损失封装
│   │   └── semantic_loss.py              # ← 新增：边界/语义对齐损失封装
│   │
│   └── common/                           # ← 新增：训练层通用工具
│       ├── __init__.py
│       ├── normalizer.py                 # 数据归一化工具
│       ├── checkpoint_util.py            # 检查点保存/加载
│       └── pytorch_util.py              # PyTorch 工具函数
│
├── scripts/                              # 仅保留 Shell 脚本 + 服务端工具
│   ├── pretrain.sh
│   ├── train_phase1.sh
│   ├── train_phase2.sh
│   ├── eval_server.py                    # 保留（WebSocket 推理服务端）
│   ├── eval_client.py                    # 保留（LIBERO 仿真客户端）
│   └── demo_robocerebra.py               # ← _demo_robocerebra.py 重命名迁移
│
├── model_weights/                        # ← checkpoints/{CLIP,Qwen} 迁移重命名
│   ├── CLIP/
│   │   └── clip-vit-base-patch16/
│   ├── Qwen2.5-0.5B/
│   └── Qwen2.5-1.5B-Instruct/
│
├── data/                                 # ← dataset/ 重命名（与代码包名解冲突）
│   ├── RoboCerebra/
│   │   ├── RoboCerebra_trainset/
│   │   └── RoboCerebraBench/
│   └── libero/                           # ← dataset/datasets/ 重命名
│       └── libero_10/
│
├── data/outputs/                         # 训练产出检查点（对标 FlowPolicy）
│   └── <run_name>/
│       ├── checkpoints/
│       └── logs/
│
├── results/                              # 结果视频（保持不变）
├── logs/                                 # 日志（保持不变）
│
└── docs/                                 # 文档（扩充）
    ├── project_status.md
    └── evo1_analysis.md
```

---

## 三、详细变更清单

### 3.1 文件移动（不改变内容，仅重新组织位置）

| 操作 | 原路径 | 新路径 | 说明 |
|------|--------|--------|------|
| 移动 | `scripts/pretrain.py` | `pretrain.py` | 主入口到根目录 |
| 移动 | `scripts/train.py` | `train.py` | 主入口到根目录 |
| 移动 | `scripts/eval.py` | `eval.py` | 主入口到根目录 |
| 移动 | `configs/` | `dual_tree_vla/config/` | 配置随包走 |
| 移动 | `configs/ds_zero2.json` | `dual_tree_vla/config/deepspeed/ds_zero2.json` | 归入子目录 |
| 移动 | `configs/ds_zero3.json` | `dual_tree_vla/config/deepspeed/ds_zero3.json` | 归入子目录 |
| 移动+重命名 | `dual_tree_vla/model/dual_tree_vla.py` | `dual_tree_vla/policy/dual_tree_policy.py` | 策略层归位 |
| 移动 | `dual_tree_vla/model/attn.py` | `dual_tree_vla/model/common/attn.py` | 通用组件归位 |
| 移动 | `dual_tree_vla/model/fusion.py` | `dual_tree_vla/model/common/fusion.py` | 通用组件归位 |
| 移动 | `dual_tree_vla/model/semantic_jump_head.py` | `dual_tree_vla/model/common/semantic_jump_head.py` | 通用组件归位 |
| 移动+重命名 | `checkpoints/CLIP/` | `model_weights/CLIP/` | 预训练权重单独存放 |
| 移动+重命名 | `checkpoints/Qwen2.5-0.5B/` | `model_weights/Qwen2.5-0.5B/` | 预训练权重单独存放 |
| 移动+重命名 | `checkpoints/Qwen2.5-1.5B-Instruct/` | `model_weights/Qwen2.5-1.5B-Instruct/` | 预训练权重单独存放 |
| 移动+重命名 | `dataset/RoboCerebra/` | `data/RoboCerebra/` | 解决与代码包命名冲突 |
| 移动+重命名 | `dataset/datasets/` | `data/libero/` | 统一数据目录命名 |
| 移动+重命名 | `_demo_robocerebra.py` | `scripts/demo_robocerebra.py` | 演示脚本归入 scripts |

### 3.2 新增文件（需要创建）

| 文件 | 内容说明 |
|------|---------|
| `setup.py` | 包安装配置（仿照 FlowPolicy/setup.py） |
| `dual_tree_vla/policy/__init__.py` | 导出 `DualTreePolicy` |
| `dual_tree_vla/policy/base_policy.py` | 抽象策略基类，定义 `predict_action()`, `reset()`, `set_normalizer()` 接口 |
| `dual_tree_vla/model/common/__init__.py` | 导出 `FlashMHA`, `CrossModalFusion`, `JumpAwareHead` |
| `dual_tree_vla/dataset/base_dataset.py` | 抽象数据集基类，定义 `__len__()`, `__getitem__()`, `get_normalizer()` 接口 |
| `dual_tree_vla/losses/__init__.py` | 导出损失函数类 |
| `dual_tree_vla/losses/flow_loss.py` | 封装 FlowMatching 训练/推理逻辑 |
| `dual_tree_vla/losses/semantic_loss.py` | 封装 L_boundary（BCE）和 L_sem（InfoNCE）损失 |
| `dual_tree_vla/common/__init__.py` | 导出通用工具 |
| `dual_tree_vla/common/normalizer.py` | 数据归一化工具（参考 FlowPolicy 的 LinearNormalizer） |
| `dual_tree_vla/common/checkpoint_util.py` | 检查点保存/恢复，支持 best-k 管理 |
| `dual_tree_vla/common/pytorch_util.py` | 设备管理、随机种子、混合精度等工具 |

### 3.3 修改现有文件（更新路径引用）

| 文件 | 需要修改的内容 |
|------|--------------|
| `dual_tree_vla/__init__.py` | 更新导入路径（从 `model` 改为 `policy`） |
| `dual_tree_vla/model/__init__.py` | 更新导入路径（移除已迁移到 policy/ 的类） |
| `pretrain.py`（迁移后）| 更新 config 路径引用 → `dual_tree_vla/config/` |
| `train.py`（迁移后）| 更新 config 路径引用 → `dual_tree_vla/config/` |
| `eval.py`（迁移后）| 更新 config 路径引用 → `dual_tree_vla/config/` |
| `dual_tree_vla/config/pretrain.yaml` | `data.root` → `data/RoboCerebra/...`；`clip_model_name` → `model_weights/CLIP/...`；`llm_path` → `model_weights/Qwen2.5-0.5B`；`ckpt_dir` → `data/outputs/pretrain` |
| `dual_tree_vla/config/train_phase1.yaml` | 同上，更新所有路径引用 |
| `dual_tree_vla/config/train_phase2.yaml` | 同上，更新所有路径引用 |
| `dual_tree_vla/config/default.yaml` | 同上，更新所有路径引用 |
| `scripts/pretrain.sh` | 更新入口指向 `python pretrain.py`；更新 config 路径 |
| `scripts/train_phase1.sh` | 更新入口指向 `python train.py`；更新 config 路径 |
| `scripts/train_phase2.sh` | 更新入口指向 `python train.py`；更新 config 路径 |
| `scripts/eval_server.py` | 更新 `ckpt_dir` 和 `config_path` 路径 |
| `scripts/demo_robocerebra.py` | 更新数据路径 → `data/RoboCerebra/...` |
| `dual_tree_vla/dataset/libero.py` | 无需修改（由配置控制路径） |
| `dual_tree_vla/dataset/robocerebra.py` | 无需修改（由配置控制路径） |

---

## 四、分阶段执行计划

### Phase 1：结构重组（纯文件移动，不改代码逻辑）
**预计影响**：低风险，只移动文件位置

1. 创建新目录：`dual_tree_vla/config/`, `dual_tree_vla/config/deepspeed/`, `dual_tree_vla/policy/`, `dual_tree_vla/model/common/`, `dual_tree_vla/common/`, `model_weights/`, `data/`
2. 移动配置文件：`configs/` → `dual_tree_vla/config/`
3. 移动主入口：`scripts/*.py`（train/eval/pretrain）→ 根目录
4. 移动演示脚本：`_demo_robocerebra.py` → `scripts/demo_robocerebra.py`
5. 移动数据目录：`dataset/` → `data/`（包括重命名 `datasets/` → `libero/`）
6. 移动预训练权重：`checkpoints/{CLIP,Qwen*}` → `model_weights/`

### Phase 2：代码包内部重组
**预计影响**：中风险，需要更新导入路径

1. 创建 `dual_tree_vla/policy/` 目录
2. 移动 `model/dual_tree_vla.py` → `policy/dual_tree_policy.py`
3. 移动 `model/attn.py`, `model/fusion.py`, `model/semantic_jump_head.py` → `model/common/`
4. 更新所有 `__init__.py` 导入路径
5. 更新 `pretrain.py`, `train.py`, `eval.py` 中的导入

### Phase 3：补充新增文件
**预计影响**：中风险，添加基类和工具

1. 创建 `setup.py`
2. 创建 `dual_tree_vla/policy/base_policy.py`（抽象基类）
3. 创建 `dual_tree_vla/dataset/base_dataset.py`（抽象基类）
4. 补充 `dual_tree_vla/losses/flow_loss.py` 和 `semantic_loss.py`
5. 创建 `dual_tree_vla/common/` 工具模块

### Phase 4：路径配置更新
**预计影响**：低风险，统一更新配置文件路径

1. 更新 `dual_tree_vla/config/*.yaml` 中的所有路径引用
2. 更新 Shell 脚本入口命令
3. 验证配置加载正确

---

## 五、重构前后对比

### 重构前（当前）
```
DualTreeVLA/
├── _demo_robocerebra.py        ← 散乱
├── configs/                    ← 游离在根目录
├── checkpoints/                ← 混杂 backbone 和训练产出
├── dataset/                    ← 与代码包名冲突
├── dual_tree_vla/
│   ├── model/
│   │   ├── attn.py             ← 通用组件平铺
│   │   ├── fusion.py           ← 通用组件平铺
│   │   ├── dual_tree_vla.py    ← 主模型混在 model/
│   │   ├── semantic_jump_head.py
│   │   ├── action_head/
│   │   ├── memory_tree/
│   │   └── sgmts/
│   ├── dataset/
│   └── losses/                 ← 空目录
└── scripts/
    ├── pretrain.py             ← 主入口混在 scripts
    ├── train.py                ← 主入口混在 scripts
    ├── eval.py                 ← 主入口混在 scripts
    ├── pretrain.sh
    ├── train_phase1.sh
    └── eval_server.py
```

### 重构后（目标）
```
DualTreeVLA/
├── pretrain.py                 ← 主入口在根目录
├── train.py                    ← 主入口在根目录
├── eval.py                     ← 主入口在根目录
├── setup.py                    ← 包配置
├── dual_tree_vla/
│   ├── config/                 ← 配置随包走
│   │   └── deepspeed/
│   ├── policy/                 ← 策略层独立
│   │   ├── base_policy.py      ← 抽象基类
│   │   └── dual_tree_policy.py ← 主策略
│   ├── model/
│   │   ├── common/             ← 通用组件独立
│   │   │   ├── attn.py
│   │   │   ├── fusion.py
│   │   │   └── semantic_jump_head.py
│   │   ├── action_head/
│   │   ├── memory_tree/
│   │   └── sgmts/
│   ├── dataset/
│   │   └── base_dataset.py     ← 新增基类
│   ├── losses/                 ← 有实际实现
│   │   ├── flow_loss.py
│   │   └── semantic_loss.py
│   └── common/                 ← 训练工具层
│       ├── normalizer.py
│       ├── checkpoint_util.py
│       └── pytorch_util.py
├── model_weights/              ← 预训练权重
├── data/                       ← 统一数据目录
│   ├── RoboCerebra/
│   └── libero/
├── data/outputs/               ← 训练产出
└── scripts/                    ← 仅 .sh + 服务工具
    ├── pretrain.sh
    ├── train_phase1.sh
    ├── eval_server.py
    └── demo_robocerebra.py
```

---

## 六、注意事项

1. **数据目录重命名风险**：`dataset/` → `data/` 和 `datasets/` → `libero/` 需要同步更新所有配置 YAML 中的 `data.root` 路径；若数据量大，建议用符号链接替代物理移动
2. **模型权重不动**：`model_weights/` 下的大文件（Qwen 权重等）若通过 Git LFS 管理，需要注意 LFS 跟踪规则
3. **训练检查点路径**：旧的 `checkpoints/runs/` 需映射到新的 `data/outputs/`，现有保存的检查点需手动迁移或更新配置
4. **向后兼容**：执行期间可以保留 `configs/` 符号链接（或临时保留旧目录）避免破坏现有 CI/CD 或远程训练脚本
5. **导入路径更新**：`model/dual_tree_vla.py` → `policy/dual_tree_policy.py` 后，所有 `from dual_tree_vla.model import DualTreeVLA` 需改为 `from dual_tree_vla.policy import DualTreePolicy`

---

## 七、最终目录与 FlowPolicy 的对应关系

| DualTreeVLA（重构后）| FlowPolicy（参考）| 说明 |
|---------------------|-----------------|------|
| `pretrain.py` / `train.py` / `eval.py` | `FlowPolicy/train.py` / `eval.py` | 主入口在包同级 |
| `dual_tree_vla/config/` | `flow_policy_3d/config/` | 配置在包内 |
| `dual_tree_vla/policy/` | `flow_policy_3d/policy/` | 策略抽象层 |
| `dual_tree_vla/model/common/` | `flow_policy_3d/model/common/` | 通用模型组件 |
| `dual_tree_vla/model/action_head/` | `flow_policy_3d/model/flow/` | 核心生成模型 |
| `dual_tree_vla/model/sgmts/` | `flow_policy_3d/model/vision/` | 视觉编码器 |
| `dual_tree_vla/dataset/` | `flow_policy_3d/dataset/` | 数据集（有基类） |
| `dual_tree_vla/losses/` | `flow_policy_3d/consistencyfm/` | 核心损失/算法 |
| `dual_tree_vla/common/` | `flow_policy_3d/common/` | 训练工具层 |
| `model_weights/` | `third_party/`（backbone）| 预训练骨干权重 |
| `data/` | `data/`（数据和输出）| 统一数据目录 |
| `scripts/*.sh` | `scripts/*.sh` | 仅 Shell 启动脚本 |

---

*计划文档生成日期: 2026-04-19*
*等待确认后执行*
