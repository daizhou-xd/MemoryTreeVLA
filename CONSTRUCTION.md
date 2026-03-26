
# MemoryTreeVLA（暂名）: 基于记忆树的视觉-语言-动作模型

## 1. 概述

MemoryTreeVLA 是一种面向长时程机器人操控的新型架构，通过显式的**层次化任务树（Hierarchical Task Tree）**记忆机制，实现复杂多步骤任务的规划、执行与回溯。该架构主体借鉴 Evo-1 的轻量化 VLA 设计，并结合 BT-TL-DMPs 的行为树任务分解思想、RoboCerebra 的长时程子任务标注方式，以及 GrootVL 的树状扫描状态空间模型（SSM），构建了一个具备长程推理能力的 VLA 系统。与上一版不同的是，**动作生成路径不再依赖 Action LLM**，而是直接将 Multimodal Mamba 输出的融合 token 序列作为动作头条件输入。

---

## 2. 核心架构

### 2.1 修正后的整体架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MemoryTreeVLA Architecture (Updated)                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Input Layer                                                               │
│   ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐          │
│   │   Visual Input   │  │    Tree.json     │  │   Robot State    │          │
│   │  (Camera Image)  │  │  (Subtask State) │  │  (Joint / EEF)   │          │
│   └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘          │
│            │                     │                     │                    │
│   Modality-Specific Encoders / Projectors                                   │
│            ▼                     ▼                     ▼                    │
│   ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐          │
│   │  Vision Mamba    │  │   Tree Mamba     │  │  State Projector │          │
│   │  (Tree Scan)     │  │    (Tree Scan)   │  │  (Linear / MLP)  │          │
│   │  Output: Z_v     │  │  Output: Z_t     │  │  Output: Z_s     │          │
│   └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘          │
│            │                     │                     │                    │
│   Multimodal Fusion                                                         │
│            └──────────────┬──────┴─────────────────────┘                    │
│                           ▼                                                  │
│              ┌──────────────────────┐                                        │
│              │   Multimodal Mamba   │                                        │
│              │  (Cross-Modal Fuse)  │                                        │
│              │ Input: [Z_v, Z_t, Z_s]│                                       │
│              │  Output: Z_fused     │                                        │
│              └──────────┬───────────┘                                        │
│                         │                                                   │
│   Action Generation                                                         │
│                         ▼                                                   │
│   ┌──────────────────────────────────────────────────┐                      │
│   │          Action Head (Diffusion / Flow / MLP)    │                      │
│   │  Condition: Z_fused Token Sequence               │                      │
│   │  Output: Action Sequence (End-effector Pose)     │                      │
│   └──────────────────────────────────────────────────┘                      │
│                                                                             │
│   ═══════════════════════════════════════════════════════════════════       │
│                                                                             │
│   Tree Management                                                           │
│   ┌──────────────────────────────────────────────────┐                      │
│   │              Tree LLM (Trainable)                │                      │
│   │  Input: Z_v (Visual Features) + Current Tree     │                      │
│   │  Output: Updated Tree.json                       │                      │
│   │  Function: Subtask completion detection          │                      │
│   └──────────────────────────────────────────────────┘                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Tree LLM + 多模态融合设计

> **架构说明**：视觉编码完全由 Vision Mamba 负责（GrootVL 树状扫描），**不使用 ViT 等视觉骨干**。系统中仅保留 **Tree LLM** 负责任务树初始化与更新；动作分支不再经过 LLM，而是由 **Multimodal Mamba 的融合 token 序列直接条件化动作头**。

| 组件 | 功能定位 | 输入 | 输出 | 架构细节 |
|------|----------|------|------|----------|
| **Vision Mamba** | 视觉时序编码（替代 ViT） | 图像序列 $I_t$ | 视觉特征 $Z_v \in \mathbb{R}^{T \times D}$ | GrootVL 树状扫描 SSM，无需预训练视觉骨干 |
| **Tree Mamba** | 结构化任务编码 | Tree.json | 树特征 $Z_t \in \mathbb{R}^{N \times D}$ | 将层次化树结构线性化为序列，保留父子关系 |
| **State Projector** | 机器人状态编码 | 关节角度 / 末端执行器位姿向量 $s_t$ | 状态 token $Z_s \in \mathbb{R}^{1 \times D}$（或少量 tokens） | 轻量 Linear / MLP 投影，将低维状态向量映射到与 $Z_v, Z_t$ 相同的 token 维度 |
| **Multimodal Mamba** | **跨模态融合** | $[Z_v; Z_t; Z_s]$ | 融合特征 $Z_{fused} \in \mathbb{R}^{L \times D}$ | **核心创新**：通过选择性 SSM 实现视觉-任务-状态动态对齐；$Z_s$ 充当即时执行状态锚点 |
| **Action Head** | 动作生成 | $Z_{fused}$ token 序列 | 动作序列 $a_{1:H}$ | 扩散 / Flow Matching / MLP 头；直接把融合 tokens 作为条件上下文 |
| **Tree LLM** | 任务管理 | $Z_v$（投影）+ Tree 文本描述 | 更新后的 Tree.json | **Qwen2.5-1.5B-Instruct**；低频调用，Instruct 版保证 JSON 格式稳定，推理能力更强 |

**关键数据流**：

| 路径 | 流向 |
|------|------|
| 视觉模态 | `Image` →(Vision Mamba)→ `Z_v` →(Concat)→ `Multimodal Mamba` → `Z_fused` |
| 树模态 | `Tree.json` →(Tree Mamba)→ `Z_t` →(Concat)→ `Multimodal Mamba` → `Z_fused` |
| 状态模态 | `RobotState` $s_t$ →(State Projector)→ `Z_s` →(Concat)→ `Multimodal Mamba` → `Z_fused` |
| 动作条件路径 | `Z_fused` → `Action Head`（作为条件 token 序列）→ `Action Sequence` |
| Tree LLM 输入 | `Z_v` →(Linear Projector)→ `v_tokens` + Tree 文本序列化 → Tree LLM |

---

## 3. 关键技术：Multimodal Mamba 设计

### 3.1 架构动机
传统的多模态融合（如简单的 Concat + Linear 或 Cross-Attention）难以处理视觉时序与任务树结构之间的**动态跨模态依赖**。Multimodal Mamba 利用状态空间模型（SSM）的选择性机制，实现：
1. **时序对齐**：视觉动态与子任务进度的时序同步
2. **选择性关注**：根据当前子任务类型，动态关注视觉特征的相关区域
3. **长程依赖**：跨越多子任务的长时程记忆保持

### 3.2 融合策略细节

**跨树选择性机制**：以任务树特征的全局均值作为条件信号，对视觉特征进行残差门控，动态调节视觉与树信息的融合比例；再将门控后的视觉 tokens 与树 tokens 拼接为统一序列，经多层 Mamba SSM 处理后输出 `Z_fused`。

**机器人状态融合**：将当前时间步的关节角度、末端执行器位姿（6-DoF）等低维状态向量 $s_t \in \mathbb{R}^{d_s}$ 通过 State Projector（线性层或 2 层 MLP）映射为 $Z_s \in \mathbb{R}^{1 \times D}$（或少量 tokens），与 $Z_v$、$Z_t$ 一同拼接后送入 Multimodal Mamba。$Z_s$ 充当即时执行状态锚点，使融合上下文同时感知视觉场景、子任务进度和机器人当前构型，有效减少 Action Head 对系统状态估计的歧义。

**统一树状扫描**：视觉树与任务树作为同一超图的子树，构建基于语义相似度的跨树连接矩阵，采用交替扫描（intra-tree → inter-tree → intra-tree）的方式统一处理两棵树的结构信息。

### 3.3 基于 GrootVL 代码的扫描实现思路

这一部分基于仓库中的 `MTVLA/models/tree_scan` 实现整理，而不是只基于论文文字描述。需要强调的是：**GrootVL 原始代码核心对应的是图像特征上的树状扫描；task tree 扫描是本仓库将同一套 Tree-SSM 递推迁移到显式任务树拓扑后的实现。**

#### 图像扫描的大致思路（Vision Mamba / GrootVL 风格）

1. **先把图像变成局部 patch 特征图**：`StemLayer` 用两层 stride-2 卷积把输入图像降到 $1/4$ 分辨率，得到 $(B, H, W, C)$ 特征图。
2. **按 4-邻域建立网格图**：把每个 patch 当成一个节点，只连接上下左右相邻 patch，形成规则网格边集。
3. **根据当前层特征自适应计算边权**：`MinimumSpanningTree` 对相邻 patch 计算余弦相似度（或 L2 距离），再映射成 MST 使用的边权。这一步依赖当前层特征，所以**每一层的树结构都可能不同**。
4. **构建最小生成树（MST）**：对每个样本分别运行 Kruskal，保留最能反映局部语义连通性的 $L-1$ 条边，把二维网格压缩成一棵树。
5. **将树线性化为 BFS 顺序**：从根节点开始做 BFS，得到 `sorted_index` 和 `sorted_parent`。前者表示“第 $t$ 个访问到的节点是谁”，后者表示“该节点的父节点在 BFS 序列中的位置”。
6. **沿 BFS 顺序做 Tree-SSM 递推**：`tree_scanning()` 先从输入特征中投影出 $\Delta, B, C$ 等 SSM 参数，再对每个节点执行

$$
h_i = dA_i \cdot h_{parent(i)} + dB_i \cdot x_i
$$

因为按 BFS 扫描，所以父节点一定已经被处理过，递推可以顺序完成。

7. **读出并回到残差块**：得到隐藏状态后，再按

$$
y_i = h_i \cdot C_i + D \cdot x_i
$$

生成输出，经过归一化、门控和 MLP 残差块，形成下一层视觉特征。

**直观理解**：GrootVL 不是把图像硬编码成行优先或列优先序列，而是让当前特征自己决定“哪些 patch 更应该先通过树连接起来”，再沿这棵输入自适应的树传播状态。

#### Task Tree 扫描的大致思路（本仓库对 GrootVL 递推的迁移）

1. **输入不是图像网格，而是显式任务树**：`Tree.json` 中已经给出了父子层级，因此不需要像图像那样再从相似度构建 MST。
2. **把任务树转成 parent map**：每个节点记录 `node_id -> parent_id`，根节点的父节点为空。
3. **预先计算并缓存 BFS 拓扑**：`build_bfs_from_adj()` 从根节点出发生成 `sorted_index` 和 `sorted_parent`。与视觉分支不同，这个 BFS 顺序只要任务树不变，就可以跨时间步复用。
4. **节点特征进入 TreeMambaLayer**：节点特征可以由子任务文本、状态嵌入、历史 token 记忆等组成，输入形状为 $(B, N, D)$。
5. **复用同一套标量 Tree-SSM 递推**：`TreeMambaLayer` 与视觉分支一样，先为每个节点生成 $\Delta, B, C$，再按 BFS 顺序执行父到子的递推：父节点隐藏状态直接影响子节点更新。
6. **叠加多层残差块形成上下文化树表示**：多层 TreeMamba 后，每个节点的表示都带有来自祖先链、兄弟执行上下文和树深度结构的信息，最终输出为 $Z_t$。

**直观理解**：图像扫描是“先从特征里长出一棵树，再扫描”；task tree 扫描是“树已经存在，直接沿显式拓扑扫描”。两者共享的核心不是建树方式，而是**沿树拓扑做选择性 SSM 状态传播**这一递推原语。

---

## 4. 训练策略（三阶段修正版）

基于 RoboCerebraBench  的高质量子任务标注数据：

### 第一阶段：Tree LLM + Multimodal Mamba 基础训练
- **目标**：建立视觉-树-动作的基础关联
- **训练对象**：Tree LLM 独立训练；Multimodal Mamba 使用对比学习预训练（对齐视觉与树特征空间）
- **数据**：RoboCerebraBench 子任务标注（平均 9.1 步/任务）

### 第二阶段：动作头与融合模块联合训练
- **目标**：优化跨模态融合到动作的映射
- **设置**：
  - 冻结：Vision Mamba, Tree Mamba, Tree LLM
  - 训练：**State Projector** + **Multimodal Mamba** + Action Head
- **损失函数**：动作重建损失 + 融合特征对齐损失
- **说明**：该阶段不再需要 Action LLM 蒸馏或 token 自回归目标，而是直接学习从 $Z_{fused}$（含 $Z_s$）token 序列到连续动作的条件映射；State Projector 从随机初始化开始随 Multimodal Mamba 一起训练

### 第三阶段：端到端全量微调
- **目标**：整体架构适应端到端任务执行
- **设置**：
  - 解冻：Multimodal Mamba（部分层）, State Projector, Action Head
  - 可选解冻：Tree LLM（若需要提升子任务完成检测与树更新鲁棒性）
  - 冻结：Vision Mamba（保留通用视觉特征）
  - 学习率分层：Tree LLM (1e-5，可选) < Multimodal Mamba (5e-5) < State Projector / Action Head (1e-4)

---

## 5. 推理流程（修正版）

### 5.1 标准执行流程（含多模态融合）

```
1. 初始化阶段
   └─> Tree LLM 根据初始视觉图像初始化 Tree.json
       └─> 生成完整子任务序列与初始状态

2. 执行循环（每个时间步 t）
   ├─> 视觉编码：Image_t → Vision Mamba → Z_v (Visual Tokens)
   ├─> 树编码：Tree.json → Tree Mamba → Z_t (Tree Tokens)
   ├─> 状态编码：RobotState s_t → State Projector → Z_s (State Token)
   ├─> 关键步骤：Multimodal Mamba 融合
   │   └─> Input: Concat[Z_v, Z_t, Z_s]
   │   └─> Process: 跨模态选择性 SSM 处理（视觉 + 任务树 + 机器人状态）
   │   └─> Output: Z_fused (Fused Representation)
   ├─> 动作头条件化：Z_fused 直接作为条件 token 序列输入 Action Head
   ├─> 动作头输出：Action Sequence (Δt 时域动作)
   ├─> 执行动作并观察状态变化
   └─> 子任务完成检测？
     ├─> 是：将当前融合 token 摘要或动作条件状态存入 Tree.json 对应子任务节点
     │       Tree LLM 更新子任务状态，激活下一子任务
       └─> 否：继续当前子任务执行

3. 任务完成：所有子任务状态为 completed，输出 success
```

### 5.2 回溯机制（利用融合上下文）

```
回溯触发条件：
暂未定义明确触发条件，计划在后续版本中基于失败子任务的特征（如连续失败次数、视觉-树特征不匹配度）设计动态触发机制

回溯流程：
1. 读取 Tree.json 中的 backtrack_pointer
2. **关键**：从 Tree.json 加载上一成功子任务缓存的融合 token 摘要 / 条件状态
3. 重新构建 Z_t（上一子任务状态）并与当前视觉 Z_v 送入 Multimodal Mamba
4. 生成融合特征 Z_fused^backtrack，包含"回溯上下文"
5. Action Head 基于 Z_fused^backtrack 直接生成修正动作
6. Tree LLM 重置当前子任务状态，重置 failure_count
```

**回溯中的多模态优势**：Multimodal Mamba 在回溯时能动态调节视觉与历史任务记忆的权重，例如当重新执行 "reach" 时，更关注当前视觉中的物体位置而非历史树状态。

---

## 6. 技术优势分析

### 6.1 Multimodal Mamba 的核心价值

1. **动态跨模态注意力**：相比 Transformer 的静态 Cross-Attention，SSM 的选择性机制允许根据当前子任务动态调整视觉与树信息的融合比例

2. **计算效率**：Mamba 的线性复杂度 $O(L)$ 使长时程任务（RoboCerebra 中长达 20 步）的融合计算可行，而 Transformer 的 $O(L^2)$ 在 $L = T + N$ 较大时开销过高

3. **时序-结构统一建模**：视觉时序（连续）与任务树（离散层次）在 SSM 的统一状态转移框架下融合，避免了模态间语义鸿沟

### 6.2 与现有方案对比

| 方案 | 长时程记忆 | 任务回溯 | 多模态融合策略 | 计算复杂度 | VLA集成 |
|------|-----------|---------|--------------|-----------|--------|
| **MemoryTreeVLA（本文）** | ✅ 显式树状记忆 | ✅ 指针式回溯 | Tree-guided SSM 门控融合 + 直接条件化动作头 | $O(L)$ | ✅ 轻量化 Mamba-VLA 主干 |
| RoboCerebra (System 2) | ✅ 内存库(Memory Bank) | ❌ 无结构化回溯 | VLM轮询（离散切换） | $O(L^2)$ | ✅ 外部VLM + VLA控制器 |
| BT-TL-DMPs | ✅ 行为树状态机 | ✅ BT反应性恢复 | 符号推理（无神经融合） | 符号求解 | ❌ 基于DMP，非端到端 |
| GrootVL | ❌ 无任务记忆 | ❌ | 树状SSM（单模态） | $O(L)$ | ❌ 视觉/文本骨干，非VLA |
| Evo-1 | ❌ 无长时程机制 | ❌ | 集成模块 + 扩散Action Head | $O(L^2)$ | ✅ 轻量化0.77B VLA |
| OpenVLA / $\pi_0$ | ❌ | ❌ | Token拼接 | $O(L^2)$ | ✅ 大模型VLA |

**核心差异化优势**：
1. **结构化记忆 vs 隐式记忆**：RoboCerebra 的 Memory Bank 是扁平化 key-value 存储，MemoryTreeVLA 的 Tree.json 保留了子任务的**层次依赖关系**，可以做有约束的回溯（仅回到父节点，而非任意跳转）。
2. **神经-符号融合**：BT-TL-DMPs 在符号层（STL→BT）做任务结构，而 MemoryTreeVLA 将树结构**嵌入神经特征空间**，通过 Tree Mamba 和 Multimodal Mamba 实现端到端可微分优化。
3. **轻量化设计**：动作分支移除了高频调用的 Action LLM，在线控制路径变为 Vision/Tree Mamba → Multimodal Mamba → Action Head，相比显式自回归动作生成更适合实时闭环控制。

---

## 7. 实验与评估建议

### 7.1 基准测试选择

对标 RoboCerebra 的评估协议，建议在以下基准上验证 MemoryTreeVLA：

| 基准 | 特性 | 适配理由 |
|------|------|----------|
| **RoboCereBraBench** | 长时程（平均 9.1 步/任务），含 Memory-Exploration / Memory-Execution / Random-Disturbance 模式 | 直接验证树状记忆的长时程规划能力与回溯机制 |
| **LIBERO**（Spatial / Object / Goal / Long） | 四类任务套组，涵盖空间关系、物体操作、目标导向与长时程序列 | 验证跨任务类型的子任务切换准确性与动作头泛化能力 |
| **RoboMME** | 多模态机器人操控评估基准，含丰富的场景与语义多样性 | 验证 Tree LLM 对多样化任务描述的树结构初始化与执行鲁棒性 |

### 7.2 消融实验设计

建议围绕三个核心设计进行消融：

```
A. 记忆树结构消融
   - MemoryTreeVLA-full：完整树记忆 + 回溯
   - MemoryTreeVLA-flat：将树退化为扁平列表（等价于 RoboCerebra Memory Bank）
   - MemoryTreeVLA-no-memory：无历史记忆（纯 reactive VLA）
   → 验证树结构相对扁平记忆的优势

B. 多模态融合模块消融
   - w/ Multimodal Mamba（本文）
   - w/ Cross-Attention 替换 Multimodal Mamba
   - w/ Concat + Linear 简单融合
   → 验证树状 SSM 融合的效果与计算效率权衡

C. 回溯机制消融
   - w/ backtrack（指针式回溯）
   - w/ restart（从头重新执行，BT-TL-DMPs 风格）
   - w/o backtrack（不触发回溯）
   → 验证指针式回溯相对全局重置的效率优势
```

### 7.3 评估指标

- **成功率（SR）**：任务完整完成（所有叶子节点 `completed`）
- **子任务精度（Sub-SR）**：每个子任务的独立完成率，衡量树状态的准确性
- **回溯利用率**：成功案例中触发回溯后最终成功的比例
- **推理延迟**：每步动作生成时间（目标参考 Evo-1 实测 ≥ 15 Hz）
- **树状态准确率（Tree-Acc）**：Tree LLM 判断子任务完成状态的 F1 分数（对比人工标注）

### 7.4 与 RoboCerebra 基线的对比设置

RoboCerebra 提出的 System 1–System 2 交互框架以 VLM 轮询方式检测子目标完成，属于**离散式监控**。MemoryTreeVLA 的 Tree LLM 以**每步视觉观测**为输入做连续检测，可以在 RoboCerebra 的三种任务模式（Memory Exploration、Memory Execution、Random Disturbance）下分别对比：
- **Memory-Exploration**：测试树结构对未见区域的预测性覆盖能力
- **Memory-Execution**：核心测试场景，验证树导引下的精准执行
- **Random-Disturbance**：验证回溯机制在随机干扰下的鲁棒恢复能力

---

## 8. Tree.json 数据结构规范

任务树以 JSON 格式存储，每个节点代表一个子任务，包含执行状态、视觉证据以及用于回溯的历史条件表征：

```json
{
  "task_id": "prepare_breakfast_001",
  "task_description": "Prepare breakfast with toast and coffee",
  "root": {
    "id": "root",
    "type": "sequence",
    "status": "in_progress",
    "children": [
      {
        "id": "subtask_0",
        "description": "pick up bread slice",
        "type": "primitive",
        "status": "completed",
        "token_sequence": "<stored_fused_tokens_or_summary>",
        "completion_evidence": {
          "frame_idx": 142,
          "confidence": 0.94
        },
        "failure_count": 0,
        "backtrack_pointer": null
      },
      {
        "id": "subtask_1",
        "description": "place bread in toaster",
        "type": "primitive",
        "status": "in_progress",
        "token_sequence": null,
        "completion_evidence": null,
        "failure_count": 1,
        "backtrack_pointer": "subtask_0"
      },
      {
        "id": "subtask_2",
        "description": "make coffee",
        "type": "sequence",
        "status": "pending",
        "children": [
          {"id": "subtask_2_0", "description": "pour water", "status": "pending"},
          {"id": "subtask_2_1", "description": "press brew button", "status": "pending"}
        ]
      }
    ]
  }
}
```

**状态迁移规则**（对应 BT-TL-DMPs 中的行为树节点状态）：
- `pending` → `in_progress`：父节点激活，前序兄弟节点全部 `completed`
- `in_progress` → `completed`：Tree LLM 检测到完成条件（置信度 > 阈值）
- `in_progress` → `failed`：`failure_count` 超过最大重试次数（默认 3 次）
- `failed` → `in_progress`：回溯指针激活，重置并重新执行

---

## 9. 总结

MemoryTreeVLA 通过以下四个核心贡献构建了一个面向长时程机器人操控的完整框架：

1. **层次化任务树记忆（Tree.json）**：借鉴 BT-TL-DMPs 的行为树模块化思想，将复杂任务分解为层次化子任务节点，每个节点存储执行状态、视觉证据及历史融合条件表征，支持精确的指针式回溯，解决了现有 VLA 模型无法有效处理长时程任务的问题。

2. **Multimodal Mamba 跨模态融合**：基于 GrootVL 的树状扫描 SSM，将视觉时序特征 $Z_v$、任务树特征 $Z_t$ 与机器人关节状态 token $Z_s$ 三路拼接后进行动态门控融合，以 $O(L)$ 线性复杂度实现比 Cross-Attention 更高效的跨模态依赖建模；$Z_s$ 的引入让融合上下文同时感知场景、任务进度和机器人构型，更贴近闭环控制需求。

3. **单 Tree LLM + 直接动作头分工（Mamba 视觉编码）**：视觉特征完全由 Vision Mamba 提取，Tree LLM 采用 **Qwen2.5-1.5B-Instruct** 负责低频任务树初始化与更新；高频控制路径中不再引入 Action LLM，而是由 Multimodal Mamba 输出 token 序列直接条件化 Action Head，从而更贴近实时控制需求。

4. **三阶段渐进训练**：受 Evo-1 两阶段训练范式启发，扩展为三阶段训练（Tree LLM / 树结构建模预训练 → 融合模块与动作头联合训练 → 端到端微调），在保留树管理能力与视觉语义表示的同时，逐步建立视觉-树-动作的联合分布。

**未来方向**：
- 回溯触发条件的自动化设计（基于视觉-树特征分布偏移检测）
- 引入 STL（Signal Temporal Logic，来自 BT-TL-DMPs）作为树节点的形式化约束，提升任务规范的精确性
- 在 RoboCerebra 1,000 条长时程轨迹上进行完整实验验证
- 探索在线树结构更新（任务执行过程中 Tree LLM 动态新增/删除子任务节点）
