# DualTreeVLA 论文初稿

副标题：A Dual-Tree Plug-and-Play Augmentation Framework for Long-Horizon Vision-Language-Action Learning

文档用途：

- 作为论文初稿骨架，直接用于继续扩写
- 作为方法章节与实验章节的统一事实底稿
- 作为配图生成提示词的集中整理文档

说明：

- 本稿以当前仓库实现为主要事实依据
- 文中涉及定量结果的地方均以“待填”标记，后续用真实实验结果替换
- 若最终投稿版本需要完全与当前实现逐项一致，应优先参考 CONSTRUCTION.md 中对“当前主实现路径”和“遗留设计路径”的区分

---

## 摘要

长时程机器人操作要求模型同时具备局部视觉辨识能力、跨阶段记忆组织能力，以及稳定的动作生成能力。现有视觉-语言-动作模型虽然在短视野操作和指令条件控制方面取得了进展，但在长时程、多子任务切换、环境扰动和历史依赖较强的场景中，仍容易出现语义遗忘、阶段边界混淆和动作条件化不足等问题。针对这一问题，我们提出 DualTreeVLA，一种面向现有 VLA 骨架的即插即用双树增强框架。该方法不重写骨架的预训练视觉主干和语言主干，而是通过最小接口将两类结构性归纳偏置注入现有模型：其一是作用于单帧 patch 图结构的视觉树模块 Semantic-Guided Mamba Tree Scan，用于对视觉 token 进行语义引导的树式扫描与增强；其二是作用于跨帧历史的层级记忆树模块 Hierarchical Memory Tree，用于在长时程执行中维护可增长、可分支、可抽象的时间层级记忆。两个模块分别建模空间结构和时间结构，并通过门控视觉融合与记忆 token 注入的方式接回 VLA 骨架，从而在不破坏原始预训练权重加载方式的前提下增强长时程操作能力。

在训练上，我们采用三阶段流程。第一阶段在 RoboCerebra 数据上预训练双树模块，学习动作边界检测、语义对齐和抽象提升；第二阶段在 LIBERO 数据上冻结骨架主干并进行流匹配热身；第三阶段进一步以低学习率解冻语言模型，实现端到端适配。实验设计覆盖离线轨迹误差、子任务边界质量、树结构统计和仿真成功率等多个维度。我们的方法强调一种面向工程现实的研究路径：在保留现有大模型骨架稳定性的同时，通过结构化外接模块显式增强长时程任务中的空间推理与记忆建模能力。初步结果表明，DualTreeVLA 在长时程操作中的条件化稳定性、结构可解释性和扩展性方面具有明显优势，并为后续将树结构显式纳入 VLA 系统提供了一个可复现的实现框架。

关键词：视觉语言动作模型，长时程操作，结构化记忆，树扫描，流匹配，机器人学习

---

## 1. 方法

### 1.1 问题定义

我们考虑长时程视觉语言动作学习问题。给定当前时刻观测图像 $I_t$、语言指令 $L$、本体感知状态 $q_t$，模型需要预测未来长度为 $H_a$ 的连续动作块：

$$
\hat{A}_{t:t+H_a-1} = f(I_t, L, q_t, \mathcal{H}_{<t})
$$

其中 $\mathcal{H}_{<t}$ 表示历史执行信息。难点在于：

- 图像内部的任务相关区域并非均匀重要
- 历史信息不是平坦序列，而更适合组织为可分支、可抽象的层级结构
- 现有 VLA 骨架通常高度耦合，直接插入新模块会破坏预训练权重加载与稳定性

为此，我们将目标转化为：在不修改骨架内部预训练结构的前提下，分别从空间和时间两个维度对骨架进行结构增强。

### 1.2 方法总览

DualTreeVLA 包含两个核心模块：

1. 视觉树模块 SGMTS，作用于当前帧的视觉 patch 图结构
2. 记忆树模块 HMT，作用于跨时刻历史片段的层级记忆组织

其核心思想不是替换原始视觉编码器或语言模型，而是以旁路增强方式接入已有 VLA 骨架。对当前仓库的主实现而言，这种接入通过三步完成：

1. 在 ViT 最后一层注册只读 forward hook，提取 patch 特征 $P_t$
2. 用 SGMTS 和 GateFusion 对 patch 特征进行结构增强，并回写到骨架视觉 token 流
3. 用 HMT 读出一个记忆 token，经线性投影后拼接到语言模型输入序列末尾

整体数据流可写为：

$$
I_t \rightarrow P_t \xrightarrow{\text{SGMTS}} Z_v \xrightarrow{\text{GateFusion}} V_t'
$$

$$
\left[E_t^{text}; E_t^{vis}(V_t'); e_{mem}\right] \xrightarrow{\text{LLM}} H_t
$$

$$
\hat{A}_{t:t+H_a-1} = \text{FlowHead}(H_t[:,0,:], q_t)
$$

其中 $e_{mem}$ 来自记忆树读出。

### 1.3 骨架适配与最小侵入原则

DualTreeVLA 的研究重点之一是保持对现有骨架的兼容性。为此，我们遵循如下原则：

- 不修改骨架的 ViT 参数结构
- 不修改骨架的语言模型结构
- 不重新设计骨架的多模态 prompt 机制
- 不阻断原始 `from_pretrained` 权重加载逻辑

这一设计使得 DualTreeVLA 不需要从头训练一个新的 VLA，而是作为现有 VLA 的结构增强层。对工程实现而言，这一选择有两个直接优势：

- 训练更稳定，因为骨架预训练能力可直接保留
- 可扩展性更高，因为同类适配思路可迁移到其他 VLA 骨架

### 1.4 视觉树：Semantic-Guided Mamba Tree Scan

#### 1.4.1 输入表示

对当前帧图像 $I_t$，骨架 ViT 输出 patch 表示：

$$
P_t = \{p_i\}_{i=1}^{N_p}, \quad p_i \in \mathbb{R}^{d_{patch}}
$$

同时，模型构造一个语义引导向量 $g_{sem}$。在当前实现中，该向量由任务文本嵌入与记忆树顶层抽象节点均值共同决定：

$$
g_{sem} = \beta g_{task} + (1-\beta)s_{top}
$$

当记忆树尚未形成可用抽象节点时，退化为：

$$
g_{sem} = g_{task}
$$

#### 1.4.2 语义重要性估计

我们将语义引导向量投影到 patch 空间，并与每个 patch 做余弦相似度计算：

$$
\sigma_i = \cos(p_i, W_g g_{sem})
$$

$\sigma_i$ 反映 patch $i$ 相对当前任务语义的重要程度。

#### 1.4.3 语义加权最大生成树

将 patch 网格视为一个局部邻接图，边权定义为：

$$
w_{ij} = \cos(p_i, p_j) + \alpha \sigma_i \sigma_j
$$

其中 $\alpha$ 控制语义偏置对图结构的影响。随后采用 Kruskal 算法在局部网格边集上构建最大生成树。根节点不再固定，而是选择语义响应最强的 patch：

$$
r^* = \arg\max_i \sigma_i
$$

这使得视觉树的扫描顺序由任务语义自适应决定。

#### 1.4.4 树式状态空间扫描

在得到 BFS 顺序后，对每个 patch 构造语义增强输入：

$$
X_i = p_i + \sigma_i W_{g'} g_{sem}
$$

然后沿树结构进行状态空间递推：

$$
h_i = \bar{A}_i \odot h_{par(i)} + \bar{B}_i \odot X_i
$$

$$
z_i = C_i^\top h_i + D \odot X_i
$$

得到增强后的视觉表示：

$$
Z_v = \{z_i\}_{i=1}^{N_p}
$$

从表示学习角度看，SGMTS 的作用不是替代视觉主干，而是在冻结视觉主干的前提下，为原始 patch 序列增加显式的语义树结构归纳偏置。

### 1.5 门控视觉融合

SGMTS 输出的增强特征 $Z_v$ 不能直接替换原始 ViT 特征，否则训练初期容易破坏骨架的预训练稳定性。为此，我们采用门控融合：

$$
\alpha = \sigma\left(W_{gate}[Z_v;V_t]\right)
$$

$$
V_t' = \alpha \odot Z_v + (1-\alpha) \odot V_t
$$

其中 $V_t$ 为原始 patch 特征。当前实现中将 $W_{gate}$ 初始化为零权重、负偏置，使得训练初期 $\alpha \approx 0$，从而保证：

$$
V_t' \approx V_t
$$

该设计的意义在于，双树模块最初近似“旁路静默”，随后再逐渐学会对关键语义区域增加影响。

### 1.6 记忆树：Hierarchical Memory Tree

#### 1.6.1 记忆树目标

长时程任务的历史不适合被简单视为平坦序列。许多动作片段存在明显的阶段边界、父子关系和语义抽象关系。HMT 通过在线生长的树结构显式组织这些历史信息。

树中包含两类节点：

- 叶子节点：表示局部执行片段，存储视觉表示、动作历史与权重
- 抽象节点：表示高层语义概念，存储语义提升后的抽象向量与权重

#### 1.6.2 边界驱动的树更新

我们使用 JumpAwareHead 对动作边界进行判定。当模型认为当前动作仍属于已有活动片段时，执行 merge；当模型认为出现子任务切换时，执行 branch，并触发后续 elevation。

#### 1.6.3 Merge

若当前片段属于已有活动叶子，则将当前信息并入活动节点，更新其：

- 视觉表示
- 动作历史
- 节点权重

这一机制适用于局部稳定执行阶段。

#### 1.6.4 Branch

当检测到动作跳变时，模型创建新叶子节点，用于表示一个新的任务阶段或子任务片段。Branch 的意义在于显式切分连续执行中的语义边界。

#### 1.6.5 Elevation

在分支之后，模型通过语义提升模块把若干低层节点组合成更高层的抽象语义：

$$
s_{parent} = \operatorname{MLPElevation}(\text{child semantics})
$$

这使得记忆树不是单纯的事件记录器，而是能逐步形成层级语义结构的记忆组织器。

### 1.7 记忆读出：TreeSSMReadout

记忆树中的抽象节点通过树式状态空间读出器转换为可供骨架使用的记忆向量。对于每个语义节点，首先构造输入：

$$
x_i = W_{abs}[s_i; \log(w_i)]
$$

随后沿 BFS 顺序递推，得到：

$$
Y \in \mathbb{R}^{N_{sem} \times d_{ssm}}
$$

在当前主实现中，记忆向量取最后一个 BFS 语义节点表示：

$$
m_{ctx} = Y[-1]
$$

再通过线性层投影到语言模型隐空间，并拼接到输入序列末尾：

$$
e_{mem} = W_{mem} m_{ctx}
$$

### 1.8 跳变感知头

JumpAwareHead 只基于动作历史和当前动作进行边界判定，不直接消费视觉和语言输入。这种设计强调一个假设：很多子任务切换首先体现在控制模式的变化上。

给定活动节点动作历史 $A_{act}$ 和当前动作 $a_{new}$，模型输出：

$$
p_{jump} = \sigma(\text{logit}(A_{act}, a_{new}))
$$

当 $p_{jump}$ 超过阈值时，触发 branch。该模块的主要价值在于：

- 将边界检测从复杂的多模态问题中抽离出来
- 为记忆树的结构更新提供稳定的离散触发信号

### 1.9 动作头：Flow Matching Action Head

动作头采用条件流匹配框架。给定条件上下文 $ctx$，模型学习从噪声动作轨迹到真实动作轨迹的速度场：

$$
v_\theta(a_t, t, ctx)
$$

训练时采样：

$$
u \sim \mathcal{N}(0,1), \quad t = \sigma(u)
$$

$$
a_0 \sim \mathcal{N}(0,I)
$$

$$
a_t = (1-t)a_0 + ta_{gt}
$$

目标速度为：

$$
v^* = a_{gt} - a_0
$$

优化目标为：

$$
\mathcal{L}_{flow} = \mathbb{E}\left[\|v_\theta(a_t, t, ctx) - v^*\|_2^2\right]
$$

推理时，从高斯噪声出发，通过 Euler 积分逐步得到动作 chunk。

### 1.10 当前实现与理想设计的差异说明

需要如实指出，当前仓库中存在一个更理想化的策略层设计版本，其中包含显式的 CrossModalFusion 模块，将视觉、记忆、语言和本体感知统一映射到融合向量中。该模块仍保留在仓库中，但当前主训练链路并未将其作为唯一主路径接入。因而本论文初稿在方法表述上应以当前 adapter 主实现为主，在讨论中再说明遗留设计与后续扩展方向。

---

## 2. 训练

### 2.1 总体训练流程

DualTreeVLA 采用三阶段训练策略：

1. Stage 0：在 RoboCerebra 上进行双树预训练
2. Phase 1：在 LIBERO 上进行流匹配热身
3. Phase 2：在 LIBERO 上进行端到端微调

该流程的基本动机是，把“结构学习”和“动作生成学习”解耦：先让双树模块学会边界、抽象和空间扫描，再让动作头学习如何利用这些增强信息。

### 2.2 Stage 0：双树预训练

预训练阶段使用 RoboCerebra 数据，目标不是直接最优动作预测，而是让双树模块获得三类能力：

- 动作边界检测能力
- 子任务语义对齐能力
- 层级抽象提升能力

当前实现中冻结以下模块：

- ViT
- LLM
- MLP projector
- FlowMatchingActionHead

可训练模块包括：

- SGMTS
- GateFusion
- sem_proj
- JumpAwareHead
- TreeSSMReadout
- MLPElevation
- mem_proj

预训练目标为：

$$
\mathcal{L}_{pretrain} = w_b\mathcal{L}_{boundary} + w_s\mathcal{L}_{sem} + w_e\mathcal{L}_{elev}
$$

其中：

#### 2.2.1 边界损失

边界损失用于训练 JumpAwareHead 识别子任务切换：

$$
\mathcal{L}_{boundary} = \operatorname{BCE}(\text{logits}, y_{boundary})
$$

标签可来自：

- 显式的子任务标注
- 或基于动作统计构造的自监督标签

#### 2.2.2 语义对齐损失

语义损失让抽象节点语义与子任务文本描述对齐，同时也约束视觉语义与任务描述一致：

$$
\mathcal{L}_{sem} = \operatorname{InfoNCE}(s_{proj}, g_{task})
$$

#### 2.2.3 提升一致性损失

提升损失约束抽象节点表示应与其子节点的聚合语义一致：

$$
\mathcal{L}_{elev} = \operatorname{MSE}(s_{parent}, \tilde{s}_{children})
$$

### 2.3 Phase 1：流匹配热身

Phase 1 的目标是：在尽量不扰动骨架主干的前提下，让模型学会利用双树增强信息进行动作条件化。

当前实现中，Phase 1 的主特点是：

- 使用 LIBERO step-level 数据
- 仅优化流匹配目标
- 通过批量化 VLM 前向提升训练速度

损失为：

$$
\mathcal{L}_{phase1} = \mathcal{L}_{flow}
$$

在当前代码实现里，Phase 1 真正开放梯度的模块不是旧文档中描述的 “CrossModalFusion + ActionHead”，而是 adapter 当前主路径上的一组模块，包括：

- SGMTS
- GateFusion
- mem_proj
- backbone.action_head

这一点在论文最终稿中必须与实现保持一致。

### 2.4 Phase 2：端到端微调

在 Phase 2 中，训练目标仍然只有流匹配损失，但优化范围扩大。当前策略是：

- 继续冻结 ViT 与视觉投影层
- 解冻双树模块
- 以较小学习率解冻 language model

具体地，语言模型使用单独参数组，并采用约 0.1 倍基础学习率。

这种设计的动机是：

- 保留视觉主干的稳定视觉表示
- 让语言主干在较低风险下适配结构增强后的输入分布

### 2.5 数据组织与采样方式

#### 2.5.1 RoboCerebra

RoboCerebra 使用轨迹级样本，适合预训练中逐帧更新记忆树。每个样本包含：

- 视频帧序列
- 动作序列
- 状态序列
- 高层任务描述
- 子任务描述与子任务索引

#### 2.5.2 LIBERO

LIBERO 在当前主训练中采用 step-level 样本形式：

- 每个样本对应当前时刻的一帧观测
- 标签是从该帧开始的未来 $H_a$ 步动作 chunk

这意味着训练是全量 step-level 覆盖，而不是只使用轨迹的前若干帧。需要特别说明，训练日志中的可视化函数可能默认只展示前 120 帧，但这不影响训练数据覆盖率。

### 2.6 训练工程实现细节

当前训练代码包含若干重要工程处理：

- 使用 bf16 混合精度
- 使用 AdamW 优化器
- 使用 warmup + cosine 衰减调度
- 在 DDP 下关闭 `find_unused_parameters=True`，减少额外图遍历开销
- 在 Phase 1/2 中引入 `_embed_batch_flow()`，将原本串行的多次骨架前向改为共享批量前向

这些细节对训练时间和可复现性均有明显影响，应写入附录或实验设置中。

---

## 3. 实验设置

### 3.1 数据集

#### 3.1.1 RoboCerebra

RoboCerebra 用于结构预训练。按照当前工作区的目录，训练集包含三个场景：

- coffee_table
- kitchen_table
- study_table

每个 case 包含：

- demo.hdf5
- mp4 视频
- task_description.json

这些标注为边界检测和语义对齐提供了可用监督。

#### 3.1.2 LIBERO

我们在 LIBERO 上进行动作学习和微调。当前工作区中包含：

- libero_10
- libero_spatial
- libero_object
- libero_goal

其中 `libero_10` 的元数据表明：

- 379 条 episode
- 101469 帧
- 10 个任务
- 10 FPS

LIBERO 数据在当前实现中采用 z-score 归一化。

#### 3.1.3 RoboCerebraBench

离线结构评估可在 RoboCerebraBench 上进行。当前工作区包含 6 个子集：

- Ideal
- Memory_Execution
- Memory_Exploration
- Mix
- Observation_Mismatching
- Random_Disturbance

它们适合评估边界质量、结构可解释性和扰动鲁棒性。

### 3.2 模型设置

当前主实现的关键配置包括：

- 骨架：InternVL3-1B 风格 VLM
- 动作维度：7
- 状态维度：LIBERO 下为 8
- 动作 horizon：16
- TreeSSM 维度：256
- SSM state 维度：16
- 记忆树最大深度：4

对于论文正文，可以概括写成：

“我们采用 InternVL3 风格的视觉语言骨架，并在其外部接入双树增强模块。视觉树和记忆树的内部维度统一设置为 256，以兼顾建模能力和训练成本。”

### 3.3 训练超参数

当前默认配置可概括为：

- 预训练：batch size 2，epoch 30，学习率 $3\times10^{-4}$
- Phase 1：batch size 8，梯度累积 2，学习率 $10^{-4}$
- Phase 2：batch size 8，较低学习率微调，language model 采用 0.1 倍学习率

若最终论文使用与此不同的集群设置，应以真实提交实验日志为准替换。

### 3.4 评估指标

建议将实验指标分为四类：

#### 3.4.1 动作质量

- action_l1
- action_l2

用于衡量离线轨迹预测误差。

#### 3.4.2 结构质量

- tree_nodes
- tree_depth
- tree_branches
- tree_elevations

用于量化记忆树的结构复杂度与抽象程度。

#### 3.4.3 边界质量

- subtask_boundary_f1
- subtask_sr

用于衡量模型生成的分支是否与真实子任务切换对齐。

#### 3.4.4 进度一致性

- prog_monotone_rate

用于衡量树结构的层级是否符合“越往下越接近目标”的进度规律。

### 3.5 消融实验建议

为了支撑论文论证，建议至少设置以下消融：

1. 移除 SGMTS，仅保留原始视觉 patch
2. 移除 GateFusion，直接使用原始视觉 token
3. 移除 HMT，仅使用无记忆版本
4. 移除 JumpAwareHead，采用固定边界或无分支
5. 只保留视觉树，不保留记忆树
6. 只保留记忆树，不保留视觉树

这些消融可以分别回答：

- 视觉树是否改善空间条件化
- 记忆树是否改善长时程依赖
- 边界检测是否是层级记忆成立的必要条件

### 3.6 结果表格模板

后续建议直接整理出如下表格：

#### 表 1：主结果

- 方法
- LIBERO success rate
- action_l1
- action_l2
- boundary_f1
- prog_monotone_rate

#### 表 2：消融结果

- Full model
- w/o visual tree
- w/o memory tree
- w/o jump head
- w/o gate fusion

#### 表 3：结构统计

- avg tree nodes
- avg tree depth
- avg branch count
- avg elevation count

### 3.7 复现实验注意事项

论文实验部分建议单独说明以下工程点：

- Flash Attention 在代码层已支持，但实际是否生效取决于环境安装
- DeepSpeed 配置文件存在，但是否启用取决于启动命令
- Phase 1/2 当前主实现使用了批量骨架前向优化
- LIBERO 仿真客户端当前图像翻转和 gripper 阈值与 Evo-1 默认设置存在差异

这些点属于“看似工程细节，实则影响最终成功率”的实现变量。

---

## 4. 配图建议与 AI 出图提示词

本节给出论文中最建议优先准备的图，以及适合直接喂给 AI 图片模型的提示词。建议统一使用：

- 学术论文图风格
- 白底或浅灰底
- 蓝色表示视觉树
- 绿色表示记忆树
- 橙色表示动作生成模块
- 黑色箭头表示主信息流
- 虚线表示辅助监督或训练信号

### 图 1：整体架构图

用途：主方法图，放在方法总览开头。

图中必须包含：

- 输入图像、语言指令、本体感知状态
- InternVL3 骨架
- ViT hook 提取 patch 特征
- SGMTS 视觉树
- GateFusion
- HMT 记忆树
- TreeSSMReadout 和 memory token
- LLM
- FlowMatchingActionHead
- 动作 chunk 输出

建议文案提示词：

"A clean academic architecture diagram for a robotics vision-language-action model named DualTreeVLA, white background, modern conference paper style, modular blocks with arrows, left-to-right pipeline, image input, language instruction input, proprioceptive state input, InternVL3 backbone, visual tree module SGMTS operating on patch tokens, gated fusion block, hierarchical memory tree HMT growing over time, memory token injected into language model sequence, flow matching action head producing action chunks, blue for visual tree, green for memory tree, orange for action head, black arrows, elegant typography, NeurIPS/CVPR style scientific figure"

### 图 2：SGMTS 视觉树细节图

用途：解释 patch 图如何被构造成语义加权树，并进行树式扫描。

图中必须包含：

- patch 网格
- 每个 patch 上的语义权重热度
- 最大生成树连边
- 动态语义根
- BFS 扫描顺序
- Tree-SSM 状态传播

建议文案提示词：

"A detailed scientific illustration of a semantic-guided visual tree over image patches, square patch grid, heatmap scores on patches, semantic root highlighted, maximum spanning tree edges drawn over the grid, breadth-first traversal order annotated, state-space propagation along the tree, blue and cyan technical color palette, white background, clear labels, conference paper visualization, high readability"

### 图 3：HMT 记忆树生长图

用途：展示 merge、branch、elevation 三种操作如何形成层级记忆。

图中必须包含：

- 时间轴上的多个动作片段
- 叶子节点逐步合并
- 新分支产生
- 抽象节点向上提升
- 最终形成多层树结构

建议文案提示词：

"A scientific diagram showing a hierarchical memory tree growing over time for robot manipulation, timeline on the bottom, leaf nodes representing short action segments, branch creation at subtask boundaries, merge operations for continuous segments, elevation operations creating abstract parent nodes, tree becoming multi-level over time, green hierarchical structure, minimal academic style, white background, clear step-by-step layout"

### 图 4：三阶段训练流程图

用途：概括预训练、热身、微调三阶段的冻结与解冻策略。

图中必须包含：

- Stage 0, Phase 1, Phase 2 三列
- 每阶段的数据集
- 冻结模块与可训练模块
- 对应损失函数
- checkpoint 传递关系

建议文案提示词：

"A professional training pipeline figure for a three-stage robot learning framework, three columns labeled Stage 0 Pretraining, Phase 1 Warm-up, Phase 2 Fine-tuning, arrows between stages, each stage shows dataset, frozen modules, trainable modules, losses, and checkpoint transfer, academic infographic style, white background, blue green orange color coding, clear boxes and arrows, suitable for a machine learning paper"

### 图 5：离线评估与仿真评估图

用途：展示训练后如何做离线指标和在线仿真成功率评估。

图中必须包含：

- 离线轨迹评估
- 树结构统计
- 边界 F1
- LIBERO 仿真 client-server 评估
- success rate 输出

建议文案提示词：

"A scientific evaluation pipeline diagram for a robotics learning paper, showing offline trajectory evaluation, tree structure metrics, boundary F1 analysis, and online LIBERO simulation with websocket client-server setup, success rate measurement, modern conference paper style, clean white background, compact and readable, with metric icons and arrows"

### 图 6：方法对比概念图

用途：强调 DualTreeVLA 与传统平坦 token 处理或无记忆方法的差异。

图中必须包含：

- 左边：flat token processing baseline
- 右边：DualTreeVLA with visual tree and memory tree
- 强调空间结构和时间结构的双重建模

建议文案提示词：

"A side-by-side comparison figure for a machine learning paper, left side shows a baseline vision-language-action model with flat token processing and no structured memory, right side shows DualTreeVLA with a visual tree over image patches and a hierarchical memory tree over time, highlighting spatial structure and temporal hierarchy, white background, elegant academic style, clear labels and arrows"

---

## 5. 当前版本需要后续补充的内容

在正式投稿前，建议把以下内容补齐：

1. 用真实实验结果替换摘要中的“初步结果表明”等描述
2. 增加 Related Work 部分，重点覆盖 VLA、长时程记忆建模、树结构建模、流匹配控制
3. 增加实验结果表格与图注
4. 增加失败案例分析
5. 增加局限性讨论，尤其是当前主实现与理想设计路径的差异

---

## 6. 可直接扩写的下一步建议

如果继续在这份文档上扩写，推荐顺序如下：

1. 在摘要后补一节“引言”
2. 在方法前补一节“相关工作”
3. 在实验设置后补“主结果”“消融实验”“可视化分析”“局限性”
4. 最后再统一润色标题、术语和图表编号

这样能最快从“方法型初稿”扩展到完整论文初稿。