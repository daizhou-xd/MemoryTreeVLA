# DualTreeVLA — 架构、实现与实验工作流全说明

> 本文档以当前工作区实现为准，覆盖源码、配置、文档、数据集目录和评测脚本的实际状态。
>
> 文档定位不是“概念草图”，而是面向论文撰写、复现实验、代码审阅和后续重构的技术基线。所有核心描述均以当前仓库中的真实代码路径为依据。

---

## 目录

1. [文档定位与扫描范围](#1-文档定位与扫描范围)
2. [项目定义与核心结论](#2-项目定义与核心结论)
3. [工作区全景与事实来源](#3-工作区全景与事实来源)
4. [当前主实现路径与遗留路径](#4-当前主实现路径与遗留路径)
5. [整体系统架构](#5-整体系统架构)
6. [与 Evo-1 骨架的接口设计](#6-与-evo-1-骨架的接口设计)
7. [主干网络：InternVL3Backbone 与 InternVL3Embedder](#7-主干网络internvl3backbone-与-internvl3embedder)
8. [DualTreeAdapter_Evo1：当前训练与推理主入口](#8-dualtreeadapter_evo1当前训练与推理主入口)
9. [视觉树：SGMTS](#9-视觉树sgmts)
10. [门控融合：GateFusion](#10-门控融合gatefusion)
11. [记忆树：HierarchicalMemoryTree](#11-记忆树hierarchicalmemorytree)
12. [树读出：TreeSSMReadout](#12-树读出treessmreadout)
13. [语义提升：MLPElevation 与树操作](#13-语义提升mlpelevation-与树操作)
14. [跳变检测：JumpAwareHead](#14-跳变检测jumpawarehead)
15. [动作生成：FlowMatchingActionHead](#15-动作生成flowmatchingactionhead)
16. [遗留设计路径：policy/CrossModalFusion 分支](#16-遗留设计路径policycrossmodalfusion-分支)
17. [数据集与数据加载](#17-数据集与数据加载)
18. [训练流程](#18-训练流程)
19. [评估与可视化流程](#19-评估与可视化流程)
20. [配置系统与关键超参数](#20-配置系统与关键超参数)
21. [工程实现细节与性能注意事项](#21-工程实现细节与性能注意事项)
22. [论文撰写时必须注明的实现事实](#22-论文撰写时必须注明的实现事实)
23. [当前项目文件结构说明](#23-当前项目文件结构说明)

---

## 1. 文档定位与扫描范围

本文档基于对当前工作区的系统扫描整理而成，覆盖范围包括：

- 根目录入口脚本：pretrain.py、train.py、eval.py
- 包代码：dual_tree_vla/adapter/、dual_tree_vla/model/、dual_tree_vla/dataset/、dual_tree_vla/losses/、dual_tree_vla/policy/、dual_tree_vla/common/
- 配置：dual_tree_vla/config/*.yaml 与 dual_tree_vla/config/deepspeed/*.json
- 文档：README.md、REFACTOR_PLAN.md、docs/project_status.md、docs/evo1_analysis.md
- 运行脚本：scripts/*.py 与 scripts/*.sh
- 数据目录结构：data/libero/、data/RoboCerebra/
- 模型权重目录结构：model_weights/InternVL3-1B/、model_weights/CLIP/

说明：

- 本文档会扫描并总结数据集目录、元数据文件和 README，但不会逐个解析大型二进制数据文件。
- 论文或技术报告中若出现与本文档不一致的描述，应以当前源码为准，再回到本文档修正，而不是相反。

---

## 2. 项目定义与核心结论

### 2.1 项目定义

DualTreeVLA 是一个面向视觉-语言-动作模型的双树增强模块。它不是独立的 VLA 骨架，而是一个围绕现有骨架构建的旁路增强层，当前主实现面向 Evo-1 风格骨架，即：

- 视觉主干：InternVL3
- 语言主干：InternVL3 内部 language model
- 动作头：Flow Matching Action Head

DualTreeVLA 的核心思想是：

1. 不拆解或重写骨架内部 ViT 与 LLM 的预训练结构。
2. 通过只读 hook 获取视觉 patch 特征。
3. 通过门控融合把增强后的视觉 patch 特征回写到骨架视觉流。
4. 通过记忆树读出一个额外记忆 token，并追加到 LLM 输入序列末尾。

### 2.2 当前代码层面的核心事实

当前仓库存在两条并行的“架构叙事”：

- 一条是当前真实训练路径：DualTreeAdapter_Evo1 + InternVL3Backbone
- 一条是遗留/重构中的设计路径：policy/DualTreeVLA + CrossModalFusion

其中真正被 train.py、pretrain.py、eval.py、scripts/eval_server.py 使用的是前者；后者仍存在于仓库中，但不是当前主训练、离线评测或在线推理入口。

### 2.3 对论文写作最重要的结论

如果基于当前代码写论文，必须以如下实现为主描述对象：

- 视觉增强：SGMTS 直接处理 InternVL3 ViT 末层 patch 特征
- 融合方式：GateFusion 将 SGMTS 输出与原始 patch 特征逐元素门控融合
- 记忆注入：记忆树读出的 m_ctx 经过 mem_proj 后拼接到 LLM 输入序列
- 动作预测：FlowMatchingActionHead 接收 LLM 最后层 CLS 位置隐状态
- Phase 1/2 当前主训练中并没有走 CrossModalFusion -> 多 token context -> action head 这条路径

这点与较早期设计文档、部分旧说明和 policy/dual_tree_policy.py 中的设计并不完全一致。

---

## 3. 工作区全景与事实来源

### 3.1 源码主目录

当前源码按功能分为如下几层：

- dual_tree_vla/adapter/：骨架适配器，当前主入口
- dual_tree_vla/model/backbone/：InternVL3 自包含骨架封装
- dual_tree_vla/model/sgmts/：视觉树模块
- dual_tree_vla/model/memory_tree/：记忆树、树操作、Tree-SSM 读出
- dual_tree_vla/model/action_head/：Flow Matching 动作头与 JumpAwareHead 导出
- dual_tree_vla/model/common/：注意力与融合等通用模块
- dual_tree_vla/dataset/：RoboCerebra、RoboCerebraBench、LIBERO 加载器
- dual_tree_vla/policy/：遗留/重构中的策略层
- dual_tree_vla/common/：训练与工程工具函数

### 3.2 根目录入口

根目录脚本对应当前标准工作流：

- pretrain.py：Stage 0 预训练入口
- train.py：Phase 1 / Phase 2 训练入口
- eval.py：离线评估入口，已切换到 InternVL3Backbone + DualTreeAdapter_Evo1 主链路

### 3.3 辅助脚本

- scripts/pretrain.sh：多卡预训练启动脚本
- scripts/train_phase1.sh：Phase 1 启动脚本
- scripts/train_phase2.sh：Phase 2 启动脚本
- scripts/pretrain_eval.py：预训练树结构与热力图可视化
- scripts/eval_server.py：WebSocket 推理服务端，加载仓内 Evo1 backbone + adapter
- scripts/eval_client.py：LIBERO 仿真客户端，发送 Evo1 风格多视角 JSON（image 列表 + image_mask）
- scripts/extract_pretrain_features.py：RoboCerebra 预提取视觉特征缓存

### 3.4 文档层事实来源

- README.md：面向使用者的训练/评估说明
- REFACTOR_PLAN.md：目录重构计划，不等于当前真实结构
- docs/project_status.md：历史问题记录，含部分已过时信息
- docs/evo1_analysis.md：Evo-1 对照分析

### 3.5 数据目录事实来源

扫描到的数据目录如下：

- data/libero/libero_10/、libero_spatial/、libero_object/、libero_goal/
- data/libero/LIBERO/：LIBERO 工具库与官方项目
- data/RoboCerebra/RoboCerebra_trainset/
- data/RoboCerebra/RoboCerebraBench/

其中：

- data/libero/libero_10/meta/info.json 明确给出 total_episodes=379、total_frames=101469、fps=10
- RoboCerebra 的结构则通过加载器代码来定义和验证

---

## 4. 当前主实现路径与遗留路径

### 4.1 当前主实现路径

当前真正用于训练、离线评测和在线推理主实验的路径为：

```text
train.py / pretrain.py / eval.py / scripts/eval_server.py
    -> dual_tree_vla.model.backbone.InternVL3Backbone
    -> dual_tree_vla.adapter.DualTreeAdapter_Evo1
    -> SGMTS + GateFusion + HMT + TreeSSMReadout + mem_proj
    -> InternVL3 language_model
    -> InternVL3Backbone.action_head.predict_action()
```

### 4.2 遗留/规划路径

仓库中还存在一条未作为当前主训练入口使用的策略路径：

```text
dual_tree_vla.policy.DualTreeVLA
    -> SGMTSEncoder
    -> CrossModalFusion
    -> FlowMatchingActionHead
```

这条路径仍然具有文档和代码价值：

- 它保留了更“论文化”的模块分层表达
- 它解释了 CrossModalFusion 这个组件为何仍存在于仓库中
- 它与 REFACTOR_PLAN.md 中的策略层目标一致

但必须强调：它不是当前 train.py、pretrain.py、eval.py 或 scripts/eval_server.py 调用的真实路径。

### 4.3 文档撰写建议

若论文描述“系统设计理念”，可以介绍二者关系：

- “设计上”有明确的视觉树、记忆树、跨模态融合、动作生成四层结构
- “当前实现上”主训练路径采用 DualTreeAdapter_Evo1，其中跨模态融合退化为“视觉门控融合 + 记忆 token 拼接”而不是单独的 CrossModalFusion 模块

---

## 5. 整体系统架构

### 5.1 顶层数据流

当前主路径的单帧前向可以写成：

$$
I_t \xrightarrow{\text{InternVL3 ViT}} P_t
$$

$$
P_t \xrightarrow{\text{SGMTS}} Z_v
$$

$$
V_t' = \operatorname{GateFusion}(Z_v, P_t)
$$

$$
V_t' \xrightarrow{\text{pixel shuffle + mlp1}} E_t^{vis}
$$

$$
\bigl[E_t^{text}; E_t^{vis}; e_{mem}\bigr] \xrightarrow{\text{LLM}} H_t
$$

$$
h_t^{cls} = H_t[:,0,:]
$$

$$
\hat{A}_{t:t+H_a-1} = \operatorname{FlowHead}(h_t^{cls}, q_t)
$$

其中：

- $P_t$ 为 ViT 末层 patch 特征
- $Z_v$ 为视觉树增强后的 patch 特征
- $e_{mem}$ 为记忆树读出后经 mem_proj 映射的记忆 token
- $h_t^{cls}$ 为 LLM 输出序列的第一个位置隐状态

### 5.2 双树并行机制

系统存在两棵树：

1. 视觉树 SGMTS
   - 作用在单帧图像 patch 图结构上
   - 建树对象是空间 patch 节点
   - 输出是增强后的 patch 序列

2. 记忆树 HMT
   - 作用在时间序列的跨帧记忆上
   - 建树对象是历史子任务片段或抽象语义节点
   - 输出是一个低维记忆表征向量

这两棵树分别对应：

- 空间结构建模
- 时间层级建模

从研究视角看，这构成了空间树与时间树的双树协同系统。

### 5.3 训练阶段的系统角色分工

- Stage 0：训练双树模块本身学会边界检测、语义提升和空间扫描
- Phase 1：在冻结骨架下学习双树输出如何影响动作预测
- Phase 2：在保留 ViT 冻结的同时，低学习率放开 LLM 进一步适配任务

---

## 6. 与 Evo-1 骨架的接口设计

### 6.1 设计原则

当前适配器实现遵循三条原则：

1. 只读提取骨架内部中间特征
2. 不改写骨架权重加载逻辑
3. 双树模块通过最少接口注入增强信息

### 6.2 接口 1：ViT 末层 forward hook

适配器在 vision_model.encoder.layers[-1] 上注册 forward hook，抓取最后一层输出：

- 若输出为 tuple，则取 output[0]
- 保存在 self._P_t_raw

其作用是获得包含 CLS 的 ViT hidden states，再在后续处理中裁掉 CLS：

$$
P_t = \text{hs}[:,1:,:]
$$

### 6.3 接口 2：视觉 token 回写

SGMTS 在 ViT patch 空间中工作，输出与原始 patch 特征同维度的 $Z_v$。随后通过 GateFusion 得到 $V_t'$，再送入：

- pixel_shuffle
- mlp1

这一步把增强后的 patch 特征重新映射回 InternVL3 的语言嵌入空间，以便继续走原生的多模态 prompt 融合流程。

### 6.4 接口 3：记忆 token 追加

记忆树读出的 $m_{ctx}$ 经线性层 mem_proj 投影后，拼接在 input_embeds 最后：

$$
e_{mem} = W_{mem} m_{ctx}
$$

$$
E_t = [E_t^{orig}; e_{mem}]
$$

同时 attention mask 末尾补 1。

### 6.5 为什么这套接口是“零侵入”的

因为：

- AutoModel.from_pretrained() 的调用方式未改变
- ViT、LLM、mlp1 的权重结构未改变
- 不需要重训骨架预训练权重
- 适配器只包裹骨架，不替换骨架内部模块

这使得 DualTreeVLA 可以被理解为“骨架外部的可训练旁路增强层”。

---

## 7. 主干网络：InternVL3Backbone 与 InternVL3Embedder

### 7.1 InternVL3Backbone 的职责

dual_tree_vla/model/backbone/backbone.py 中的 InternVL3Backbone 负责两件事：

1. 持有 InternVL3Embedder
2. 持有 FlowMatchingActionHead

其接口非常薄，核心方法是：

- predict_action(fused_tokens, state, actions_gt=None, action_mask=None)

### 7.2 InternVL3Embedder 的职责

InternVL3Embedder 封装了：

- tokenizer
- InternVL3 主模型
- 图像预处理
- 构造多模态 prompt
- 将视觉 token 注入到 language model 输入 embedding 序列

### 7.3 图像预处理

_preprocess_images() 的输入是 List[Image | Tensor]，输出：

- pixel_values：所有图像切片拼接后的张量
- num_tiles_list：每张图对应的 tile 数

该函数支持：

- PIL 图像
- Tensor 图像
- 多余维度自动下采样到单帧

### 7.4 多模态 prompt 构建

_build_multimodal_prompt() 会把 <image> 占位符替换成 InternVL3 所需的：

- <img>
- 若干 <IMG_CONTEXT> token
- </img>

这意味着视觉 token 的序列长度由 tile 数决定，而不是固定写死。

### 7.5 LLM 输入 embedding 注入

_prepare_and_fuse_embeddings() 会：

1. 找到 prompt 中图像上下文 token 的位置
2. 用 vit_embeds 覆盖这些位置的 embedding
3. 按 image_mask 屏蔽无效相机 token

这里 DualTreeAdapter 的作用，就是把原本来自骨架 extract_feature() 的 vit_embeds 替换为双树增强后的版本。

### 7.6 Flash Attention 支持

当前骨架加载时传入：

- use_flash_attn=True

此外，注意力实现层还支持两级回退：

1. flash_attn 包
2. torch.nn.functional.scaled_dot_product_attention
3. 手写 softmax attention fallback

因此仓库对 Flash Attention 是“代码已支持，是否真正生效取决于运行环境和依赖安装”的状态。

---

## 8. DualTreeAdapter_Evo1：当前训练与推理主入口

### 8.1 类职责

dual_tree_vla/adapter/evo1_adapter.py 中的 DualTreeAdapter_Evo1 是当前真实主模型。它把以下组件组合在一起：

- backbone
- sgmts
- gate_fuse
- jump_head
- tree_ssm
- mlp_elev
- mem_proj
- sem_proj

### 8.2 适配器维护的状态

它内部维护三类重要状态：

1. _P_t_raw
   - 保存 hook 抓取到的 ViT hidden states

2. _tree_pool
   - 字典，batch_idx -> HierarchicalMemoryTree
   - 用于多样本并行时为每个样本保存独立记忆树

3. _mount_tau / _delta_w
   - 用于 HMT 插入规则

### 8.3 _embed_with_dual_tree()

这是单样本主前向的核心步骤：

1. 图像预处理与 extract_feature()
2. 通过 hook 获取 P_t
3. 编码任务文本得到 g_task
4. 从当前记忆树提取顶层抽象节点均值 s_top
5. SGMTS(P_t, g_task_tiled, s_top_list) -> Z_v
6. GateFusion(Z_v, P_t) -> V_t'
7. pixel_shuffle + mlp1 -> vit_embeds_new
8. 构建 prompt 与 input_embeds
9. 从 HMT 读取 m_ctx，经 mem_proj 拼接到序列末尾
10. 调用 language model，返回最后层 hidden states

### 8.4 _embed_batch_flow()

这是近期为解决 Phase 1/2 训练过慢问题引入的批量前向优化路径。它的关键思想是：

- Phase 1/2 并不在主训练中逐步更新 HMT 结构
- 因此可以把每个 batch 中每个样本原本串行的 ViT + LLM 前向合并成：
  - 一次共享 ViT forward
  - 一次共享 LLM forward

这使得 Phase 1/2 不再因为 for b in range(B) 重复走完整骨架而被严重拖慢。

### 8.5 forward() 的三种模式

适配器的 forward() 按 mode 分三类：

1. pretrain
   - 保留逐样本逻辑
   - 因为需要逐轨迹、逐帧更新 HMT
   - 计算 L_boundary、L_sem、L_elev

2. phase1
   - 走 _embed_batch_flow()
   - 仅计算 L_flow

3. phase2
   - 同样走批量 flow 路径
   - 仅计算 L_flow

### 8.6 inference()

推理路径复用单样本嵌入逻辑，最后调用：

$$
\hat{A} = \text{backbone.predict_action}(h^{cls}, q)
$$

其中 predict_action() 在没有 actions_gt 时返回采样得到的动作 chunk。

---

## 9. 视觉树：SGMTS

### 9.1 模块定位

SGMTS 位于 dual_tree_vla/model/sgmts/sgmts.py，是当前系统中负责空间结构建模的核心模块。与较早版本最大的不同是：

- 不再独立使用 CLIP 视觉编码器作为主训练路径输入
- 直接消费 InternVL3 ViT 最后一层 patch 特征

这使它不再承担“另起一套视觉主干”的角色，而是骨架视觉特征的结构增强器。

### 9.2 输入与输出

输入：

- P_t：形状 (B, N_p, d_patch)
- g_task：形状 (B, d_vit)
- s_top：长度为 B 的列表，每项为抽象节点均值或 None

输出：

- Z_v：形状 (B, N_p, d_patch)

可选输出：

- sigma_maps：每个 patch 的语义重要性分数

### 9.3 语义引导向量

对第 $b$ 个样本，SGMTS 首先构造：

$$
g_{sem}^{(b)} = \beta g_{task}^{(b)} + (1-\beta)s_{top}^{(b)}
$$

如果没有可用的 s_top，则退化为纯任务语义引导：

$$
g_{sem}^{(b)} = g_{task}^{(b)}
$$

### 9.4 语义重要性图

把 g_sem 投影到 patch 空间后，与每个 patch 计算 cosine 相似度：

$$
\sigma_i = \cos(p_i, W_g g_{sem})
$$

其中：

- $p_i$ 是第 $i$ 个 patch 特征
- $W_g$ 由 lang_gate 实现

### 9.5 语义加权最大生成树

SGMTS 并不是在完整图上建 MST，而是在规则网格的 4 邻域或 8 邻域边集上构图。边权定义为：

$$
w_{ij} = \cos(p_i, p_j) + \alpha \sigma_i \sigma_j
$$

当前默认：

- alpha = 0.5
- connectivity = 4

然后使用 Kruskal 构造最大生成树。

### 9.6 动态语义根与 BFS 扫描

扫描根节点取：

$$
r^* = \arg\max_i \sigma_i
$$

也就是说不是固定从左上角开始，而是从语义响应最强的 patch 开始。然后按 BFS 生成树扫描顺序。

### 9.7 Tree-SSM 扫描

每个 patch 首先注入语义偏置：

$$
X_i = p_i + \sigma_i W_{g'} g_{sem}
$$

再沿 BFS 序执行树状 SSM 递推：

$$
h_i = \bar{A}_i \odot h_{par(i)} + \bar{B}_i \odot X_i
$$

$$
z_i = C_i^\top h_i + D \odot X_i
$$

最后把扫描结果重排回原 patch 顺序，得到 $Z_v$。

### 9.8 实现意义

从论文角度，SGMTS 的意义可以概括为：

- 使用语义引导的最优树结构代替局部卷积或全连接注意力
- 在视觉主干被冻结时，仍为视觉 token 增加空间结构归纳偏置
- 通过树状 SSM 将空间建图复杂度从全局注意力风格的二次成本压缩到线性或近线性成本

---

## 10. 门控融合：GateFusion

### 10.1 模块位置

dual_tree_vla/model/gate_fusion.py

### 10.2 数学形式

给定：

- Z_v：SGMTS 输出
- V_t：原始 ViT patch 特征

先拼接再线性映射，得到门控权重：

$$
\alpha = \sigma\bigl(W_{gate}[Z_v; V_t]\bigr)
$$

然后融合：

$$
V_t' = \alpha \odot Z_v + (1-\alpha) \odot V_t
$$

### 10.3 初始化策略

W_gate.weight 初始化为 0，bias 初始化为 -5。这意味着训练初期：

$$
\sigma(-5) \approx 0.0067
$$

即：

$$
V_t' \approx V_t
$$

这样做的工程价值很大：

- 刚开始训练时不会大幅扰动骨架行为
- 双树模块以“接近零增益”的方式接入系统
- 后续再逐渐学到对哪些 patch 应增加增强分量

这是当前实现中一个非常关键、也非常值得写入论文实现细节的稳定化设计。

---

## 11. 记忆树：HierarchicalMemoryTree

### 11.1 模块定位

dual_tree_vla/model/memory_tree/tree.py

记忆树 HMT 用于在时间轴上维护任务执行的层级结构。它与视觉树不同，不在空间 patch 图上工作，而是在历史状态片段上工作。

### 11.2 节点类型

当前 HMT 节点支持两类：

1. 叶子节点
   - z_v：视觉语义向量
   - a_hist：动作历史
   - w：权重

2. 抽象节点
   - s：语义提升后的抽象向量
   - w：权重

### 11.3 维护目标

HMT 试图把时间序列组织成一棵层级树，以回答两个问题：

1. 当前片段是否应并入已有子任务
2. 当前片段是否应该触发一个新的分支并形成新的高层抽象

### 11.4 树更新的触发来源

在预训练中，HMT 更新主要由：

- 动作序列
- JumpAwareHead 的边界判定
- 当前视觉语义向量

共同驱动。

### 11.5 insert() 的意义

当前实现中，insert() 是 HMT 的核心在线更新接口，它会根据：

- 新的视觉表示
- 当前动作
- force_branch
- 可选当前语义表示

决定是：

- merge 到活动叶子
- branch 出新叶子
- 触发后续 elevation

### 11.6 深度控制

配置项 max_tree_depth 用于限制树深。超过上限后会进行剪枝，以避免长轨迹导致记忆树无限增长。

### 11.7 当前 Phase 1/2 与 HMT 的关系

需要特别说明：

- 预训练阶段，HMT 是主动更新的
- 当前 Phase 1/2 主训练路径中，为了批量化提速，主训练并不在每个 batch step 上完整维护 HMT 动态
- 因此 Phase 1/2 的“记忆 token”在批量路径中可以退化为零向量或简化读出

这说明当前代码的研究重心更偏向：

- 先把空间视觉增强与基本动作训练跑通
- 再逐步恢复和强化时序记忆机制

论文中若要宣称“完整在线记忆树已在所有阶段 fully active”，必须先核对实验具体使用的分支实现。

---

## 12. 树读出：TreeSSMReadout

### 12.1 模块位置

dual_tree_vla/model/memory_tree/tree_ssm.py

### 12.2 输入对象

输入不是普通张量，而是整个 HierarchicalMemoryTree 实例。

### 12.3 只扫描“有语义的抽象节点”

这是当前实现中极易被旧文档写错的一点。

代码里真正的筛选条件是：

- node.s is not None

而不是简单的：

- not node.is_leaf()

原因是：

- 剪枝后某些语义节点可能形式上变成没有子节点的“叶子”
- 但它们仍然携带抽象语义 s
- 因此仍应参与读出

### 12.4 输入投影

对每个语义节点，读出器构造：

$$
x_i = W_{abs}[s_i; \log(w_i)]
$$

映射到 d_ssm 维空间。

### 12.5 树状 SSM 递推

给定父节点隐藏状态 $h_{par(i)}$，当前节点执行：

$$
\Delta_i = \operatorname{softplus}(W_\Delta x_i) \odot \sigma(W_w \log w_i)
$$

$$
\bar{A}_i = \exp(\Delta_i A)
$$

$$
h_i = \bar{A}_i \odot h_{par(i)} + \bar{B}_i \odot x_i
$$

$$
y_i = C(x_i)^\top h_i + D \odot x_i
$$

### 12.6 输出语义

输出为 BFS 顺序上的语义节点表示矩阵：

$$
Y \in \mathbb{R}^{N_{sem} \times d_{ssm}}
$$

在当前适配器主路径中，真正使用的是：

$$
m_{ctx} = Y[-1]
$$

即最后一个 BFS 语义节点作为当前记忆向量。

---

## 13. 语义提升：MLPElevation 与树操作

### 13.1 模块组成

相关实现位于：

- dual_tree_vla/model/memory_tree/operations.py
- dual_tree_vla/model/memory_tree/tree.py

### 13.2 语义提升的意义

HMT 不应只是把所有历史帧当作链表堆起来，而应在边界处把多个低层片段压缩成高层语义节点。这个压缩过程由 MLPElevation 负责。

### 13.3 当前代码中的主要操作

树操作主要包括：

- merge
- branch
- semantic_elevation
- propagate_elevation_to_root

### 13.4 Merge

当当前片段被判定为延续已有活动叶子时，系统会对叶节点的：

- z_v
- a_hist
- w

做增量更新。

### 13.5 Branch

当 JumpAwareHead 判断出现边界时，系统创建新叶子节点，并准备触发语义提升。

### 13.6 Elevation

对于某个待提升父节点，MLPElevation 把其子节点的语义组合成新的抽象节点表示：

$$
s_{parent} = \operatorname{MLPElevation}(\text{child semantics})
$$

再向根方向传播，形成层级抽象。

### 13.7 当前实现中的设备桥接

树节点的长期存储张量通常保留在 CPU 侧，而 mlp_elev 权重位于训练设备上。因此适配器中专门实现了 _mlp_elev_cpu()，用于：

- 把 CPU tensor 移到 mlp_elev 所在设备
- 前向后再 .detach().cpu() 回到树中存储

这是一个很典型、也很容易在论文中被忽略的工程点：树结构本身是在线状态容器，而不是简单的 GPU 全驻留张量图。

---

## 14. 跳变检测：JumpAwareHead

### 14.1 模块位置

dual_tree_vla/model/action_head/jump_aware_head.py

这个文件本身只是导出封装，实际实现位于：

- dual_tree_vla/model/common/semantic_jump_head.py

### 14.2 输入

JumpAwareHead 只消费动作，不直接看视觉或文本：

- A_act：活动节点动作历史
- a_new：当前动作

### 14.3 结构

它是一个基于 Mamba/SSM 思想的轻量动作序列编码器。相比把边界检测建立在视觉或语言上，这样做有两个好处：

1. 边界信号更接近执行控制本身
2. 模块可以跨视觉骨架复用

### 14.4 输出

输出：

- p_jump
- logit

其中：

$$
p_{jump} = \sigma(\text{logit})
$$

在当前预训练路径里，force_branch 通常由是否超过阈值来决定。

### 14.5 边界监督

当前代码同时支持两类边界标签来源：

1. 若给定 subtask_ids，则相邻帧子任务 ID 变化处为边界
2. 否则退化为基于动作统计的自监督边界标签

这使 RoboCerebra 既可以用显式步骤标注，也可以在弱监督下训练。

---

## 15. 动作生成：FlowMatchingActionHead

### 15.1 模块位置

dual_tree_vla/model/action_head/flow_matching.py

### 15.2 架构定位

这是一个条件流匹配动作生成头。相比直接回归动作，它学习的是把噪声轨迹输运到动作轨迹的速度场。

### 15.3 输入与输出

输入：

- a_noisy：噪声或插值动作序列
- t：时间标量
- ctx：上下文 token 序列

输出：

- v_theta(a_t, t, ctx)

### 15.4 内部结构

该动作头由以下部分组成：

1. 动作投影层 a_in
2. 位置嵌入 pos_emb
3. 时间嵌入 TimestepEmbedding
4. 多层 FlowBlock
5. 输出层 out_proj

每个 FlowBlock 包含：

- 因果 self-attention
- 对上下文 token 的 cross-attention
- FFN
- 由时间嵌入生成的 AdaLN 调制参数

### 15.5 当前默认超参数

在当前实现中，默认是：

- d_model = 256
- n_layers = 4
- n_heads = 8
- H_a = 16
- N_ode = 20

这与 Evo-1 的某些更大配置不同，属于当前仓库实际实现而非概念配置。

### 15.6 训练损失

flow_loss() 中的采样机制为：

1. 采样时间：

$$
u \sim \mathcal{N}(0,1), \quad t = \sigma(u)
$$

2. 采样噪声：

$$
a_0 \sim \mathcal{N}(0,I)
$$

3. 插值：

$$
a_t = (1-t)a_0 + ta_{gt}
$$

4. 目标速度：

$$
v^* = a_{gt} - a_0
$$

5. 优化：

$$
\mathcal{L}_{flow} = \mathbb{E}\|v_\theta(a_t, t, ctx) - v^*\|_2^2
$$

### 15.7 推理

推理时从高斯噪声开始，进行 Euler 积分：

$$
a_{t+\Delta t} = a_t + \Delta t \cdot v_\theta(a_t, t, ctx)
$$

当前默认积分步数为 N_ode = 20。

---

## 16. 遗留设计路径：policy/CrossModalFusion 分支

### 16.1 为什么这部分仍要保留在文档里

因为仓库中确实存在：

- dual_tree_vla/policy/dual_tree_policy.py
- dual_tree_vla/model/common/fusion.py

并且它们表达了更标准的“视觉-记忆-语言-状态”融合设计。

### 16.2 CrossModalFusion 的数学形式

给定：

- z_v
- m_ctx
- g_lang
- q

代码实现的是：

$$
g = \sigma(W_g[z_v; m_{ctx}; q; g_{lang}])
$$

$$
f_1 = W_1[z_v; m_{ctx}], \qquad f_2 = W_2[q; g_{lang}]
$$

$$
f_{fused} = g \odot f_1 + (1-g) \odot f_2
$$

最后输出 (B, 1, d) 形式的单 token 上下文。

### 16.3 它与当前主路径的关系

当前主路径没有把 CrossModalFusion 接到训练主循环中，但它代表了仓库更完整的设计意图：

- 视觉树读出
- 记忆树读出
- 语言语义
- 本体感知状态

统一融合后再驱动动作头。

### 16.4 论文中如何处理

若论文要写“完整架构设计”，可以把 CrossModalFusion 写成：

- 仓库中实现并保留的跨模态融合层
- 当前主实验代码尚未把它作为唯一主路径接入

这种写法真实且可辩护，优于把它直接写成“当前所有实验都严格使用”的主链路。

---

## 17. 数据集与数据加载

## 17.1 RoboCerebra 训练集

目录结构：

```text
data/RoboCerebra/RoboCerebra_trainset/
    coffee_table/
    kitchen_table/
    study_table/
        caseN/
            demo.hdf5
            caseN.mp4
            task_description.json
```

加载器：dual_tree_vla/dataset/robocerebra.py

### 17.1.1 HDF5 内容

每个 demo.hdf5 下可能包含多个 demo_x，每个 demo 被视作一条独立轨迹。加载器读取：

- actions
- states

### 17.1.2 视频帧

若未启用特征缓存，则从 mp4 中按 subsample 提取帧并 resize 到配置尺寸。

### 17.1.3 子任务标注

task_description.json 中包含：

- 高层任务描述
- 每个步骤的自然语言子任务描述
- 起止时间戳

加载器会把它转换成：

- instruction
- subtask_ids
- subtask_descs

### 17.1.4 边界 mask

加载器还会生成基于动作统计的 boundary_mask，供预训练评估或弱监督使用。

### 17.1.5 预提取特征缓存

若设置 feat_cache_dir，加载器优先读取缓存特征：

- P_t_raw
- z_v_feat

这样预训练可以跳过重复的 ViT 前向。

## 17.2 RoboCerebraBench

目录结构：

```text
data/RoboCerebra/RoboCerebraBench/
    Ideal/
    Memory_Execution/
    Memory_Exploration/
    Mix/
    Observation_Mismatching/
    Random_Disturbance/
```

加载器：dual_tree_vla/dataset/robocerebra_bench.py

每个 case 下包含：

- demo.hdf5
- caseN.mp4
- task_description.txt
- goal.json
- *.bddl
- 可选 distractor 信息

该数据集用于离线评测，不需要仿真器。

## 17.3 LIBERO

当前扫描到的 LIBERO 子集包括：

- libero_10
- libero_spatial
- libero_object
- libero_goal

加载器：dual_tree_vla/dataset/libero.py

### 17.3.1 支持三种布局

LIBERO 加载器支持：

1. Layout A：自定义 inline bytes parquet
2. Layout B：LeRobot v2，parquet + mp4
3. Layout C：LeRobot v3，chunked parquet

### 17.3.2 当前 libero_10 的元数据

根据 data/libero/libero_10/meta/info.json：

- total_episodes = 379
- total_frames = 101469
- total_tasks = 10
- fps = 10
- 图像有两路：image 与 wrist_image
- 状态维度 8
- 动作维度 7

### 17.3.3 step-level 训练

LiberoDataset 默认在训练中使用：

- step_level = True

这意味着：

- 每个样本对应一帧观察
- 标签是从该帧开始的未来 H_a 步动作 chunk
- 整个 epoch 会枚举所有可用起始帧，而不是只取每条轨迹前 120 帧

因此当前训练已经是全量帧训练，不存在“只训练前 120 帧”的逻辑。120 只出现在可视化函数里。

### 17.3.4 缓存机制

为了避免反复解码整条视频，LIBERO 加载器采用：

- 以 episode 为粒度的 LRU cache
- 训练时 step-level 只解码当前一帧 JPEG
- episode-level 模式才会解码整条轨迹

这部分是训练速度能否接受的关键工程手段之一。

---

## 18. 训练流程

### 18.1 Stage 0：预训练

入口：pretrain.py

目标：

- 学会动作边界检测
- 学会视觉语义对齐
- 学会抽象节点提升

冻结模块：

- ViT
- LLM
- MLP projector
- FlowMatchingActionHead

可训练模块：

- SGMTS
- GateFusion
- sem_proj
- JumpAwareHead
- TreeSSMReadout
- MLPElevation
- mem_proj

损失：

$$
\mathcal{L}_{pretrain} = w_b \mathcal{L}_{boundary} + w_s \mathcal{L}_{sem} + w_e \mathcal{L}_{elev}
$$

默认权重来自 pretrain.yaml：

- w_boundary = 1.0
- w_sem = 0.5
- w_elev = 0.2

### 18.2 Phase 1：FlowMatching warm-up

入口：train.py --phase 1

当前真实可训练模块为：

- SGMTS
- GateFusion
- mem_proj
- backbone.action_head

也就是说，当前代码中的 Phase 1 注释与早期文档存在差异：不是“只训练 CrossModalFusion + ActionHead”，而是走适配器当前真实 L_flow 路径所需模块。

损失只有：

$$
\mathcal{L}_{flow}
$$

### 18.3 Phase 2：全量微调

入口：train.py --phase 2

策略：

- 先冻结全部参数
- 解冻双树模块
- 只解冻 language model
- ViT 和 mlp1 保持冻结

并且 language model 使用 0.1 倍学习率。

### 18.4 优化器与调度器

训练脚本使用：

- AdamW
- cosine decay with linear warmup

近期修复点：

- scheduler.step() 只在真实 optimizer.step() 发生时执行
- 避免了梯度累积时学习率调度错误加速的问题

### 18.5 DDP 与 DeepSpeed 状态

当前仓库内有：

- dual_tree_vla/config/deepspeed/ds_zero2.json
- dual_tree_vla/config/deepspeed/ds_zero3.json

但是否真正启用取决于启动命令。当前 Python 训练入口本身没有直接构造 deepspeed_plugin，所以 DeepSpeed 是“配置文件已存在，但需由外部启动方式显式接入”的状态。

### 18.6 当前已做的性能修复

为解决训练过慢，当前主路径已经做了三项关键修复：

1. Phase 1/2 引入批量 VLM 前向 _embed_batch_flow()
2. DDP 中关闭 find_unused_parameters=True
3. 默认把训练可视化频率从每个 epoch 调整为每 5 个 epoch 一次

这些修复对训练吞吐影响显著，属于论文复现实验应记录的工程配置差异。

---

## 19. 评估与可视化流程

### 19.1 离线评估

入口：eval.py

支持三类数据集：

- robocerebra
- robocerebra_bench
- libero

评估指标包括：

- action_l1
- action_l2
- tree_nodes
- tree_depth
- tree_branches
- tree_elevations
- subtask_boundary_f1
- prog_monotone_rate
- subtask_sr

### 19.2 预训练可视化

pretrain.py 中集成了 _run_pretrain_eval()，但默认 eval_every=999，即训练中几乎不自动执行。主要可视化入口是：

- scripts/pretrain_eval.py

它会输出：

- 树 JSON
- 视觉树语义重要性热力图

### 19.3 训练中视频可视化

train.py 中的 visualize_epoch() 会生成：

- GT vs Pred 对比视频

注意：这个函数默认只截取 viz_max_frames 帧，不影响训练样本覆盖率，只影响日志视频长度。

### 19.4 WebSocket 在线评估

仓库还提供 server/client 评估链路：

- scripts/eval_server.py
- scripts/eval_client.py

其作用是：

- 在服务端加载模型并保留 GPU
- 客户端驱动 LIBERO 仿真环境
- 二者通过 WebSocket 交互观测和动作块
- 当前 `scripts/eval_server.py` 提供 argparse 入口；`scripts/eval_client.py` 仍通过顶部 `Args` 类配置运行参数，而不是命令行参数

### 19.5 图像翻转与 gripper 阈值

这一部分在当前仓库中仍是重要实现细节：

- eval_client.py 当前对 agentview_image 和 wrist_image 都使用 [::-1, ::-1]，即旋转 180 度
- 这已经与当前 Evo1 对照实现保持一致

gripper 二值化方面：

- 当前客户端使用阈值 0.0
- 这是基于 z-score 反归一化后的动作空间假设

这两点在真实仿真成功率上都可能产生显著影响，论文实验需要固定说明。

---

## 20. 配置系统与关键超参数

### 20.1 配置文件

当前主要配置文件为：

- dual_tree_vla/config/pretrain.yaml
- dual_tree_vla/config/train_phase1.yaml
- dual_tree_vla/config/train_phase2.yaml
- dual_tree_vla/config/default.yaml

### 20.2 关键模型维度

当前主路径常用关键超参数：

- d_vit = 896
- d_ssm = 256
- d_state = 16
- d_a = 7
- H_a = 16
- n_ode = 20 或配置中指定值
- mount_tau = 0.4
- alpha = 0.5
- delta_w = 0.1
- max_tree_depth = 4

### 20.3 LIBERO 数据配置

根据 train_phase1.yaml：

- root = data/libero/libero_10
- img_h = img_w = 224
- d_q = 8
- normalize = true

### 20.4 RoboCerebra 数据配置

根据 pretrain.yaml：

- subsample = 4
- max_seqlen = 1400
- img_h = img_w = 224

### 20.5 训练超参数概览

预训练默认：

- batch_size = 2
- epochs = 30
- lr = 3e-4

Phase 1 默认：

- batch_size = 8
- grad_accum = 2
- lr = 1e-4

Phase 2 默认：

- batch_size = 8
- lr = 3e-5 级别
- language model 使用 0.1 倍学习率

---

## 21. 工程实现细节与性能注意事项

### 21.1 依赖现状

requirements.txt 中声明：

- torch
- torchvision
- transformers
- accelerate
- timm
- deepspeed
- opencv-python
- h5py
- wandb
- websockets

并明确写明：

- flash-attn 为可选项，建议单独安装

### 21.2 Flash Attention

仓库在三个层面支持 Flash Attention：

1. InternVL3 加载时 use_flash_attn=True
2. FlashMHA 中优先尝试 flash_attn 包
3. 若不可用，则回退到 PyTorch SDPA

### 21.3 DeepSpeed

仓库包含：

- ZeRO-2 配置
- ZeRO-3 配置

但当前主入口不会自动启用，需要外部脚本或启动命令正确传入 accelerate/deepspeed 配置。

### 21.4 tokenizer 并行

训练脚本显式设置：

```text
TOKENIZERS_PARALLELISM=false
```

这是为了减少多进程日志污染和并发不可控行为。

### 21.5 mixed precision

当前主训练默认使用：

- bf16

这也是当前 A6000/Ampere 平台上较合理的默认选择。

---

## 22. 论文撰写时必须注明的实现事实

以下事实如果不写清楚，会导致论文描述与代码实现失配：

### 22.1 当前主实验路径不是旧版 CrossModalFusion 主链路

主训练和主推理依赖 DualTreeAdapter_Evo1，而不是 policy/DualTreeVLA。

### 22.2 当前视觉树直接消费骨架 ViT patch，而不是独立 CLIP 主干

旧文档和历史状态里曾出现 CLIP 版叙述，但当前主实现已经转为 ViT hook 路径。

### 22.3 当前动作头上下文来自 LLM 最后层序列，而不是明确的多 token 双树融合上下文块

当前适配器最终传给 predict_action() 的是 fused_hidden[:,0,:]，然后在 backbone 中被扩展成 1-token context。

### 22.4 HMT 在预训练中是 fully active，在 Phase 1/2 当前批量训练中并非完整逐步更新

这是一个必须如实说明的系统实现现实。

### 22.5 LIBERO 训练是全量 step-level，而不是固定长度片段前 120 帧

120 帧仅用于视频可视化。

### 22.6 当前仓库同时包含“现实现”和“重构目标”

REFACTOR_PLAN.md 和 policy/ 提供了更理想化的结构目标，但论文结果若基于当前训练脚本，应以 adapter/backbone 主路径为准。

---

## 23. 当前项目文件结构说明

下面给出面向当前实现的结构化说明。

```text
DualTreeVLA/
├── CONSTRUCTION.md
├── README.md
├── REFACTOR_PLAN.md
├── pretrain.py
├── train.py
├── eval.py
├── requirements.txt
├── setup.py
│
├── docs/
│   ├── project_status.md
│   └── evo1_analysis.md
│
├── dual_tree_vla/
│   ├── __init__.py
│   ├── adapter/
│   │   ├── __init__.py
│   │   ├── base_adapter.py
│   │   └── evo1_adapter.py
│   ├── common/
│   │   ├── checkpoint_util.py
│   │   ├── normalizer.py
│   │   └── pytorch_util.py
│   ├── config/
│   │   ├── default.yaml
│   │   ├── pretrain.yaml
│   │   ├── train_phase1.yaml
│   │   ├── train_phase2.yaml
│   │   └── deepspeed/
│   │       ├── ds_zero2.json
│   │       └── ds_zero3.json
│   ├── dataset/
│   │   ├── __init__.py
│   │   ├── base_dataset.py
│   │   ├── libero.py
│   │   ├── robocerebra.py
│   │   └── robocerebra_bench.py
│   ├── losses/
│   │   ├── __init__.py
│   │   └── tree_losses.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── gate_fusion.py
│   │   ├── action_head/
│   │   │   ├── __init__.py
│   │   │   ├── flow_matching.py
│   │   │   └── jump_aware_head.py
│   │   ├── backbone/
│   │   │   ├── __init__.py
│   │   │   ├── backbone.py
│   │   │   └── internvl3_embedder.py
│   │   ├── common/
│   │   │   ├── __init__.py
│   │   │   ├── attn.py
│   │   │   ├── fusion.py
│   │   │   └── semantic_jump_head.py
│   │   ├── memory_tree/
│   │   │   ├── __init__.py
│   │   │   ├── node.py
│   │   │   ├── operations.py
│   │   │   ├── tree.py
│   │   │   └── tree_ssm.py
│   │   └── sgmts/
│   │       ├── __init__.py
│   │       └── sgmts.py
│   └── policy/
│       ├── __init__.py
│       ├── base_policy.py
│       └── dual_tree_policy.py
│
├── scripts/
│   ├── demo_robocerebra.py
│   ├── eval_client.py
│   ├── eval_server.py
│   ├── extract_pretrain_features.py
│   ├── pretrain.sh
│   ├── pretrain_eval.py
│   ├── train_phase1.sh
│   └── train_phase2.sh
│
├── data/
│   ├── libero/
│   │   ├── LIBERO/
│   │   ├── libero_10/
│   │   ├── libero_goal/
│   │   ├── libero_object/
│   │   └── libero_spatial/
│   └── RoboCerebra/
│       ├── RoboCerebraBench/
│       └── RoboCerebra_trainset/
│
├── model_weights/
│   ├── CLIP/
│   ├── Evo1_LIBERO/
│   └── InternVL3-1B/
│
├── logs/
├── outputs/
└── results/
```

---

## 结语

如果把 DualTreeVLA 作为论文对象，最准确的表述方式是：

- 它是一个围绕 Evo-1 风格骨架构建的双树增强系统
- 空间端由 SGMTS 提供结构化视觉增强
- 时间端由 HMT 提供层级记忆组织
- 当前主实现通过 GateFusion 与记忆 token 追加把双树信息接回骨架
- 仓库中同时保留了更理想化的策略层与跨模态融合实现，为后续重构和论文抽象提供基础

这样的写法既能真实反映当前代码，又保留了理论叙述空间，不会在答辩、复现或审稿时被源码反证。
