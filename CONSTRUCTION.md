# MemoryTreeVLA — 论文框架构建文档

> **版本**: v0.1 · **日期**: 2026-03-31  
> **状态**: 交互讨论草稿，欢迎随时提出修改意见

---

## 一、论文定位与标题候选

### 问题动机

当前 VLA（Vision-Language-Action）模型在**超长程任务**（long-horizon tasks）上存在两大根本缺陷：

1. **无结构记忆**：以往方法（如 RoboMamba、OpenVLA）仅靠固定长度窗口的 token 序列表示历史，无法有效保留跨阶段的因果关系与分层语义；缺乏对"重要子任务节点"的强化保留机制。
2. **视觉编码语义盲区**：ViT 的规则 patch 扫描对任务无关区域投入等同注意力，导致高分辨率图像推理中存在大量冗余，且无法动态聚焦任务相关区域。

### 标题候选

| 候选标题 | 侧重点 |
|---|---|
| **MemoryTreeVLA: Hierarchical Memory Trees with Semantic-Guided Mamba Scanning for Long-Horizon Robot Manipulation** | 全面，适合主会 |
| **Growing Memory: Tree-Structured Episodic Representations for Long-Horizon Robotic VLA** | 记忆树为主 |
| **TreeScan-VLA: Minimum Spanning Tree Mamba Encoding for Semantic-Aware Robot Manipulation** | SGMTS 为主 |

> **推荐**: 第一个（全称），可根据投稿会议再精炼。

---

## 二、整体架构概览

```
输入: 任务语言描述 ℓ + 当前图像 I_t + 关节状态 q_t
                        │
           ┌────────────┼────────────┐
           ▼            ▼            ▼
      [SGMTS模块]  [记忆树读取]  [本体感知编码]
       视觉Token     记忆Token     关节Token
        Z^V           Z^M           e^Q
           │            │            │
           └────────────┼────────────┘
                        ▼
                 [层次跨模态融合]
                    Z^fused
                        │
                        ▼
                [Qwen2.5 LLM]
                        │
                        ▼
                  [动作生成头]
                     a_t
                        │
                        ▼
              [记忆树更新模块] ←── 写回新节点 v_t
```

---

## 三、核心创新一：层次记忆树（Hierarchical Memory Tree, HMT）

### 3.1 理论动机

生物记忆研究表明，情景记忆（episodic memory）以层次化、因果树状结构组织——重要事件被强化，无关细节被遗忘，高层语义对低层细节进行抽象。我们将这一机制引入机械臂操控的历史建模中。

### 3.2 节点定义

时刻 $t$ 的记忆树 $\mathcal{T}_t$ 中，每个节点 $v_i$ 存储一个**六元组**：

$$
v_i = \left(\mathbf{z}_i^v,\;\mathbf{A}_i,\;\mathbf{q}_i,\;\mathbf{s}_i,\;n_i,\;w_i\right)
$$

| 符号 | 含义 | 维度 |
|---|---|---|
| $\mathbf{z}_i^v$ | 视觉嵌入的**在线均值**（滚动更新） | $\mathbb{R}^d$ |
| $\mathbf{A}_i$ | 本节点所有时间步动作的**有序列表** | $(\mathbb{R}^{d_a})^{n_i}$ |
| $\mathbf{q}_i$ | 最近时刻的关节状态（滚动更新） | $\mathbb{R}^{d_q}$ |
| $\mathbf{s}_i$ | 语义嵌入的**在线均值**（可随层提升后固化） | $\mathbb{R}^d$ |
| $n_i$ | 融合入本节点的时间步计数 | $\mathbb{Z}^+$ |
| $w_i$ | 记忆重要性权重（含时长信号） | $\mathbb{R}^+$ |

> SSM 读取时取 $\mathbf{a}_i = \mathbf{A}_i[-1]$（最近动作）作为代表动作。

### 3.3 树的动态构建——三路插入决策

#### 语义距离与新时刻语义初始化

$$
d_{\text{sem}}(\mathbf{s}_i, \mathbf{s}_j) = 1 - \frac{\mathbf{s}_i \cdot \mathbf{s}_j}{\|\mathbf{s}_i\|\|\mathbf{s}_j\|} \in [0,2]
$$

$$
\mathbf{s}_t = \text{MLP}_{\text{sem}}\!\left([\mathbf{z}_t^v;\, \mathbf{g}]\right), \qquad \mathbf{g} = \text{TextEnc}(\ell)
$$

设当前"活跃节点"为 $v_{\text{act}}$（最近一次被创建或更新的节点），计算：

$$
d_t = d_{\text{sem}}\!\left(\mathbf{s}_t,\;\mathbf{s}_{\text{act}}\right)
$$

以阈值 $\theta_{\text{fuse}} \in (0,2)$ 为界，依据 $d_t$ 执行**二路决策**：

$$
\boxed{\text{Decision}(d_t) = \begin{cases}
\textbf{(A) 融合更新} & d_t < \theta_{\text{fuse}} \\
\textbf{(B) 分支创建} & d_t \geq \theta_{\text{fuse}}
\end{cases}}
$$

两个区间的语义对应：**A** = 同一语义阶段持续（如"持续施力拧螺丝"），当前观测与活跃节点在同一语义邻域内；**B** = 可辨别的语义跳变（子任务切换，如"抓取→放置"），需要在树中新建节点以标记这一转折。层次化的抽象结构由语义提升操作（操作②）主动构建，而非依赖顺序插入形成的线性链。

---

#### 决策 A — 融合更新（Merge Update）

$d_t < \theta_{\text{fuse}}$，机器人处于与活跃节点**相同的语义状态**，就地更新 $v_{\text{act}}$，**不创建新节点**：

**在线均值更新**（Welford 递推，无偏估计）：

$$
n_{\text{act}} \leftarrow n_{\text{act}} + 1
$$

$$
\mathbf{z}_{\text{act}}^v \;\leftarrow\; \mathbf{z}_{\text{act}}^v + \frac{\mathbf{z}_t^v - \mathbf{z}_{\text{act}}^v}{n_{\text{act}}}, \qquad
\mathbf{s}_{\text{act}} \;\leftarrow\; \mathbf{s}_{\text{act}} + \frac{\mathbf{s}_t - \mathbf{s}_{\text{act}}}{n_{\text{act}}}
$$

**关节状态与动作追加**：

$$
\mathbf{q}_{\text{act}} \leftarrow \mathbf{q}_t, \qquad \mathbf{A}_{\text{act}} \leftarrow \mathbf{A}_{\text{act}} \;\|\; [\mathbf{a}_t]
$$

**权重增量**（时长信号：在此语义状态停留越久，节点越重要）：

$$
\boxed{w_{\text{act}} \leftarrow w_{\text{act}} + \delta_w}
$$

活跃节点 $v_{\text{act}}$ 保持不变，树结构不生长。

> **自保持性分析**：每步融合满足 $d_t < \theta_{\text{fuse}}$，EMA 步长 $1/n_{\text{act}}$ 随访问次数单调递减，语义中心漂移速率为 $O(1/n)$，语义嵌入被"锚定"在语义邻域内，不会无限漂移。
>
> **与 SSM 的自然耦合**：融合次数多 → $w_i$ 高 → $\Delta_i$ 大（见 3.5 节）→ 稳定长阶段的记忆在树递推中衰减最慢，这一机制完全自动涌现，无需额外设计。

---

#### 决策 B — 分支创建（Branch Split）

$d_t \geq \theta_{\text{fuse}}$，发生**语义跳变**。沿祖先链向上寻找语义最近的"最优分叉点"：

$$
v_{\text{anc}}^* = \underset{v_k \in \mathcal{A}(v_{\text{act}}) \cup \{v_{\text{act}}\}}{\arg\min}\; d_{\text{sem}}(\mathbf{s}_t, \mathbf{s}_k)
$$

$$
\text{par}(v_t) \leftarrow v_{\text{anc}}^*, \qquad v_{\text{act}} \leftarrow v_t
$$

分支创建后，若 $v_{\text{anc}}^*$ 的子节点数达到触发阈值 $K_{\text{elev}}$，立即执行一次**语义提升**（操作②）。

---

#### 训练时梯度可微化（Soft Gating）

硬阈值切换在反向传播中断梯度（离散决策）。训练阶段改用软门控：

$$
\mu_t = \sigma\!\left(\frac{\theta_{\text{fuse}} - d_t}{\tau}\right) \in (0,1)
$$

视觉嵌入的软更新（融合与新建并行计算，加权混合）：

$$
\tilde{\mathbf{z}}_{\text{out}} = \mu_t \cdot \underbrace{\left(\mathbf{z}_{\text{act}}^v + \frac{\mathbf{z}_t^v - \mathbf{z}_{\text{act}}^v}{n_{\text{act}}+1}\right)}_{\text{融合路径}} + (1-\mu_t) \cdot \underbrace{\mathbf{z}_t^v}_{\text{新节点路径}}
$$

梯度经 $\mu_t$（可用 straight-through estimator）和 $\mathbf{z}_t^v$ 正常回传。推理时令 $\tau \to 0^+$ 退化为硬决策。

### 3.4 四种树操作

#### 操作 ① — 记忆强化（Memory Reinforcement）

记忆强化模拟人类"用进废退"效应，通过**访问频率追踪**、**梯度驱动重要性更新**和**任务相关表示更新**三维机制，提升关键经验的保留与检索效率。

**（a）访问频率追踪**

访问计数器 $n_{\text{access}}$ 在节点 6-元组中作为 $n_i$ 字段维护，每次检索或在 3.3 节的融合更新中命中该节点时递增：

$$
n_i \leftarrow n_i + 1
$$

**（b）梯度驱动的重要性加权更新**

节点 $v_C$ 的重要性权重 $w_C$ 依据其对当前任务损失的贡献自动调整——梯度幅度大的节点表明其表示对当前预测影响显著，应予以强化：

$$
\boxed{w_C^{\text{new}} = w_C + \eta \cdot \|\nabla_{\Phi_C} L_{\text{task}}\|_2 \cdot \mathbf{1}\!\left[\|\nabla_{\Phi_C} L_{\text{task}}\|_2 > \theta_{\text{grad}}\right]}
$$

其中 $\Phi_C$ 为节点 $v_C$ 的可微分嵌入参数（即 $\mathbf{z}_C^v$ 和 $\mathbf{s}_C$），$\eta$ 为权重更新步长，$\theta_{\text{grad}}$ 为梯度阈值（过滤噪声节点）。该机制在每次 episode 的反向传播后执行，无需额外奖励信号即可自动识别对任务成功至关重要的记忆片段。

**（c）任务相关采样的节点表示更新**

节点表示采用**指数移动平均（EMA）与任务相关采样**相结合的方式更新，确保更新方向与当前任务需求一致：

$$
\boxed{\Phi_C^{\text{new}} = (1 - \alpha_{\text{ema}}) \cdot \Phi_C + \alpha_{\text{ema}} \cdot \frac{\displaystyle\sum_{i \in \text{batch}} w_i^{\text{task}} \cdot \Phi_i}{\displaystyle\sum_{i \in \text{batch}} w_i^{\text{task}}}}
$$

其中任务相关采样权重为：

$$
w_i^{\text{task}} = \exp\!\left(\frac{\text{sim}(\Phi_i,\, \Phi_{\text{query}})}{\tau_{\text{task}}}\right)
$$

$\Phi_{\text{query}}$ 为当前时间步的查询嵌入（由当前视觉观测与语言指令融合得到），$\tau_{\text{task}}$ 为温度系数，$\alpha_{\text{ema}} \in (0,1)$ 控制历史与当前 batch 的混合比例。

#### 操作 ② — 语义提升（Semantic Elevation）

**动机**：随着多次分支创建，某个节点 $v_p$ 会积累多个子节点，这些子节点代表在 $v_p$ 语义基础上衍生出的不同子阶段。当子节点数量达到阈值 $K_{\text{elev}}$ 时，说明 $v_p$ 下的子阶段已足够丰富，适合引入一层新的抽象。语义提升操作通过**新建一个抽象父节点 $v_{\text{abs}}$**，将 $v_p$ 的部分子节点归并到其下，在树中主动插入更高层次的语义抽象，使树形成真正的层次结构而非扁平分叉。

**触发条件**：$|\text{ch}(v_p)| \geq K_{\text{elev}}$

**操作流程**：

① 对 $v_p$ 的子节点集合 $\text{ch}(v_p) = \{v_1, \ldots, v_K\}$ 按语义相似度聚类，选出语义最集中的一个子集 $\mathcal{G} \subseteq \text{ch}(v_p)$

$$
\mathcal{G} = \underset{S \subseteq \text{ch}(v_p),\;|S|=\lfloor K/2 \rfloor}{\arg\max} \sum_{v_i \in S} w_i
$$

② 创建新抽象节点 $v_{\text{abs}}$，其语义嵌入由 $\mathcal{G}$ 中节点加权聚合生成：

$$
\mathbf{s}_{\text{abs}} = \text{MLP}_{\text{elev}}\!\left(\frac{\sum_{v_i \in \mathcal{G}} w_i\,[\mathbf{z}_i^v;\,\mathbf{s}_i]}{\sum_{v_i \in \mathcal{G}} w_i}\right)
$$

$$
w_{\text{abs}} = \sum_{v_i \in \mathcal{G}} w_i, \qquad n_{\text{abs}} = \sum_{v_i \in \mathcal{G}} n_i
$$

其余字段取 $\mathcal{G}$ 中权重最大节点的 $\mathbf{z}^v$ 和 $\mathbf{q}$ 作为 $v_{\text{abs}}$ 的初始值。

③ 将 $v_{\text{abs}}$ 插入 $v_p$ 与 $\mathcal{G}$ 之间：

$$
\text{par}(v_{\text{abs}}) \leftarrow v_p, \qquad \forall v_i \in \mathcal{G}:\;\text{par}(v_i) \leftarrow v_{\text{abs}}
$$

$v_p$ 的子节点集合中 $\mathcal{G}$ 的位置被 $v_{\text{abs}}$ 替代，$\mathcal{G}$ 的成员深度各增加 1。

**结构效果**：

```
提升前:          提升后（新建 v_abs）:
   v_p              v_p
  ╱│╲              ╱  ╲
v1 v2 v3         v_abs  v3
                 ╱   ╲
               v1     v2
```

深度增加一层，$v_{\text{abs}}$ 成为 $\{v_1, v_2\}$ 的语义章节标题，而 $v_3$（语义差异较大）继续挂在 $v_p$ 下等待下次提升或分组。

#### 操作 ③ — 剪枝（Pruning）

定期扫描记忆树，将**不重要且没有子节点**的叶节点删除，防止树无限膨胀。满足以下两个条件的节点会被移除：

- **条件一**：$w_i < \theta_w$，即该节点的重要性权重低于阈值（长期未被用到、对任务贡献小）
- **条件二**：$\text{isLeaf}(v_i)$，即该节点是叶节点（删除中间节点会断开子树，因此只删叶节点）

用集合符号描述为——将当前树 $\mathcal{T}_t$ 中所有满足上述两个条件的节点 $v_i$ 从树中移除，得到更新后的树：

$$
\boxed{\mathcal{T}_t \;\leftarrow\; \mathcal{T}_t \;\setminus\; \bigl\{\,v_i \;:\; \underbrace{w_i < \theta_w}_{\text{权重太低}} \;\wedge\; \underbrace{\text{isLeaf}(v_i)}_{\text{是叶节点}}\,\bigr\}}
$$

**直觉示例**：

```
剪枝前：           剪枝后（v3、v5 权重低于 θ_w）：
    root               root
   ╱    ╲             ╱    ╲
  v1     v2          v1     v2
 ╱  ╲     ╲         ╱
v3   v4    v5      v4
(低) (高)  (低)    (高)
```

叶节点 $v_3$（低权重）和 $v_5$（低权重）被删除；$v_4$ 权重高，保留；$v_2$ 虽然子节点被删完、变为叶节点，但其自身 $w_{v_2}$ 若高于 $\theta_w$ 则保留，下轮再判断。

剪枝后，若被删节点的父节点从多子变为单子（或无子），对其重新判断是否触发语义提升。

### 3.5 记忆树 SSM 读取（Tree-SSM Readout）

#### 设计动机：只扫描上层高语义节点

记忆树中，**上层节点**（深度 $\leq K'$，尤其是语义提升操作②创建的抽象节点）经过多次 Welford 融合与跨子任务聚合，其语义嵌入 $\mathbf{s}_i$ 是稳定、抽象的子任务摘要；**底层叶节点**是当前 episode 最新插入的临时观测，尚未经过语义沉淀，引入 SSM 只会引入序列噪声。

因此，Tree-SSM Readout 采用**扫描前预过滤（Pre-filter）**策略：

$$
\boxed{\mathcal{V}_{\text{upper}} = \left\{\,v_i \in \mathcal{T} \;\Big|\; \text{depth}(v_i) \leq K'\,\right\}}
$$

SSM 递推仅在 $\mathcal{V}_{\text{upper}}$ 上进行，叶节点及深层临时节点**完全不参与计算**。由于 BFS 序保证父节点早于子节点出现，且 $\text{depth}(\text{par}(v)) = \text{depth}(v) - 1$，若 $v \in \mathcal{V}_{\text{upper}}$ 则其父节点必然也在 $\mathcal{V}_{\text{upper}}$ 中，因此预过滤后父子隐状态传递关系完全一致，不会断链。

---

取 $\mathcal{V}_{\text{upper}}$ 的 BFS 序 $\pi = (v_1, v_2, \ldots, v_{N_M})$（$N_M = |\mathcal{V}_{\text{upper}}|$），仅在此子集上执行 SSM：

**输入投影**：

$$
x_i = W_{\text{in}}\,[\mathbf{z}_i^v;\, \mathbf{a}_i;\, \mathbf{q}_i;\, \mathbf{s}_i;\, \log w_i] + b_{\text{in}}
$$

**节点自适应时间步**（重要性越高，时间步越大，信息保留越多）：

$$
\Delta_i = \text{softplus}\!\left(W_\Delta x_i + b_\Delta\right) \odot \sigma(W_w \log w_i + b_w)
$$

**离散化**（零阶保持）：

$$
\bar{A}_i = \exp(\Delta_i \cdot A), \qquad \bar{B}_i = (e^{\Delta_i A} - I) A^{-1} B(x_i)
$$

**树递推**（沿父子边传递隐状态，仅在 $\mathcal{V}_{\text{upper}}$ 内传递）：

$$
\boxed{h_i = \bar{A}_i \odot h_{\text{par}(i)} + \bar{B}_i \odot x_i}, \qquad h_{\text{root}} = \mathbf{0}
$$

**输出与记忆 Token**（$\mathcal{V}_{\text{upper}}$ 全部节点输出即为 $Z^M$，无需再次过滤）：

$$
y_i = C(x_i)\, h_i + D\, x_i, \qquad Z^M = \left\{y_i \;:\; v_i \in \mathcal{V}_{\text{upper}}\right\} \in \mathbb{R}^{N_M \times d}
$$

| 对比维度 | 原设计（后过滤） | **新设计（预过滤）** |
|---|---|---|
| SSM 计算范围 | 所有 $N$ 个节点 | 仅 $N_M \leq N$ 个上层节点 |
| 叶节点隐状态 | 计算但丢弃 | **不计算** |
| 父子链完整性 | 完整 | **完整**（depth 性质保证） |
| 计算效率 | $O(N)$ | $O(N_M)$，$N_M \ll N$ 时显著提速 |
| 语义纯净度 | 上层+噪声叶混合 | **仅稳定抽象语义** |

> **关键特性**：$\Delta_i$ 的权重自适应使得重要节点（高 $w_i$）的记忆在树向上传播时衰减更慢，实现"重要记忆不遗忘"的效果。结合预过滤，$Z^M$ 完全由经过语义沉淀的抽象节点构成，为后续 LLM 条件化提供干净的长程历史表示。

### 3.6 记忆树训练损失

$$
\mathcal{L}_{\text{tree}} = \mathcal{L}_{\text{recon}} + \lambda_1\,\mathcal{L}_{\text{sem}} + \lambda_2\,\mathcal{L}_{\text{prog}} + \lambda_3\,\mathcal{L}_{\text{depth}}
$$

**① 语义重建损失**（语义提升的信息保留性）：

$$
\mathcal{L}_{\text{recon}} = \sum_{v_p} \left\|\text{Dec}_{\text{sem}}(\mathbf{s}_p) - \frac{1}{|\text{ch}(v_p)|}\sum_{v_i \in \text{ch}(v_p)} \mathbf{s}_i\right\|_2^2
$$

**② 子任务对比损失**（不同子任务语义应可分）：

$$
\mathcal{L}_{\text{sem}} = -\frac{1}{|P|}\sum_{(i,j) \in P} \log \frac{\exp(\cos(\mathbf{s}_i, \mathbf{s}_j^+) / \tau_s)}{\sum_{k} \exp(\cos(\mathbf{s}_i, \mathbf{s}_k) / \tau_s)}
$$

其中 $(i, j^+)$ 为同一子任务阶段的正样本对，负样本从不同子任务采样。

**③ 任务进度单调损失**（沿祖先-后代路径，进度应单调递增）：

首先定义进度预测头：

$$
p_i = \sigma\!\left(\text{MLP}_{\text{prog}}(\mathbf{s}_i)\right) \in [0,1]
$$

$p_i$ 的含义是：从节点 $v_i$ 的语义嵌入出发，预测此刻任务已完成的比例（0=刚开始，1=已结束）。直觉上，任务越往后执行，对应的语义嵌入应编码越高的完成度。

**排序约束仅施加在祖先-后代对（同一条根到叶路径上）**，不约束不同分支上的兄弟节点（它们不存在绝对时序）：

$$
\boxed{\mathcal{L}_{\text{prog}} = \frac{1}{|\mathcal{P}|}\sum_{(v_i,\, v_j)\,\in\,\mathcal{P}} \max\!\left(0,\; p_i - p_j + \epsilon\right)}
$$

$$
\mathcal{P} = \left\{(v_i, v_j) \;\Big|\; v_i \in \mathcal{A}(v_j),\; v_j \in \mathcal{T}\right\}
$$

即 $\mathcal{P}$ 是树中所有满足"$v_i$ 是 $v_j$ 的祖先"的有序节点对。损失惩罚一切**祖先进度 $p_i$ 大于后代进度 $p_j$ 的违例情况**（允许 $\epsilon$ 的松弛间隔）。

**直觉示例**（以"抓取→移动→放置"三节点主链为例）：

```
根(p≈0.0) → v_抓取(p≈0.3) → v_移动(p≈0.6) → v_放置(p≈0.9)
```

若 $p_{v\_抓取} > p_{v\_移动}$（违反单调性），则产生正的损失项；若正确单调则损失为零。不同分支上的节点（平行探索的子树）彼此间**不施加此约束**，因为它们不存在绝对时序先后。

> **与原公式的本质区别**：原公式用 `depth(vi) < depth(vj)` 作为排序代理，在有分支的树中是错的——不同分支上深度3的节点和深度2的节点未必有时序先后关系，强行约束会引入矛盾的梯度。正确的做法是只约束结构上有传递关系的祖先-后代对。

**④ 语义提升一致性损失**（新建的抽象节点应确实比其子节点更通用）：

$$
\boxed{\mathcal{L}_{\text{elev}} = \sum_{v_{\text{abs}}} \max\!\left(0,\; d_{\text{sem}}(\mathbf{s}_{\text{abs}},\, \bar{\mathbf{s}}_{\mathcal{G}}) - \gamma\right)}
$$

$$
\bar{\mathbf{s}}_{\mathcal{G}} = \frac{\sum_{v_i \in \mathcal{G}} w_i\,\mathbf{s}_i}{\sum_{v_i \in \mathcal{G}} w_i}
$$

惩罚 $v_{\text{abs}}$ 的语义嵌入与其子节点均值偏离过大（超过间隔 $\gamma$），确保抽象节点是子节点语义的真实概括而非随机嵌入。

更新总损失为：

$$
\mathcal{L}_{\text{tree}} = \mathcal{L}_{\text{recon}} + \lambda_1\,\mathcal{L}_{\text{sem}} + \lambda_2\,\mathcal{L}_{\text{prog}} + \lambda_3\,\mathcal{L}_{\text{elev}}
$$

---

## 四、核心创新二：语义引导 Mamba 最小生成树扫描（SGMTS）

### 4.1 理论动机

传统 ViT 将图像切分为均匀 patch，以全局自注意力建模全局依赖，计算复杂度为 $O(P^2)$，且对任务无关 background 区域与语义关键区域投入等量计算资源。VisionMamba 引入一维 S 形蛇形序列，将复杂度降为 $O(P)$，但蛇形扫描将空间邻近却方向不同的 patch 分离到序列两端，破坏了局部内容关联。

**GrootVL（NeurIPS 2024 Spotlight，arXiv:2406.02395）** 提出以**最小生成树（MST）**组织 patch 扫描拓扑：在图像 patch 特征构成的 4-连通图上用余弦相似度定义边权，通过 Kruskal 算法提取 MST，再沿 BFS 遍历顺序执行树状 Mamba SSM 递推。视觉特征相近的 patch 在 MST 中相互靠近，SSM "父传子"递推使得**空间连续的语义区域**共享隐状态，兼顾了 $O(L)$ 线性复杂度与空间层次上下文建模。

然而，GrootVL 的 MST 构建**纯粹依赖视觉特征**，与当前任务的语言指令完全无关——执行"抓取红色杯子"时，桌面背景与红色杯子在 MST 边权上无任何差异。本节提出 **SGMTS（Semantic-Guided Mamba Tree Scanning）**，在 GrootVL 树状 SSM 框架的基础上，在 MST 边权、BFS 根节点选取以及 SSM 时间步三处注入语言语义引导，其余结构与 GrootVL 保持完全兼容，可直接复用其 CUDA 核（`TreeScan`）。

### 4.2 GrootVL 原理精确回顾（基于源码）

本节基于 GrootVL 开源代码（`tree_scan_core.py` / `tree_scanning.py` / `grootv.py`）精确还原其三个核心步骤，为后续 SGMTS 改动提供清晰基线。

#### 4.2.1 四连通图 + 余弦相似度边权 → MST

给定视觉特征图 $F \in \mathbb{R}^{B \times C \times H \times W}$，构建**四连通邻接图**（仅水平与垂直相邻边，共 $(H{-}1)W + H(W{-}1)$ 条候选边）：

```python
# tree_scan_core.py: _build_feature_weight_cosine()
weight_row = torch.cosine_similarity(fm[:,:,:-1,:], fm[:,:,1:,:], dim=1)  # 水平边
weight_col = torch.cosine_similarity(fm[:,:,:,:-1], fm[:,:,:,1:], dim=1)  # 垂直边
weight = torch.cat([weight_row, weight_col], dim=1)
# 对 min-tree：取 -weight（余弦相似度越大，负值越小，MST 优先选择）
weight = mapping_func(-weight)
```

对相邻 patch 对 $(i, j)$ 的 MST 目标等价于：

$$
\mathcal{T}^* = \arg\min_{\mathcal{T} \subseteq E} \sum_{(i,j) \in \mathcal{T}} (-\cos_{ij}), \qquad \cos_{ij} = \frac{\mathbf{f}_i^\top \mathbf{f}_j}{\|\mathbf{f}_i\|\|\mathbf{f}_j\|}
$$

即**余弦相似度最大的相邻 patch 对**会被 MST 优先收录，从而让视觉特征相近的 patch 在同一子树中。MST 通过 CUDA C++ 扩展 `_C.mst_forward`（Kruskal 算法）以近 $O(P)$ 复杂度实现。

#### 4.2.2 BFS 遍历 → 扫描序

对 MST $\mathcal{T}^*$ 执行广度优先搜索（`_C.bfs_forward`），生成：

$$
(\text{sorted\_index},\ \text{sorted\_parent},\ \text{sorted\_child})
$$

其中 `sorted_index[k]` 为 BFS 访问序第 $k$ 位的 patch 索引，`sorted_parent[k]` 为其父节点索引。此三元组完整编码树结构，后续 CUDA 核仅需遍历此线性序列即可完成 SSM 递推。

#### 4.2.3 树状 SSM 递推（`tree_scan_refine_forward`）

沿 BFS 序，对每个 patch $i$ 计算选择性状态空间递推（标准 Mamba，S4D 初始化）：

$$
\Delta_i = \mathrm{softplus}(W_\Delta r_i + b_\Delta), \qquad \bar{A}_i = e^{\Delta_i A}, \qquad \bar{B}_i = (e^{\Delta_i A} - I)A^{-1}B(x_i)
$$

$$
h_i = \bar{A}_i \odot h_{\mathrm{par}(i)} + \bar{B}_i \odot x_i, \qquad y_i = C(x_i)\,h_i + D\,x_i
$$

隐状态 $h_{\mathrm{par}(i)}$ 来自**树中父节点**而非序列上一元素，是区别于常规 Mamba 的核心处。CUDA 实现采用**自底向上聚合 + 自顶向下广播**的两遍动态规划，整体复杂度 $O(P)$。

`edge_coef`（边权系数）在精化传播 `tree_scan_refine_forward` 中控制父子信息的混合比例，提供对树结构信心的软调节。

#### 4.2.4 GrootVLayer 整体架构

```python
# grootv.py: GrootVLayer.forward()
x_in = in_proj(x)          # 投影成 x, z（门控）
x, z  = x_in.chunk(2, -1)
x     = act(conv2d(x))     # 深度卷积（局部上下文，d_conv > 1 时启用）
y     = forward_core(x)    # Tree_SSM: tree_scanning()
y     = y * act(z)         # 门控激活
out   = out_proj(y)

# 残差 + Layer Scale
x = x + drop_path(γ₁ · TreeSSM(norm₁(x)))
x = x + drop_path(γ₂ · MLP(norm₂(x)))
```

$\gamma_1, \gamma_2 \in \mathbb{R}^d$ 为逐通道可学习缩放因子（Layer Scale），抑制深层训练不稳定性。多个 `GrootVLayer` 堆叠构成 `GrootVBlock`，对应 ViT 中的 Transformer Block。

### 4.3 SGMTS：在 GrootVL 基础上的四处语义引导改进

SGMTS 在 GrootVL 三个核心步骤的**四处关键位置**注入语言语义，其余代码逻辑与 GrootVL 保持完全兼容。

#### 4.3.1 语言指令语义评分（前置共享计算）

将任务语言指令 $\ell$ 经文本编码器（冻结的 Qwen2.5 前若干层，隐状态均值池化）得到指令向量 $\mathbf{g} \in \mathbb{R}^{d_g}$，投影后与各 patch 特征计算语义相关度：

$$
\boxed{r_i^{\text{sem}} = \sigma\!\left(\frac{(W_g \mathbf{g})^\top \mathbf{f}_i}{\sqrt{d_f}}\right) \in [0,1], \quad i = 1,\ldots, P}
$$

$r_i^{\text{sem}}$ 越高，表示 patch $i$ 的视觉内容与当前任务指令越相关。该向量在下面四处复用。

#### 4.3.2 改进①：语义调制 MST 边权

将 GrootVL 的纯视觉余弦边权替换为：

$$
\boxed{w_{ij}^{\text{SGMTS}} = (1 - r_i^{\text{sem}})(1 - r_j^{\text{sem}}) \cdot (-\cos_{ij}) + \epsilon}
$$

**直觉**：若 $p_i, p_j$ 均与任务高度相关（$r_i, r_j \to 1$），则 $(1-r_i)(1-r_j) \to 0$，边权 $\approx \epsilon$（极小），Kruskal 算法优先纳入此边，使任务相关区域在 MST 中**直接相连**，形成连续子树；若两端均与任务无关，则退化为 GrootVL 原始的余弦距离权重，保持后向兼容。

#### 4.3.3 改进②：语义引导根节点

GrootVL 默认以固定顶点（如左上角）为 BFS 根，SGMTS 改为：

$$
\text{root} = \arg\max_{i} r_i^{\text{sem}}
$$

以**语义得分最高的 patch** 为 BFS 起点，任务焦点区域在 BFS 序中排列最前，SSM 递推时最先积累语义上下文，高层次任务状态从任务焦点向外传播。

#### 4.3.4 改进③：语义注入 SSM 输入

GrootVL 的 SSM 输入仅为视觉特征线性投影 $W_\text{in}\mathbf{f}_i$。SGMTS 以相关性加权的方式注入语言条件：

$$
x_i = W_{\text{in}} \mathbf{f}_i + r_i^{\text{sem}} \cdot W_g' \mathbf{g}
$$

语义相关性强的 patch 受语言引导影响更大；任务无关区域（$r_i \approx 0$）保持纯视觉驱动，避免语言噪声干扰低相关 patch 的特征。

#### 4.3.5 改进④：语义自适应时间步

$$
\boxed{\Delta_i = \mathrm{softplus}(W_\Delta x_i) \cdot (1 + \beta \cdot r_i^{\text{sem}})}
$$

更大的 $\Delta_i$ 使状态矩阵 $\bar{A}_i = e^{\Delta_i A}$ 衰减更慢，子节点可更强地继承父节点隐状态。语义相关 patch 因此在 SSM 递推链中**信息保留更持久**，其特征向量可有效传播到更深层的子节点。$\beta \geq 0$ 为可学习标量参数（初始化为 0，训练中自适应）。

### 4.4 计算复杂度分析

| 步骤 | 操作 | 复杂度 |
|---|---|---|
| 语义评分 $r_i^{\text{sem}}$（改进①②③④共享） | 投影 + 内积 | $O(P \cdot d_f)$ |
| 语义调制边权计算 | 逐元素乘 | $O(P)$ |
| MST 构建（CUDA Kruskal） | `_C.mst_forward` | $O(P\,\alpha(P)) \approx O(P)$ |
| BFS 排序 | `_C.bfs_forward` | $O(P)$ |
| 树状 SSM 递推（上下两遍 DP） | `tree_scan_refine_forward` | $O(P)$ |
| **SGMTS 端到端总复杂度** | — | $\mathbf{O(P \cdot d_f)}$ |

相比 ViT 自注意力 $O(P^2 d)$，SGMTS 保持 GrootVL 的线性复杂度 $O(P)$，四处语义引导改进对渐近复杂度无额外量级开销（均为 $O(P)$ 或 $O(P d_f)$）。

### 4.5 SGMTS 与相关方法精确对比

| 特性 | ViT | VisionMamba | **GrootVL（源码）** | **SGMTS（本文）** |
|---|---|---|---|---|
| 扫描拓扑 | 无（全局 Attn） | 蛇形序列 | 4-连通 MST（余弦相似度） | 4-连通 MST（**语义调制**余弦相似度） |
| 边权定义 | N/A | N/A | $-\cos_{ij}$（纯视觉） | $(1{-}r_i)(1{-}r_j)(-\cos_{ij})$（语义调制） |
| BFS 根节点 | N/A | 固定左上 | **固定顶点**（源码默认） | $\arg\max_i r_i^{\text{sem}}$（任务焦点） |
| SSM 输入 | N/A | $W_\text{in}x_i$ | $W_\text{in}\mathbf{f}_i$ | $W_\text{in}\mathbf{f}_i + r_i^{\text{sem}} W_g'\mathbf{g}$ |
| 时间步 $\Delta_i$ | N/A | 输入依赖 | 输入依赖 | **任务语义**调制 |
| 语言指令引导 | ✗ | ✗ | ✗ | ✅（边权 + 根 + 输入 + $\Delta$） |
| 任务自适应 | ✗ | ✗ | ✗ | ✅ 每帧动态重建 |
| 渐近复杂度 | $O(P^2 d)$ | $O(P)$ | $O(P)$ | $O(P \cdot d_f)$ |

---

## 五、核心创新三：多模态融合与动作生成

### 5.1 三流特征

| 流 | 来源 | 维度 |
|---|---|---|
| $Z^V$ | SGMTS 视觉 Token | $\mathbb{R}^{P \times d}$ |
| $Z^M$ | 记忆树 SSM 读取 Token | $\mathbb{R}^{N_M \times d}$ |
| $\mathbf{e}^Q$ | 关节状态 MLP 编码 | $\mathbb{R}^{d}$ |

### 5.2 层次跨模态融合

**第一层：记忆引导视觉聚合**（用记忆 query 视觉 K/V，挑选与过往经历相关的视觉区域）

$$
\tilde{Z}^{VM} = \text{MultiHeadCrossAttn}\!\!\left(\,Q=Z^M,\; K=Z^V,\; V=Z^V\,\right)
$$

**第二层：本体感知集成**

$$
\mathbf{e}^Q_{\text{exp}} = \mathbf{e}^Q \cdot \mathbf{1}_{N_M}^\top \in \mathbb{R}^{N_M \times d}
$$

$$
Z^{\text{fused}} = \text{LayerNorm}\!\left(\tilde{Z}^{VM} + \text{MLP}\!\left([\tilde{Z}^{VM};\, \mathbf{e}^Q_{\text{exp}}]\right)\right) \in \mathbb{R}^{N_M \times d}
$$

### 5.3 LLM 条件化

将 $Z^{\text{fused}}$ 线性投影后，拼接到 Qwen2.5 的语言 Token 前缀：

$$
X^{\text{LLM}} = \left[\text{Proj}_{\text{VL}}(Z^{\text{fused}});\; \text{TokenEmb}(\ell)\right] \in \mathbb{R}^{(N_M + L) \times d_{\text{LLM}}}
$$

LLM 自回归处理后输出最终隐状态 $H^{\text{LLM}} \in \mathbb{R}^{(N_M+L) \times d_{\text{LLM}}}$。

### 5.4 动作生成头（Flow Matching Action Head）

参考 Evo-1（CVPR 2026，arXiv:2511.04555）的 `FlowmatchingActionHead` 设计，结合本工程的融合特征 $Z^{\text{fused}}$ 和本体状态 $\mathbf{e}^Q$，采用**动作块预测 + 线性流匹配 ODE + 跨模态去噪 Transformer**架构，替代直接回归，以捕捉多模态动作分布。

#### 5.4.1 为什么选 Flow Matching 而非直接回归

直接 MSE 回归动作均值会对多模态动作分布（同一状态下多种合理操作策略）产生"模糊"预测，导致机器人在关键路径段动作犹豫。Flow Matching 将动作生成建模为从**标准噪声 → 专家动作**的连续 ODE 过程，学习速度场而非点估计，天然支持多峰分布。

#### 5.4.2 动作块定义

预测未来 $H_a$ 步的动作块（Action Chunk），而非单步动作，避免高频控制带来的短视行为：

$$
\mathbf{a}_{t:t+H_a} \in \mathbb{R}^{H_a \times d_a}, \qquad d_a = 7 \; (\text{Franka: 6D }\Delta\text{-pose} + \text{1D gripper})
$$

#### 5.4.3 条件上下文构建

将融合特征与本体状态拼接，构成去噪 Transformer 的跨注意力键值：

$$
\mathbf{e}^Q_{\text{enc}} = \text{MLP}_{\text{state}}(\mathbf{e}^Q) \in \mathbb{R}^d
$$

$$
C = \left[Z^{\text{fused}};\; \mathbf{e}^Q_{\text{enc}}\right] \in \mathbb{R}^{(N_M+1) \times d}
$$

$Z^{\text{fused}}$ 携带记忆树语义与视觉融合信息，$\mathbf{e}^Q_{\text{enc}}$ 提供当前实时本体感知，二者共同约束动作生成方向。

#### 5.4.4 动作编码器（ActionTokenizer）

参考 Evo-1 的 `MultiEmbodimentActionEncoder`，将含噪动作块编码为与上下文同维的 Token 序列：

$$
X_a = \text{ActionEnc}\!\left(\mathbf{a}^{\tau}_{t:t+H_a}\right) \in \mathbb{R}^{H_a \times d}
$$

具体地，对批内每一时间步 $k$：

$$
X_a^{(k)} = \text{ReLU}(W_3\,(\text{ReLU}(W_1\,\mathbf{a}^{\tau,(k)}) + \text{PE}(k) + \text{ReLU}(W_2\,(\ldots))))
$$

其中 PE$(k)$ 为正弦位置编码，用于区分动作块内不同时间步。

#### 5.4.5 时间步嵌入

将连续流时间 $\tau \in [0,1]$ 离散化为正弦嵌入，注入去噪 Transformer 每一层的前馈 sublayer：

$$
\mathbf{t}_{\text{emb}} = \text{PE}\!\left(\lfloor \tau \times 1000 \rfloor\right) \in \mathbb{R}^d
$$

#### 5.4.6 跨模态去噪 Transformer

堆叠 $L$ 层 `BasicTransformerBlock`，每层以动作 Token 为 Query，以条件上下文 $C$ 为 Key/Value，时间嵌入注入前馈层：

$$
X_a^{(0)} = \text{ActionEnc}(\mathbf{a}^\tau_{t:t+H_a})
$$

$$
X_a^{(l)} = X_a^{(l-1)} + \text{CrossAttn}\!\left(\text{LN}(X_a^{(l-1)}),\; C,\; C\right)
$$

$$
X_a^{(l)} \leftarrow X_a^{(l)} + \text{FFN}\!\left(\text{LN}(X_a^{(l)}) + \mathbf{t}_{\text{emb}}\right), \quad l = 1, \ldots, L
$$

最终对动作 Token 序列做扁平化池化后通过 MLP 输出预测速度场：

$$
\hat{\mathbf{v}} = \text{MLP}_{\text{head}}\!\left(W_{\text{pool}}\,\text{flatten}(X_a^{(L)})\right) \in \mathbb{R}^{H_a \times d_a}
$$

#### 5.4.7 训练：线性插值 ODE + 速度回归损失

**时间采样**：参考 Evo-1，采用 Beta(2, 2) 分布采样流时间 $\tau$，该分布在 $[0,1]$ 中部密集，避免过度拟合噪声端($\tau \to 0$)和数据端($\tau \to 1$)的极端退化情形：

$$
\tau \sim \text{Beta}(2,\,2),\quad \tau \in [0.02, 0.98]
$$

**线性插值构造含噪动作**（直线 ODE 路径，噪声到真实动作的恒定速度场）：

$$
\boldsymbol{\varepsilon} \sim \text{Uniform}[-1, 1]^{H_a \times d_a}, \qquad \mathbf{a}^\tau = (1-\tau)\,\boldsymbol{\varepsilon} + \tau\,\mathbf{a}^*_{t:t+H_a}
$$

**训练目标**：回归恒定速度场 $\mathbf{v}^* = \mathbf{a}^* - \boldsymbol{\varepsilon}$（LLM 速度回归而非 score matching）：

$$
\boxed{L_{\text{flow}} = \mathbb{E}_{\tau,\,\boldsymbol{\varepsilon},\,\mathbf{a}^*}\!\left[\left\|\hat{\mathbf{v}}_\theta\!\left(\mathbf{a}^\tau,\, C,\, \tau\right) - (\mathbf{a}^* - \boldsymbol{\varepsilon})\right\|_2^2\right]}
$$

#### 5.4.8 推理：Euler ODE 积分

从均匀噪声出发，以 $N$ 步 Euler 积分沿速度场推进到动作端点（$\tau: 0 \to 1$）：

$$
\mathbf{a}^{(0)} \sim \text{Uniform}[-1,1]^{H_a \times d_a}
$$

$$
\mathbf{a}^{(i+1)} = \mathbf{a}^{(i)} + \frac{1}{N} \cdot \hat{\mathbf{v}}_\theta\!\left(\mathbf{a}^{(i)},\, C,\, \frac{i}{N}\right), \quad i = 0, 1, \ldots, N{-}1
$$

最终 $\mathbf{a}^{(N)} = \mathbf{a}_{t:t+H_a}$ 即为输出动作块，机器人执行前 $k$（$k < H_a$）步后以新观测重新预测（时序聚合策略）。

#### 5.4.9 与 Evo-1 原版的差异对比

| 设计维度 | Evo-1 原版 | **本工程适配** |
|---|---|---|
| 条件来源 | InternVL3 图像 Token + 语言 Token | **记忆树融合特征 $Z^{\text{fused}}$**（含历史语义） |
| 状态注入 | `CategorySpecificMLP(state)` | `MLP_state(e^Q)` — 本体感知嵌入 |
| 多实体支持 | `CategorySpecificLinear`（多 embodiment 独立权重） | 单实体（Franka），预留扩展接口 |
| 动作维度 | 可配置（MetaWorld 24D） | $d_a = 7$（6D $\Delta$-pose + gripper） |
| 时间采样 | Beta(2,2) ∩ [0.02, 0.98] | **保留**（同 Evo-1） |
| ODE 步数 | $N = 20$ | $N = 20$（可调） |
| 上下文长度 | 固定帧数 | $N_M$（由记忆树节点数动态决定） |

---

## 六、完整训练策略

### 6.1 数据集分析与对应关系

本工程使用两个互补的数据集驱动三阶段训练，各数据集的结构特点与模型组件形成直接对应：

#### RoboCerebra（NeurIPS 2025，arXiv:2506.06677）

- **数据规模**：大规模长程仿真数据集，家庭操作环境，延伸任务时域（extended task horizon）
- **标注特点**：GPT 生成任务指令 → 自动分解为 $K$ 个子任务序列 → 人工遥操作执行各子任务 → 形成**层次化、密集标注**的轨迹数据
- **与本工程的对应**：
  - 子任务边界标签 → **分支创建（B）** 的监督信号（子任务切换时 $d_t \geq \theta_{\text{fuse}}$）
  - 子任务序列顺序 → **进度单调损失 $\mathcal{L}_{\text{prog}}$** 的正样本对
  - 任务 → 子任务层次结构 → **语义提升（②）** 的目标树拓扑（验证 $v_{\text{abs}}$ 是否与 GPT 分解一致）
  - System 2 规划/记忆评估维度 → 直接测试 HMT 的记忆积累质量

#### LIBERO（NeurIPS 2023，arXiv:2306.03310）

- **数据规模**：130 个任务，四个子集，每任务 50 条人工遥操作演示
- **四个子集及用途**：

| 子集 | 任务数 | 核心挑战 | 本工程用途 |
|---|---|---|---|
| LIBERO-SPATIAL | 10 | 空间关系泛化 | SGMTS 语义引导验证 |
| LIBERO-OBJECT | 10 | 物体类别迁移 | 记忆树跨物体检索 |
| LIBERO-GOAL | 10 | 目标多样性 | 语言条件动作头 |
| **LIBERO-LONG** | 10 | **长程 8+ 步操作** | **主训练集（动作策略）** |

- **与本工程的对应**：高质量遥操作演示驱动 Flow Matching 动作头训练；LIBERO-LONG 为长程记忆能力提供行为克隆监督

---

### 6.2 三阶段渐进式训练课程

采用**模块渐进解冻**策略，共分三个阶段逐步引入模型组件与损失。每阶段 LLM 骨干（默认 Qwen2.5-0.5B）保持冻结，直至 Phase 3。

| 阶段 | 训练模块 | 损失函数 | 建议 epochs | 建议 lr |
|---|---|---|---|---|
| **Phase 1** 视觉预热 | SGMTS + s_proj + TreeSSMReadout + recon_decoder | $\mathcal{L}_{\text{recon}}$ | 20 | 1e-4 |
| **Phase 2** 动作头 | （Phase 1 全部）+ CrossModalFusion + prog_head + FlowMatchingActionHead | $\mathcal{L}_{\text{flow}} + \mathcal{L}_{\text{prog}}$ | 30 | 1e-4 |
| **Phase 3** 联合微调 | 全部参数（含 LLM） | $\mathcal{L}_{\text{flow}} + \mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{prog}}$ | 10 | 1e-5 |

#### Phase 1：视觉语义预热

```
目标:   让 SGMTS + s_proj + TreeSSMReadout 建立有效的语义特征表达
数据:   RoboCerebra 完整轨迹
损失:   L_recon — 连续帧语义预测：recon_decoder(s_t) ≈ s_{t+1}
        （辅助：树中父节点 s_p → 子节点均值 s_ch 的语义重建）
冻结:   LLM、CrossModalFusion、FlowMatchingActionHead
训练:   SGMTS、s_proj、TreeSSMReadout、recon_decoder
```

#### Phase 2：动作预测头训练

```
目标:   在已收敛的视觉与记忆特征基础上训练动作生成
数据:   RoboCerebra 完整轨迹
损失:   L_flow  — Flow Matching 速度场 MSE（Beta(2,2) 时间采样）
        L_prog  — 进度单调损失（维持树结构语义秩序）
冻结:   LLM
训练:   Phase 1 全部 + CrossModalFusion + prog_head + FlowMatchingActionHead
```

#### Phase 3：全参数联合微调

```
目标:   端到端优化全模型，提升跨任务泛化与长程操作成功率
数据:   RoboCerebra 完整轨迹（可混入其他数据集）
损失:   L_total = L_flow + w_recon·L_recon + w_prog·L_prog
冻结:   无（LLM 全参数解冻，建议使用 ds_zero3.json 降低显存压力）
训练:   全模型
```

---

### 6.3 总体损失函数

$$
\mathcal{L}_{\text{total}} = \underbrace{\mathcal{L}_{\text{flow}}}_{\text{Flow Matching 动作损失}} + w_{\text{recon}} \underbrace{\mathcal{L}_{\text{recon}}}_{\text{语义重建损失}} + w_{\text{prog}} \underbrace{\mathcal{L}_{\text{prog}}}_{\text{进度单调损失}}
$$

| 损失项 | 计算方式 | 权重 | 激活阶段 |
|---|---|---|---|
| $\mathcal{L}_{\text{flow}}$ | Flow Matching 速度场 MSE（Beta(2,2) 时间采样，见 5.4.7 节） | 1.0 | Phase 2-3 |
| $\mathcal{L}_{\text{recon}}$ | `recon_decoder(s_t) ≈ s_{t+1}`（连续帧语义预测）+ 树父→子均值重建 | 0.5 | Phase 1, 3 |
| $\mathcal{L}_{\text{prog}}$ | 祖先-后代对进度单调 hinge：$\text{mean}(\max(0,\, p_{\text{anc}} - p_{\text{desc}} + \varepsilon))$ | 0.3 | Phase 2-3 |

---

### 6.4 记忆树操作触发时机

| 操作 | 触发条件 | 频率 |
|---|---|---|
| 融合更新（A） | $d_t < \theta_{\text{fuse}}$，每步 | $t=1,2,\ldots$ |
| 分支创建（B） | $d_t \geq \theta_{\text{fuse}}$，每步 | 按需（子任务边界处密集） |
| 记忆强化（①） | Phase 3 每步反向传播后，依据 $\|\nabla_{\Phi_C}L_{\text{task}}\|$ | 每训练步 |
| 语义提升（②） | $|\text{ch}(v_p)| \geq K_{\text{elev}}$（RoboCerebra 中对应子任务数 $\leq 6$） | 按需 |
| 剪枝（③） | 树节点数 $> N_{\text{max}}$ 或 episode 结束 | 每 $K=50$ 步 |

## 十、符号表速查

| 符号 | 含义 |
|---|---|
| $\mathcal{T}_t$ | 时刻 $t$ 的记忆树 |
| $v_i$ | 记忆树节点 $i$ |
| $v_{\text{act}}$ | 当前活跃节点 |
| $\mathbf{z}_i^v$ | 节点 $i$ 的视觉嵌入在线均值 |
| $\mathbf{z}_i^{v,\text{init}}$ | 节点 $i$ 初建时的视觉嵌入（固定） |
| $\mathbf{A}_i$ | 节点 $i$ 融合的动作时序列表 |
| $\mathbf{q}_i^{\text{init}}$ | 节点 $i$ 初建时的关节状态（固定，用于回溯） |
| $\mathbf{s}_i$ | 节点 $i$ 的语义嵌入在线均值 |
| $n_i$ | 节点 $i$ 融合的时间步计数 |
| $w_i$ | 节点 $i$ 的记忆权重 |
| $h_i$ | 节点 $i$ 的 SSM 隐状态 |
| $\Delta_i$ | 节点 $i$ 的自适应时间步 |
| $\theta_{\text{fuse}}$ | 融合/分支的唯一判断阈值 |
| $K_{\text{elev}}$ | 触发语义提升的子节点数阈值 |
| $\delta_w$ | 融合更新权重增量 |
| $p_i$ | 节点 $i$ 预测任务进度 |
| $\mathcal{P}$ | 树中所有祖先-后代有序对集合（用于进度损失） |
| $\mathcal{A}(v_i)$ | 节点 $v_i$ 的祖先集合 |
| $\mathcal{G}$ | 语义提升时选出的子节点分组 |
| $v_{\text{abs}}$ | 语义提升新建的抽象父节点 |
| $Z^V$ | SGMTS 输出视觉 Token |
| $Z^M$ | 记忆树 SSM 输出 Token |
| $\mathbf{e}^Q$ | 关节状态嵌入 |
| $Z^{\text{fused}}$ | 融合后 Token |
| $\mathbf{g}$ | 任务文本嵌入 |
| $r_i^{\text{sem}}$ | Patch $i$ 的语义相关性得分 |
| $p_i$ | 节点 $i$ 预测任务进度 |
| $\ell$ | 任务语言描述 |
| $a_t$ | 时刻 $t$ 输出动作 |
| $\mathcal{T}_k$ | 子树 $k$ |
| $\mathcal{A}(v_t)$ | 节点 $v_t$ 的祖先集合 |

