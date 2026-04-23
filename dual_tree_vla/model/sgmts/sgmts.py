"""
SGMTS — Semantic-Guided Mamba Tree Scan (CONSTRUCTION.md §4.1)

新设计（旁路增强模式）：
  不再使用独立的 CLIP Vision Encoder。
  直接接收 VLA 骨架 ViT 最后一层输出的 patch 特征 P_t（通过 forward hook 提取），
  输出语义增强视觉特征 Z_v，维度与输入相同（d_vit）。

  SGMTS 对骨架 ViT 是只读的，不修改任何骨架参数。

数据流:
  P_t  (B, N_p, d_vit)  ← 骨架 ViT 最后一层 patch 特征 (CLS 已剔除)
  g_task (B, d_vit)     ← 骨架 LLM 对任务描述的均值嵌入 (冻结)
  s_top  list[B]        ← HMT 顶层抽象节点语义均值 (可为 None)
      ↓
  g_sem = β·g_task + (1-β)·s_top
      ↓
  σ_i = cos(p_i, W_g · g_sem)   — 语义重要性图
  r*  = argmax σ_i               — 动态语义根
  w_ij = cos(p_i, p_j) + α·σ_i·σ_j  — 语义加权 MST 边权
      ↓
  [Kruskal 最大生成树 → BFS 扫描序]
      ↓
  X_i = p_i + σ_i · W_g' · g_sem  — 语义增强输入
  Tree-SSM (ZOH, 父节点传播)
      ↓
  Z_v (B, N_p, d_vit)
"""
from __future__ import annotations

import math
from collections import deque
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def _kruskal_mst_max(edge_src, edge_dst, edge_w, num_nodes):
    """Kruskal MAX spanning tree. Returns selected edge indices."""
    sorted_idx = torch.argsort(edge_w, descending=True).tolist()
    parent = list(range(num_nodes))
    rank   = [0] * num_nodes

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return False
        if rank[rx] < rank[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        if rank[rx] == rank[ry]:
            rank[rx] += 1
        return True

    mst_edges = []
    src_list = edge_src.tolist()
    dst_list = edge_dst.tolist()
    for idx in sorted_idx:
        if union(int(src_list[idx]), int(dst_list[idx])):
            mst_edges.append(idx)
        if len(mst_edges) == num_nodes - 1:
            break
    return mst_edges


class SGMTS(nn.Module):
    """
    Semantic-Guided Mamba Tree Scan (CONSTRUCTION.md §4.1)

    直接接收骨架 ViT 输出的 patch 特征，无独立视觉编码器。

    Args:
        d_vit        : 语言模型侧特征维度（InternVL3-1B LLM 为 896）
        d_patch      : ViT 原始输出维度（InternVL3-1B ViT 为 1024）。
                       若为 None 则默认与 d_vit 相同（向后兼容）。
        d_state      : SSM 隐状态维度
        alpha        : 语义偏置权重（MST 边权中的语义项系数，默认 0.5）
        connectivity : MST 4-邻域或 8-邻域
    """

    def __init__(
        self,
        d_vit: int = 896,
        d_patch: Optional[int] = None,
        d_state: int = 16,
        alpha: float = 0.5,
        connectivity: int = 4,
    ):
        super().__init__()
        # d_patch = ViT 原始输出维度（可能与 LLM 不同）
        d_patch = d_patch if d_patch is not None else d_vit
        self.d_vit        = d_vit
        self.d_patch      = d_patch
        self.d_state      = d_state
        self.alpha        = alpha
        self.connectivity = connectivity

        # W_g: g_sem (d_vit) → patch 空间 (d_patch)，用于语义相似度计算
        self.lang_gate = nn.Linear(d_vit, d_patch, bias=False)
        # W_g': g_sem (d_vit) → patch 空间 (d_patch)，注入 SSM 输入
        self.W_g_prime = nn.Linear(d_vit, d_patch, bias=False)

        # Tree-SSM 参数（全部在 d_patch 空间运算）
        A_init = torch.arange(1, d_state + 1, dtype=torch.float) \
                      .unsqueeze(0).expand(d_patch, -1)
        self.A_log   = nn.Parameter(torch.log(A_init.clone()))
        self.D       = nn.Parameter(torch.ones(d_patch))
        self.B_proj  = nn.Linear(d_patch, d_state)
        self.C_proj  = nn.Linear(d_patch, d_state)
        self.W_delta = nn.Linear(d_patch, d_patch, bias=True)

        self.out_norm = nn.LayerNorm(d_patch)

        self._grid_edge_cache: dict = {}
        self._init_weights()

    def _init_weights(self):
        dt_min, dt_max = 0.001, 0.1
        dt = torch.exp(
            torch.rand(self.d_patch) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.W_delta.bias.copy_(inv_dt)

    def _get_grid_edges(self, nH: int, nW: int) -> tuple:
        key = (nH, nW, self.connectivity)
        if key not in self._grid_edge_cache:
            src_list, dst_list = [], []
            for i in range(nH):
                for j in range(nW):
                    u = i * nW + j
                    if j + 1 < nW:
                        src_list.append(u); dst_list.append(i * nW + (j + 1))
                    if i + 1 < nH:
                        src_list.append(u); dst_list.append((i + 1) * nW + j)
                    if self.connectivity == 8:
                        if i + 1 < nH and j + 1 < nW:
                            src_list.append(u); dst_list.append((i + 1) * nW + (j + 1))
                        if i + 1 < nH and j - 1 >= 0:
                            src_list.append(u); dst_list.append((i + 1) * nW + (j - 1))
            self._grid_edge_cache[key] = (
                torch.tensor(src_list, dtype=torch.long),
                torch.tensor(dst_list, dtype=torch.long),
                src_list,
                dst_list,
            )
        return self._grid_edge_cache[key]

    def forward(
        self,
        P_t: torch.Tensor,
        g_task: torch.Tensor,
        s_top: Optional[List[Optional[torch.Tensor]]] = None,
        beta: Optional[List[float]] = None,
        return_attn: bool = False,
    ) -> Union[torch.Tensor, tuple]:
        """
        Args:
            P_t    : (B, N_p, d_vit)  骨架 ViT patch 特征 (CLS 已剔除)
            g_task : (B, d_vit)       骨架 LLM 任务描述均值嵌入 (冻结)
            s_top  : list[B], 每项 (d_vit,) 或 None，HMT 顶层抽象节点均值
            beta   : list[B] floats in [0.3, 1.0]
            return_attn : 若为 True，额外返回语义重要性分数 σ list (B 个 (N_p,) tensor)
        Returns:
            Z_v    : (B, N_p, d_vit)
            sigma_maps (只在 return_attn=True 时) : List[Tensor(N_p,)]，每帧的语义重要性分数
        """
        B, N_p, _ = P_t.shape
        device = P_t.device

        nH = nW = int(math.isqrt(N_p))
        if nH * nW != N_p:
            nH, nW = 1, N_p

        g_sem_list = []
        for b in range(B):
            b_val = 1.0 if (beta is None) else float(beta[b])
            g_t   = g_task[b]
            if s_top is not None and s_top[b] is not None:
                s_b     = s_top[b].to(device=device, dtype=g_t.dtype)
                g_sem_b = b_val * g_t + (1.0 - b_val) * s_b
            else:
                g_sem_b = g_t
            g_sem_list.append(g_sem_b)
        g_sem = torch.stack(g_sem_list, dim=0)

        results = [
            self._scan_one(P_t[b], g_sem[b], nH, nW, device, return_sigma=return_attn)
            for b in range(B)
        ]
        if return_attn:
            Z_v_list     = [r[0] for r in results]
            sigma_list   = [r[1] for r in results]
            return torch.stack(Z_v_list, dim=0), sigma_list
        return torch.stack(results, dim=0)

    def _scan_one(
        self,
        f: torch.Tensor,
        g_sem: torch.Tensor,
        nH: int,
        nW: int,
        device: torch.device,
        return_sigma: bool = False,
    ) -> Union[torch.Tensor, tuple]:
        P = nH * nW
        edge_src, edge_dst, src_list, dst_list = self._get_grid_edges(nH, nW)

        # lang_gate 投影必须保留梯度——sem_score 通过 sem_bfs → X → SSM 输出 → loss
        # 反向传播到 lang_gate.weight。Kruskal MST 本身不可微，但 sem_bfs 作为
        # 连续权重嵌入到 X 中，梯度路径完整。
        g_gate_cpu = self.lang_gate(g_sem).cpu().float()  # 不能 detach！
        f_cpu      = f.detach().cpu().float()  # ViT 特征冻结，只用于 MST 构建
        f_norm     = F.normalize(f_cpu, dim=1)
        g_norm     = F.normalize(g_gate_cpu.unsqueeze(0), dim=1)

        cos_patch = (f_norm[edge_src] * f_norm[edge_dst]).sum(dim=1)
        sem_score = (f_norm @ g_norm.T).squeeze(-1)
        w_ij = cos_patch + self.alpha * sem_score[edge_src] * sem_score[edge_dst]

        mst_idx = _kruskal_mst_max(edge_src, edge_dst, w_ij, P)

        adj = [[] for _ in range(P)]
        for idx in mst_idx:
            u, v = int(src_list[idx]), int(dst_list[idx])
            adj[u].append(v); adj[v].append(u)

        root      = int(sem_score.argmax().item())
        bfs_order = []
        parent_of = [-1] * P
        level_of  = [0]  * P
        visited   = [False] * P
        q = deque([root])
        visited[root] = True
        while q:
            node = q.popleft()
            bfs_order.append(node)
            lv = level_of[node]
            for nb in adj[node]:
                if not visited[nb]:
                    visited[nb]   = True
                    parent_of[nb] = node
                    level_of[nb]  = lv + 1
                    q.append(nb)

        A = -torch.exp(self.A_log.float()).to(device)
        bfs_t_cpu = torch.tensor(bfs_order, dtype=torch.long)
        bfs_t     = bfs_t_cpu.to(device)
        f_sorted  = f[bfs_t]

        g_prime  = self.W_g_prime(g_sem.to(device))
        sem_bfs  = sem_score[bfs_t_cpu].to(device)
        X = (f_sorted + sem_bfs.unsqueeze(1) * g_prime.unsqueeze(0)).to(f.dtype)

        delta     = F.softplus(self.W_delta(X))
        B_proj    = self.B_proj(X)
        C_proj    = self.C_proj(X)
        A_bar_all = torch.exp(delta.unsqueeze(2) * A.unsqueeze(0))
        Bx_all    = (delta * X).unsqueeze(2) * B_proj.unsqueeze(1)

        orig2bfs = [0] * P
        for bfs_i, orig_i in enumerate(bfs_order):
            orig2bfs[orig_i] = bfs_i
        parent_bfs = [
            orig2bfs[parent_of[bfs_order[i]]] if parent_of[bfs_order[i]] != -1 else -1
            for i in range(P)
        ]

        level_bfs = [level_of[bfs_order[i]] for i in range(P)]
        max_lv    = max(level_bfs)
        lv_groups: list = [[] for _ in range(max_lv + 1)]
        for bfs_i, lv in enumerate(level_bfs):
            lv_groups[lv].append(bfs_i)

        H = X.new_zeros(P, self.d_patch, self.d_state)
        for lv_nodes in lv_groups:
            idx   = torch.tensor(lv_nodes, dtype=torch.long, device=device)
            p_idx = torch.tensor([parent_bfs[i] for i in lv_nodes],
                                  dtype=torch.long, device=device)
            mask  = p_idx >= 0
            h_par = H.new_zeros(len(lv_nodes), self.d_patch, self.d_state)
            if mask.any():
                h_par[mask] = H[p_idx[mask]]
            H[idx] = A_bar_all[idx] * h_par + Bx_all[idx]

        Y = (H * C_proj.unsqueeze(1)).sum(dim=2) + self.D.to(X.dtype) * X

        Y_raster = Y.new_zeros(P, self.d_patch)
        Y_raster[bfs_t] = Y

        out = self.out_norm(Y_raster)
        if return_sigma:
            # sem_score 是 CPU float，转为 GPU tensor 返回
            return out, sem_score.to(device=device, dtype=out.dtype)
        return out


# 向后兼容别名
SGMTSEncoder = SGMTS
