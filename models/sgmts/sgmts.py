"""
SGMTS — Semantic Graph-guided Minimum spanning Tree Scan
CONSTRUCTION.md Section 4.

Pipeline for a single frame (H×W RGB image):
  1. Patch CNN → patch features F ∈ R^{P × d_f}
  2. Language semantic scoring: r_i = σ(W_g g · f_i / √d_f)
  3. Build MST over a 4/8-connected patch grid using semantic edge weights:
        w_ij = (1 − r_i)(1 − r_j)(−cos_ij) + ε            (CONSTRUCTION 4.3.2)
  4. Root = argmax r_i^sem
  5. BFS from root → ordered sequence
  6. Tree-SSM with semantic-adaptive input and Δ:
        x_i = W_in·f_i + r_i^sem · W_g'·g                 (CONSTRUCTION 4.3.4)
        Δ_i = softplus(W_Δ x_i) · (1 + β r_i)             (CONSTRUCTION 4.3.5)
  7. Output: Z^V ∈ R^{P × d_visual}

All MST computation uses CPU union-find; no CUDA extension required.
"""
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ========================================================
#  Patch feature extractor (lightweight CNN)
# ========================================================

class PatchCNN(nn.Module):
    """
    Splits image into non-overlapping patches and embeds each with a
    small CNN.  Mimics ViT patch embedding but keeps a receptive field
    that respects local texture inductive bias.
    """

    def __init__(self, patch_size: int = 16, d_f: int = 256, in_channels: int = 3):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, 64,  kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),   # pool each patch to a scalar map
        )
        # After pool: (B*P, 128, 1, 1) → flatten → (B*P, 128)
        self.fc = nn.Linear(128, d_f)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """
        x : (B, C, H, W)
        Returns:
            feats : (B, P, d_f)   P = (H/patch_size) * (W/patch_size)
            nH    : int number of patch rows
            nW    : int number of patch columns
        """
        B, C, H, W = x.shape
        ps = self.patch_size
        nH, nW = H // ps, W // ps

        # Unfold into patches: (B, C, nH, nW, ps, ps)
        patches = x.unfold(2, ps, ps).unfold(3, ps, ps)        # (B,C,nH,nW,ps,ps)
        patches = patches.contiguous().view(B * nH * nW, C, ps, ps)

        feats = self.proj(patches).view(B * nH * nW, -1)       # (B*P, 128)
        feats = self.fc(feats).view(B, nH * nW, -1)            # (B, P, d_f)
        return feats, nH, nW


# ========================================================
#  Pure-PyTorch Kruskal MST (CPU)
# ========================================================

def _kruskal_mst(edge_src: torch.Tensor, edge_dst: torch.Tensor,
                 edge_w: torch.Tensor, num_nodes: int):
    """
    Kruskal minimum spanning tree via union-find.
    All inputs on CPU.
    Returns list of selected edge indices (len = num_nodes - 1).
    """
    sorted_idx = torch.argsort(edge_w).tolist()
    parent = list(range(num_nodes))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return False
        parent[rx] = ry
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


# ========================================================
#  SGMTS Encoder
# ========================================================

class SGMTSEncoder(nn.Module):
    """
    Encodes a batch of frames via Semantic Guided MST Scan.

    Args:
        d_f        : patch CNN output dim
        d_lang     : Qwen2.5 hidden dim (for language gate vector g)
        d_visual   : output dim for each patch token (= d in main model)
        patch_size : pixels per patch side
        d_state    : SSM hidden state dim
        beta       : scale for semantic-adaptive delta boost
        connectivity: 4 or 8 neighbor connectivity
    """

    def __init__(
        self,
        d_f: int = 256,
        d_lang: int = 896,
        d_visual: int = 256,
        patch_size: int = 16,
        d_state: int = 16,
        beta: float = 2.0,
        connectivity: int = 4,
    ):
        super().__init__()
        self.d_f      = d_f
        self.d_state  = d_state
        self.beta     = beta
        self.connectivity = connectivity

        # Patch extractor
        self.patch_cnn = PatchCNN(patch_size=patch_size, d_f=d_f)

        # Language gate (for semantic scoring r_sem)
        self.W_gate   = nn.Linear(d_lang, d_f, bias=False)
        # Separate projection for SSM input conditioning (Section 4.3.4)
        self.W_g_prime = nn.Linear(d_lang, d_f, bias=False)

        # SSM parameters
        A_init = torch.arange(1, d_state + 1, dtype=torch.float).unsqueeze(0).expand(d_f, -1)
        self.A_log  = nn.Parameter(torch.log(A_init.clone()))   # (d_f, d_state)
        self.D      = nn.Parameter(torch.ones(d_f))

        self.B_proj = nn.Linear(d_f, d_state)
        self.C_proj = nn.Linear(d_f, d_state)
        self.W_delta = nn.Linear(d_f, d_f, bias=True)

        # Output projection
        self.out_proj = nn.Linear(d_f, d_visual)
        self.out_norm = nn.LayerNorm(d_visual)

        self._init_weights()

    def _init_weights(self):
        dt_min, dt_max = 0.001, 0.1
        dt = torch.exp(
            torch.rand(self.d_f) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.W_delta.bias.copy_(inv_dt)

    # -------------------------------------------------------------- #

    def forward(
        self,
        images: torch.Tensor,          # (B, C, H, W)
        lang_g: torch.Tensor,          # (B, d_lang) — CLS/mean-pool of LLM hidden
    ) -> torch.Tensor:
        """Returns Z_V : (B, P, d_visual)"""
        B = images.shape[0]
        device = images.device

        feats, nH, nW = self.patch_cnn(images)   # (B, P, d_f)
        P = nH * nW

        # ── Language semantic scores ─────────────────────────────────
        g_proj = self.W_gate(lang_g)             # (B, d_f)
        r_sem  = torch.sigmoid(
            torch.einsum("bd,bpd->bp", g_proj, feats) / math.sqrt(self.d_f)
        )                                         # (B, P)

        # ── Build MST per sample ─────────────────────────────────────
        all_Y = []
        for b in range(B):
            f_b  = feats[b]          # (P, d_f)
            r_b  = r_sem[b]          # (P,)
            Y_b  = self._scan_one(f_b, r_b, nH, nW, device, lang_g[b])   # (P, d_visual)
            all_Y.append(Y_b)

        return torch.stack(all_Y, dim=0)   # (B, P, d_visual)

    # -------------------------------------------------------------- #
    #  Per-sample MST + tree SSM scan                                  #
    # -------------------------------------------------------------- #

    def _scan_one(
        self,
        f: torch.Tensor,        # (P, d_f)
        r: torch.Tensor,        # (P,)   semantic scores
        nH: int, nW: int,
        device: torch.device,
        lang_g_b: torch.Tensor, # (d_lang,)  language gate vector for this sample
    ) -> torch.Tensor:
        P = nH * nW

        # Build edge list (4- or 8-connected grid)
        src_list, dst_list = [], []
        for i in range(nH):
            for j in range(nW):
                u = i * nW + j
                neighbors = []
                if j + 1 < nW:  neighbors.append(i * nW + (j + 1))       # right
                if i + 1 < nH:  neighbors.append((i + 1) * nW + j)        # down
                if self.connectivity == 8:
                    if i + 1 < nH and j + 1 < nW: neighbors.append((i+1)*nW+(j+1))
                    if i + 1 < nH and j - 1 >= 0: neighbors.append((i+1)*nW+(j-1))
                for v in neighbors:
                    src_list.append(u)
                    dst_list.append(v)

        edge_src = torch.tensor(src_list, dtype=torch.long)
        edge_dst = torch.tensor(dst_list, dtype=torch.long)

        # Semantic edge weights (CPU)
        f_cpu = f.detach().cpu().float()
        r_cpu = r.detach().cpu().float()
        f_norm = F.normalize(f_cpu, dim=1)
        cos_ij = (f_norm[edge_src] * f_norm[edge_dst]).sum(dim=1)
        r_u    = r_cpu[edge_src]
        r_v    = r_cpu[edge_dst]
        # CONSTRUCTION Section 4.3.2: w_ij = (1-r_i)(1-r_j) * (-cos_ij) + ε
        w_ij   = (1 - r_u) * (1 - r_v) * (-cos_ij) + 1e-6

        # Kruskal MST
        mst_idx = _kruskal_mst(edge_src, edge_dst, w_ij, P)

        # Build adjacency list (undirected)
        adj = [[] for _ in range(P)]
        for idx in mst_idx:
            u, v = int(src_list[idx]), int(dst_list[idx])
            adj[u].append(v)
            adj[v].append(u)

        # Root = patch with highest semantic score
        root = int(r_cpu.argmax().item())

        # BFS ordering
        bfs_order, visited, parent_of = [], [False] * P, [-1] * P
        queue = [root]
        visited[root] = True
        while queue:
            node = queue.pop(0)
            bfs_order.append(node)
            for nb in adj[node]:
                if not visited[nb]:
                    visited[nb] = True
                    parent_of[nb] = node
                    queue.append(nb)

        # ── Tree-SSM scan ────────────────────────────────────────────
        A = -torch.exp(self.A_log.float()).to(device)   # (d_f, d_state)

        bfs_t     = torch.tensor(bfs_order, dtype=torch.long, device=device)
        r_sorted  = r[bfs_t]                            # (P,)

        # CONSTRUCTION Section 4.3.4: x_i = W_in·f_i + r_i^sem · W_g'·g
        g_prime = self.W_g_prime(lang_g_b.to(device))   # (d_f,)
        X = f[bfs_t] + r_sorted.unsqueeze(1) * g_prime.unsqueeze(0)  # (P, d_f)

        delta     = F.softplus(self.W_delta(X))         # (P, d_f)
        delta     = delta * (1.0 + self.beta * r_sorted.unsqueeze(1))  # semantic boost

        B_proj    = self.B_proj(X)                      # (P, d_state)
        C_proj    = self.C_proj(X)                      # (P, d_state)

        # Map original → BFS index
        orig2bfs = [0] * P
        for bfs_i, orig_i in enumerate(bfs_order):
            orig2bfs[orig_i] = bfs_i

        H = X.new_zeros(P, self.d_f, self.d_state)
        Y = X.new_zeros(P, self.d_f)

        for bfs_i in range(P):
            d_i    = delta[bfs_i]                               # (d_f,)
            A_bar  = torch.exp(d_i.unsqueeze(1) * A)            # (d_f, d_state)
            B_bar  = d_i.unsqueeze(1) * B_proj[bfs_i].unsqueeze(0)  # (d_f, d_state)

            par_orig = parent_of[bfs_order[bfs_i]]
            h_par    = H[orig2bfs[par_orig]] if par_orig != -1 else H.new_zeros(self.d_f, self.d_state)

            h_i      = A_bar * h_par + B_bar * X[bfs_i].unsqueeze(1)
            H[bfs_i] = h_i
            Y[bfs_i] = (h_i * C_proj[bfs_i].unsqueeze(0)).sum(dim=1) + self.D * X[bfs_i]

        # Reorder Y back to raster order (i.e., patch 0..P-1)
        Y_raster = Y.new_zeros(P, self.d_f)
        for bfs_i, orig_i in enumerate(bfs_order):
            Y_raster[orig_i] = Y[bfs_i]

        out = self.out_norm(self.out_proj(Y_raster))   # (P, d_visual)
        return out
