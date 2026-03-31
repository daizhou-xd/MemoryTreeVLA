"""
Tree-SSM Readout — CONSTRUCTION.md Section 3.5.

Traverses the memory tree in BFS order and applies a Mamba-style selective
state-space recurrence where each node's hidden state propagates from its
parent (instead of the sequence predecessor).
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tree import HierarchicalMemoryTree


class TreeSSMReadout(nn.Module):
    """
    Input  : HierarchicalMemoryTree  (any size)
    Output : Z_M ∈ R^{N × d_ssm}   (BFS-ordered node features)

    Node input vector:
        x_i = W_in [z_v_i ; a_last_i ; q_i ; s_i ; log(w_i)] + b_in

    Weight-adaptive time step:
        Δ_i = softplus(W_Δ x_i + b_Δ) ⊙ σ(W_w log(w_i) + b_w)

    Tree SSM recurrence (ZOH discretisation):
        Ā_i = exp(Δ_i · A)
        B̄_i = Δ_i · B(x_i)          (simplified Euler approx.)
        h_i = Ā_i ⊙ h_par(i) + B̄_i ⊙ x_i
        y_i = C(x_i) h_i + D x_i
    """

    def __init__(
        self,
        d_node: int,   # visual / semantic embedding dim  (= d)
        d_a: int,      # action dim
        d_q: int,      # joint-state dim
        d_ssm: int,    # SSM inner dim
        d_state: int = 16,
        max_depth: Optional[int] = None,
    ):
        super().__init__()
        self.d_ssm  = d_ssm
        self.d_state = d_state
        self.max_depth = max_depth

        # ── Input projection ─────────────────────────────────────────
        d_in = d_node + d_a + d_q + d_node + 1   # z_v, a, q, s, log_w
        self.in_proj = nn.Linear(d_in, d_ssm)

        # ── Weight-adaptive delta ────────────────────────────────────
        self.W_delta = nn.Linear(d_ssm, d_ssm, bias=True)
        self.W_w     = nn.Linear(1, d_ssm, bias=True)

        # ── SSM parameters (S4D real init) ───────────────────────────
        A_init = torch.arange(1, d_state + 1, dtype=torch.float).unsqueeze(0).expand(d_ssm, -1)
        self.A_log   = nn.Parameter(torch.log(A_init.clone()))   # (d_ssm, d_state)
        self.D       = nn.Parameter(torch.ones(d_ssm))

        # ── Selective B, C projections ───────────────────────────────
        self.B_proj  = nn.Linear(d_ssm, d_state)
        self.C_proj  = nn.Linear(d_ssm, d_state)

        # ── Output norm ──────────────────────────────────────────────
        self.out_norm = nn.LayerNorm(d_ssm)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.zeros_(self.in_proj.bias)
        nn.init.zeros_(self.W_delta.bias)
        # Initialize delta bias so softplus(bias) lands in [dt_min, dt_max]
        dt_min, dt_max = 0.001, 0.1
        dt = torch.exp(
            torch.rand(self.d_ssm) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.W_delta.bias.copy_(inv_dt)

    # ------------------------------------------------------------------ #

    def forward(
        self,
        tree: HierarchicalMemoryTree,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Returns Z_M of shape (N, d_ssm), where N = tree.size().
        If max_depth is set, only returns nodes within that depth.
        """
        bfs_ids = tree.bfs_order()
        if not bfs_ids:
            d = next(self.parameters()).device if device is None else device
            return torch.zeros(1, self.d_ssm, device=d)

        device = next(self.parameters()).device

        # ── Build input tensor X ─────────────────────────────────────
        rows = []
        for nid in bfs_ids:
            node = tree.nodes[nid]
            z_v   = node.z_v.to(device).float()
            a     = node.a_last.to(device).float()
            q     = node.q.to(device).float()
            s     = node.s.to(device).float()
            log_w = torch.log(torch.tensor([node.w + 1e-6], device=device))
            rows.append(torch.cat([z_v, a, q, s, log_w], dim=0))

        X     = torch.stack(rows, dim=0)          # (N, d_in)
        X_p   = self.in_proj(X)                   # (N, d_ssm)

        # ── Time steps ───────────────────────────────────────────────
        log_ws = X[:, -1:]                        # (N, 1) — already log_w
        delta = F.softplus(self.W_delta(X_p)) * torch.sigmoid(self.W_w(log_ws))
        # delta: (N, d_ssm)

        # ── SSM (A negative real for stability, S4D) ─────────────────
        A = -torch.exp(self.A_log.float())        # (d_ssm, d_state)
        B = self.B_proj(X_p)                      # (N, d_state)
        C = self.C_proj(X_p)                      # (N, d_state)

        # ── BFS-order tree recurrence ────────────────────────────────
        N = len(bfs_ids)
        node2idx = {nid: i for i, nid in enumerate(bfs_ids)}
        H = X_p.new_zeros(N, self.d_ssm, self.d_state)   # hidden states
        Y = X_p.new_zeros(N, self.d_ssm)

        for i, nid in enumerate(bfs_ids):
            d_i = delta[i]                                # (d_ssm,)
            # Discretise (ZOH simplified: Ā = exp(Δ A), B̄ ≈ Δ B(x))
            A_bar = torch.exp(d_i.unsqueeze(1) * A)       # (d_ssm, d_state)
            B_bar = d_i.unsqueeze(1) * B[i].unsqueeze(0)  # (d_ssm, d_state)

            par_id = tree.nodes[nid].parent_id
            h_par  = H[node2idx[par_id]] if par_id is not None else H.new_zeros(self.d_ssm, self.d_state)

            h_i    = A_bar * h_par + B_bar * X_p[i].unsqueeze(1)  # (d_ssm, d_state)
            H[i]   = h_i

            # y_i = C(x_i) · h_i + D · x_i
            y_i    = (h_i * C[i].unsqueeze(0)).sum(dim=1) + self.D * X_p[i]
            Y[i]   = y_i

        Y = self.out_norm(Y)

        # ── Optional depth filtering ──────────────────────────────────
        if self.max_depth is not None:
            mask = [tree.depth(nid) <= self.max_depth for nid in bfs_ids]
            mask_t = torch.tensor(mask, dtype=torch.bool, device=device)
            Y = Y[mask_t]

        return Y   # (N_kept, d_ssm)
