"""
Tree operations: Reinforcement, Semantic Elevation, Pruning.
CONSTRUCTION.md Section 3.4
"""
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .node import MemoryNode
from .tree import HierarchicalMemoryTree


# ======================================================================= #
#  Operation ① — Memory Reinforcement                                      #
# ======================================================================= #

def reinforce(
    tree: HierarchicalMemoryTree,
    grad_norms: Dict[int, float],   # {node_id: ||∇_{Φ_C} L||_2}
    eta: float = 0.01,
    theta_grad: float = 0.1,
    alpha_ema: float = 0.05,
    tau_task: float = 0.1,
    phi_query: Optional[torch.Tensor] = None,
    phi_updates: Optional[Dict[int, torch.Tensor]] = None,
):
    """
    (a) Gradient-driven weight update:
        w_C ← w_C + η · ‖∇Φ_C L‖₂ · 1[‖∇‖ > θ_grad]
    (b) EMA + task-weighted representation update (if phi_updates provided):
        Φ_C ← (1-α)Φ_C + α · Σ w_i^task Φ_i / Σ w_i^task
    """
    # (a) gradient-driven weight update
    for node_id, grad_norm in grad_norms.items():
        if node_id not in tree.nodes:
            continue
        if grad_norm > theta_grad:
            tree.nodes[node_id].w += eta * grad_norm

    # (b) EMA task-weighted representation update
    if phi_updates is None or phi_query is None:
        return

    # compute task-relevance weights for all candidate nodes
    node_ids = list(phi_updates.keys())
    phis = torch.stack([phi_updates[nid] for nid in node_ids], dim=0)  # (N, d)
    phi_query_n = F.normalize(phi_query.float().unsqueeze(0), dim=-1)   # (1, d)
    phis_n      = F.normalize(phis.float(), dim=-1)                      # (N, d)
    sims = (phis_n @ phi_query_n.T).squeeze(-1)                          # (N,)
    w_task = torch.softmax(sims / tau_task, dim=0)                        # (N,)

    # weighted mean of new representations
    phi_mean = (w_task.unsqueeze(1) * phis).sum(0)                        # (d,)

    for node_id in node_ids:
        if node_id not in tree.nodes:
            continue
        node = tree.nodes[node_id]
        node.s = ((1 - alpha_ema) * node.s.float() + alpha_ema * phi_mean).to(node.s.dtype)


# ======================================================================= #
#  Operation ② — Semantic Elevation                                        #
# ======================================================================= #

class MLPElevation(nn.Module):
    """
    Small MLP that maps weighted-pooled [z_v; s] → s_abs for a new abstract
    parent node v_abs.  Section 3.4 Operation ②.
    """

    def __init__(self, d: int, hidden: int = None):
        super().__init__()
        hidden = hidden or d * 2
        self.net = nn.Sequential(
            nn.Linear(d * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, d),
            nn.LayerNorm(d),
        )

    def forward(self, z_pool: torch.Tensor, s_pool: torch.Tensor) -> torch.Tensor:
        """z_pool, s_pool: (d,) each → s_abs: (d,)"""
        return self.net(torch.cat([z_pool, s_pool], dim=-1))


def semantic_elevation(
    tree: HierarchicalMemoryTree,
    parent_id: int,
    mlp_elev: MLPElevation,
    device: torch.device = torch.device("cpu"),
) -> Optional[int]:
    """
    Trigger semantic elevation on v_p = tree.nodes[parent_id].

    Selects the top-⌊K/2⌋ children by importance weight as group G,
    creates v_abs between v_p and G.  Section 3.4 Operation ②.

    Returns the new v_abs node_id, or None if elevation was skipped.
    """
    v_p = tree.nodes[parent_id]
    children = v_p.children_ids

    if len(children) < 2:
        return None

    K = len(children)
    group_size = max(2, K // 2)

    # Sort children by weight, select top group_size
    sorted_children = sorted(children, key=lambda nid: tree.nodes[nid].w, reverse=True)
    G = sorted_children[:group_size]

    # Weighted pool z_v and s over G
    ws = torch.tensor([tree.nodes[nid].w for nid in G], dtype=torch.float, device=device)
    ws = ws / ws.sum()

    z_pool = sum(ws[i] * tree.nodes[nid].z_v.to(device) for i, nid in enumerate(G))
    s_pool = sum(ws[i] * tree.nodes[nid].s.to(device)   for i, nid in enumerate(G))

    with torch.no_grad():
        s_abs = mlp_elev(z_pool.float(), s_pool.float())

    # Representative z_v and q from highest-weight node in G
    top_nid = G[0]
    z_abs = tree.nodes[top_nid].z_v.clone()
    q_abs = tree.nodes[top_nid].q.clone()
    w_abs = sum(tree.nodes[nid].w for nid in G)
    n_abs = sum(tree.nodes[nid].n for nid in G)

    # Create v_abs
    abs_id = tree.alloc_id()
    v_abs = MemoryNode(
        node_id=abs_id,
        z_v=z_abs,
        A=[tree.nodes[top_nid].a_last.clone()],
        q=q_abs,
        s=s_abs.detach().cpu(),
        n=n_abs,
        w=w_abs,
        parent_id=parent_id,
        children_ids=list(G),
    )
    tree.add_node(v_abs)

    # Re-wire: v_p's children list replaces G with v_abs
    v_p.children_ids = [nid for nid in v_p.children_ids if nid not in G]
    v_p.children_ids.append(abs_id)

    # Update parent pointers of G members
    for nid in G:
        tree.nodes[nid].parent_id = abs_id

    return abs_id


# ======================================================================= #
#  Operation ③ — Pruning                                                   #
# ======================================================================= #

def prune(
    tree: HierarchicalMemoryTree,
    theta_w: float = 0.3,
) -> List[int]:
    """
    Remove leaf nodes with w_i < theta_w.  Section 3.4 Operation ③.
    Iterates until no more leaves can be pruned (cascading effect).

    Returns list of pruned node_ids.
    """
    pruned: List[int] = []

    changed = True
    while changed:
        changed = False
        for node_id in list(tree.nodes.keys()):
            node = tree.nodes[node_id]
            if node.is_leaf() and node.w < theta_w and not node.is_root():
                # Remove from parent's children list
                par = tree.nodes[node.parent_id]
                par.children_ids.remove(node_id)
                # If active node was pruned, fall back to its parent
                if tree.active_id == node_id:
                    tree.active_id = node.parent_id
                del tree.nodes[node_id]
                pruned.append(node_id)
                changed = True

    return pruned
