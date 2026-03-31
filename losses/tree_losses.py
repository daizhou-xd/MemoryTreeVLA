"""
Tree-aware auxiliary losses — CONSTRUCTION.md Section 6.

L_recon  : node visual reconstruction
           MSE( Decoder(z_v_i), frame_patch_mean_i )

L_sem    : language–visual semantic alignment
           InfoNCE between s_i and subtask language embed g_i

L_prog   : progression ordering loss
           For ancestor–descendant pairs (v_i, v_j):
               margin hinge: d(s_i, q_task) + γ < d(s_j, q_task)
           (i.e., descendants should be *closer* to task goal)

L_elev   : elevation consistency loss
           s_abs ≈ pooled(children s_j)
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================================================================
#  Node visual reconstruction loss
# ==================================================================

class NodeReconDecoder(nn.Module):
    """Simple MLP decoder: z_v (d,) → d_patch (mean patch reconstruction)."""

    def __init__(self, d: int, d_patch: int = 256 * 3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d, d * 2),
            nn.GELU(),
            nn.Linear(d * 2, d_patch),
        )

    def forward(self, z_v: torch.Tensor) -> torch.Tensor:
        return self.mlp(z_v)


def l_recon(
    decoder: NodeReconDecoder,
    z_v_batch: torch.Tensor,       # (N, d)  — node visual embeddings
    patch_targets: torch.Tensor,   # (N, d_patch) — target mean-patch pixels
) -> torch.Tensor:
    """MSE reconstruction loss between decoded z_v and mean patch target."""
    recon = decoder(z_v_batch)
    return F.mse_loss(recon, patch_targets)


# ==================================================================
#  Semantic alignment loss (InfoNCE)
# ==================================================================

def l_sem(
    s_nodes: torch.Tensor,     # (N, d)  — node semantic embeddings
    s_text: torch.Tensor,      # (N, d)  — corresponding subtask embeddings
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Symmetric InfoNCE / NT-Xent loss between visual node semantics and
    language subtask semantics.
    """
    s_n  = F.normalize(s_nodes, dim=-1)   # (N, d)
    s_t  = F.normalize(s_text,  dim=-1)   # (N, d)

    logits = (s_n @ s_t.T) / temperature   # (N, N)
    targets = torch.arange(logits.shape[0], device=logits.device)

    loss_i2t = F.cross_entropy(logits, targets)
    loss_t2i = F.cross_entropy(logits.T, targets)
    return (loss_i2t + loss_t2i) * 0.5


# ==================================================================
#  Progression ordering loss
# ==================================================================

def l_prog(
    s_nodes: dict,            # {node_id: s_tensor (d,)}
    pairs: List[tuple],       # [(anc_id, desc_id), ...]
    s_goal: torch.Tensor,     # (d,)  — mean of leaf-node semantics
    gamma: float = 0.1,
) -> torch.Tensor:
    """
    Margin loss: distance(ancestor, goal) > distance(descendant, goal) + γ
    i.e., as we go deeper in the tree we should be closer to the task goal.
    """
    if not pairs:
        return torch.tensor(0.0)

    loss = torch.tensor(0.0, device=s_goal.device)
    count = 0
    for anc_id, desc_id in pairs:
        if anc_id not in s_nodes or desc_id not in s_nodes:
            continue
        s_a  = F.normalize(s_nodes[anc_id].unsqueeze(0),  dim=-1)
        s_d  = F.normalize(s_nodes[desc_id].unsqueeze(0), dim=-1)
        s_g  = F.normalize(s_goal.unsqueeze(0),            dim=-1)

        # Distance = 1 – cosine similarity
        dist_a = 1.0 - (s_a * s_g).sum()
        dist_d = 1.0 - (s_d * s_g).sum()

        # Hinge: ancestor should be farther from goal than descendant
        loss = loss + F.relu(dist_d - dist_a + gamma)
        count += 1

    return loss / max(count, 1)


# ==================================================================
#  Elevation consistency loss
# ==================================================================

def l_elev(
    s_abs: torch.Tensor,          # (d,) — abstract node semantic
    s_children: List[torch.Tensor],
    w_children: List[float],
) -> torch.Tensor:
    """
    s_abs should match the weighted-mean of children semantics.
    MSE loss in normalised embedding space.
    """
    if not s_children:
        return torch.tensor(0.0)

    w_tensor = torch.tensor(w_children, device=s_abs.device)
    w_tensor = w_tensor / w_tensor.sum().clamp(min=1e-6)
    s_stack  = torch.stack(s_children, dim=0)               # (K, d)
    s_target = (w_tensor.unsqueeze(1) * s_stack).sum(0)     # (d,)

    return F.mse_loss(
        F.normalize(s_abs.unsqueeze(0), dim=-1),
        F.normalize(s_target.unsqueeze(0), dim=-1),
    )


# ==================================================================
#  Combined tree loss
# ==================================================================

def tree_loss(
    L_flow: torch.Tensor,
    L_recon: Optional[torch.Tensor] = None,
    L_sem: Optional[torch.Tensor] = None,
    L_prog: Optional[torch.Tensor] = None,
    L_elev_val: Optional[torch.Tensor] = None,
    w_flow: float = 1.0,
    w_recon: float = 0.5,
    w_sem: float = 0.5,
    w_prog: float = 0.3,
    w_elev: float = 0.2,
) -> torch.Tensor:
    total = w_flow * L_flow
    if L_recon is not None:
        total = total + w_recon * L_recon
    if L_sem is not None:
        total = total + w_sem * L_sem
    if L_prog is not None:
        total = total + w_prog * L_prog
    if L_elev_val is not None:
        total = total + w_elev * L_elev_val
    return total
