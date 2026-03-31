from dataclasses import dataclass, field
from typing import List, Optional

import torch


@dataclass
class MemoryNode:
    """
    Six-tuple memory node as defined in CONSTRUCTION.md Section 3.2.
    v_i = (z_v, A, q, s, n, w)
    """
    node_id: int

    # ── Core fields ───────────────────────────────────────────────────
    z_v: torch.Tensor          # (d,)   visual embedding online mean
    A:   List[torch.Tensor]    # list of (d_a,) action tensors, ordered
    q:   torch.Tensor          # (d_q,) latest joint state (rolling update)
    s:   torch.Tensor          # (d,)   semantic embedding online mean
    n:   int   = 1             # merge count (= access frequency n_access)
    w:   float = 1.0           # importance weight

    # ── Tree structure ─────────────────────────────────────────────────
    parent_id:    Optional[int]  = None
    children_ids: List[int]      = field(default_factory=list)

    # ------------------------------------------------------------------

    @property
    def a_last(self) -> torch.Tensor:
        """Representative action for SSM readout (latest in A)."""
        return self.A[-1] if self.A else torch.zeros_like(self.q)

    def is_leaf(self) -> bool:
        return len(self.children_ids) == 0

    def is_root(self) -> bool:
        return self.parent_id is None
