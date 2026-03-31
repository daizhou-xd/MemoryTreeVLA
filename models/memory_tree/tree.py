from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .node import MemoryNode


class HierarchicalMemoryTree:
    """
    Hierarchical Memory Tree (HMT) — CONSTRUCTION.md Section 3.

    Each call to `insert()` advances the tree by one timestep:
      Decision A (merge)  : d_t < theta_fuse  → Welford update active node
      Decision B (branch) : d_t >= theta_fuse → create new child node

    Tree operations (reinforce / elevation / prune) are delegated to
    models/memory_tree/operations.py and triggered externally.
    """

    def __init__(
        self,
        d: int,              # embedding dim
        d_a: int,            # action dim
        d_q: int,            # joint-state dim
        theta_fuse: float = 0.4,
        K_elev: int = 4,     # elevation trigger threshold
        delta_w: float = 0.1,
        tau: float = 0.1,    # soft-gating temperature (training)
    ):
        self.d = d
        self.d_a = d_a
        self.d_q = d_q
        self.theta_fuse = theta_fuse
        self.K_elev = K_elev
        self.delta_w = delta_w
        self.tau = tau

        self._nodes: Dict[int, MemoryNode] = {}
        self._next_id: int = 0
        self.root_id: Optional[int] = None
        self.active_id: Optional[int] = None
        # set by insert() when a branch triggers the elevation condition
        self.elevation_pending_parent: Optional[int] = None

    # ------------------------------------------------------------------ #
    #  Read-only helpers                                                   #
    # ------------------------------------------------------------------ #

    @property
    def nodes(self) -> Dict[int, MemoryNode]:
        return self._nodes

    def size(self) -> int:
        return len(self._nodes)

    def depth(self, node_id: int) -> int:
        return len(self.get_ancestors(node_id))

    def get_ancestors(self, node_id: int) -> List[int]:
        """Ordered from parent → root."""
        ancestors: List[int] = []
        current = self._nodes[node_id]
        while current.parent_id is not None:
            ancestors.append(current.parent_id)
            current = self._nodes[current.parent_id]
        return ancestors

    def bfs_order(self) -> List[int]:
        """Return all node ids in breadth-first order."""
        if self.root_id is None:
            return []
        result, queue = [], [self.root_id]
        while queue:
            nid = queue.pop(0)
            result.append(nid)
            queue.extend(self._nodes[nid].children_ids)
        return result

    def ancestor_descendant_pairs(self) -> List[Tuple[int, int]]:
        """All (v_i, v_j) where v_i is a strict ancestor of v_j — for L_prog."""
        pairs: List[Tuple[int, int]] = []
        for nid in self._nodes:
            for anc in self.get_ancestors(nid):
                pairs.append((anc, nid))
        return pairs

    # ------------------------------------------------------------------ #
    #  Mutation API                                                        #
    # ------------------------------------------------------------------ #

    def reset(self):
        """Clear the tree for a new episode."""
        self._nodes.clear()
        self._next_id = 0
        self.root_id = None
        self.active_id = None
        self.elevation_pending_parent = None

    def insert(
        self,
        z_v: torch.Tensor,
        a: torch.Tensor,
        q: torch.Tensor,
        s: torch.Tensor,
    ) -> Tuple[float, int]:
        """
        Insert a new timestep observation (Section 3.3).

        Returns
        -------
        mu_t : float  — soft-gate value in (0,1); use for Soft Gating in training
        active_id : int — id of the (possibly newly created) active node
        """
        self.elevation_pending_parent = None

        if self.root_id is None:
            node = self._new_node(z_v, a, q, s, parent_id=None)
            self.root_id = node.node_id
            self.active_id = node.node_id
            return 1.0, node.node_id

        v_act = self._nodes[self.active_id]
        d_t = self._sem_dist(s, v_act.s)

        # Soft gate (Section 3.3, training differentiability)
        mu_t = torch.sigmoid(
            torch.tensor((self.theta_fuse - d_t) / self.tau, dtype=torch.float)
        ).item()

        if d_t < self.theta_fuse:
            # ── Decision A: Merge Update ────────────────────────────────
            self._merge_update(v_act, z_v, a, q, s)
        else:
            # ── Decision B: Branch Split ────────────────────────────────
            self._branch_split(z_v, a, q, s)

        return mu_t, self.active_id

    def add_node(self, node: MemoryNode):
        """Register an externally constructed node (used by elevation)."""
        self._nodes[node.node_id] = node
        if node.node_id >= self._next_id:
            self._next_id = node.node_id + 1

    def alloc_id(self) -> int:
        nid = self._next_id
        self._next_id += 1
        return nid

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _sem_dist(s1: torch.Tensor, s2: torch.Tensor) -> float:
        """Cosine distance ∈ [0, 2]."""
        cos = F.cosine_similarity(
            s1.detach().float().unsqueeze(0),
            s2.detach().float().unsqueeze(0),
        ).item()
        return 1.0 - cos

    def _new_node(
        self,
        z_v: torch.Tensor,
        a: torch.Tensor,
        q: torch.Tensor,
        s: torch.Tensor,
        parent_id: Optional[int],
    ) -> MemoryNode:
        node = MemoryNode(
            node_id=self._next_id,
            z_v=z_v.detach().clone(),
            A=[a.detach().clone()],
            q=q.detach().clone(),
            s=s.detach().clone(),
            n=1,
            w=1.0,
            parent_id=parent_id,
        )
        self._nodes[self._next_id] = node
        self._next_id += 1
        return node

    def _merge_update(
        self,
        v_act: MemoryNode,
        z_v: torch.Tensor,
        a: torch.Tensor,
        q: torch.Tensor,
        s: torch.Tensor,
    ):
        """Welford online-mean update (Section 3.3 Decision A)."""
        v_act.n += 1
        n = v_act.n
        v_act.z_v = (v_act.z_v + (z_v.detach() - v_act.z_v) / n).clone()
        v_act.s   = (v_act.s   + (s.detach()   - v_act.s)   / n).clone()
        v_act.q   = q.detach().clone()
        v_act.A.append(a.detach().clone())
        v_act.w  += self.delta_w

    def _branch_split(
        self,
        z_v: torch.Tensor,
        a: torch.Tensor,
        q: torch.Tensor,
        s: torch.Tensor,
    ):
        """Find best ancestor, create new child (Section 3.3 Decision B)."""
        candidates = [self.active_id] + self.get_ancestors(self.active_id)
        best_id = min(candidates, key=lambda k: self._sem_dist(s, self._nodes[k].s))

        new_node = self._new_node(z_v, a, q, s, parent_id=best_id)
        self._nodes[best_id].children_ids.append(new_node.node_id)
        self.active_id = new_node.node_id

        # Signal if elevation threshold is met
        if len(self._nodes[best_id].children_ids) >= self.K_elev:
            self.elevation_pending_parent = best_id
