"""
MemoryTree: Hierarchical memory structure for long-horizon manipulation tasks.
Stores task decomposition as a tree, with each node representing a sub-goal.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class MemoryNode:
    """A single node in the memory tree, corresponding to a sub-goal."""
    node_id: int
    description: str
    embedding: Optional[torch.Tensor] = None
    children: List["MemoryNode"] = field(default_factory=list)
    parent: Optional["MemoryNode"] = None
    is_completed: bool = False


class MemoryTree(nn.Module):
    """
    Hierarchical memory tree that decomposes long-horizon tasks into
    a structured sequence of sub-goals.
    """

    def __init__(self, embed_dim: int = 512, max_depth: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_depth = max_depth
        self.root: Optional[MemoryNode] = None
        self._node_counter = 0

    def build_from_task(self, task_description: str, task_embedding: torch.Tensor):
        """Build the memory tree from a high-level task description."""
        self.root = MemoryNode(
            node_id=self._new_id(),
            description=task_description,
            embedding=task_embedding,
        )

    def get_current_subgoal(self) -> Optional[MemoryNode]:
        """Retrieve the current active sub-goal node."""
        return self._find_active_node(self.root)

    def mark_completed(self, node: MemoryNode):
        """Mark a sub-goal as completed and advance to the next."""
        node.is_completed = True

    def _find_active_node(self, node: Optional[MemoryNode]) -> Optional[MemoryNode]:
        if node is None or node.is_completed:
            return None
        if not node.children:
            return node
        for child in node.children:
            result = self._find_active_node(child)
            if result is not None:
                return result
        return None

    def _new_id(self) -> int:
        self._node_counter += 1
        return self._node_counter

    def reset(self):
        self.root = None
        self._node_counter = 0
