"""
MemoryTreeVLA — main model class.
CONSTRUCTION.md Sections 3–5.

Information flow (per timestep during rollout / training):
    image (B,C,H,W), language tokens (B,L,d_lang),
    state q (B,d_q), action a (B,d_a)
         │
         ├── [A] SGMTS encoder:  image + lang_g → Z_V (B,P,d_visual)
         │
         ├── [B] LLM backbone:   lang tokens → lang_hidden (B,L,d_lang)
         │                       mean-pool → lang_g  (B,d_lang)
         │
         ├── [C] Memory tree:    receive (z_v_bar, a, q, s) per step
         │         │              z_v_bar = mean_pool(Z_V) ∈ R^d_visual
         │         └── TreeSSMReadout → Z_M (N_M, d_ssm) [per sample]
         │
         ├── [D] CrossModalFusion: (Z_M, Z_V, lang_g, q) → Z_fused
         │
         └── [E] FlowMatchingActionHead: Z_fused → action prediction

Training computes:
    L_flow  (action head flow matching)   — always
    L_recon (node visual reconstruction)  — phase 1
    L_sem   (language–visual alignment)   — phase 1
    L_prog  (progression ordering)        — phase 2
    L_elev  (elevation consistency)       — phase 3
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from .memory_tree import (
    HierarchicalMemoryTree,
    MemoryNode,
    MLPElevation,
    TreeSSMReadout,
    prune,
    reinforce,
    semantic_elevation,
)
from .sgmts import SGMTSEncoder
from .fusion import CrossModalFusion
from .action_head import FlowMatchingActionHead
from losses.tree_losses import NodeReconDecoder


class MemoryTreeVLA(nn.Module):
    """
    Parameters
    ----------
    llm_path    : directory with Qwen2.5 weights
    d           : unified embedding dim
    d_a         : action dim (default 7)
    d_q         : proprioceptive state dim (default 84)
    d_visual    : SGMTS output dim (= d by default)
    d_ssm       : TreeSSM output dim (= d by default)
    d_state     : SSM hidden-state dim
    patch_size  : image patch size for SGMTS
    H_a         : action prediction horizon
    n_ode       : ODE steps at inference
    theta_fuse  : memory tree merge threshold
    K_elev      : elevation trigger threshold
    delta_w     : weight increment per merge
    tau         : soft-gate temperature
    freeze_llm  : whether to freeze LLM weights
    """

    def __init__(
        self,
        llm_path: str,
        d: int = 256,
        d_a: int = 7,
        d_q: int = 84,
        d_visual: int = 256,
        d_ssm: int = 256,
        d_state: int = 16,
        patch_size: int = 16,
        H_a: int = 16,
        n_ode: int = 20,
        theta_fuse: float = 0.4,
        K_elev: int = 4,
        delta_w: float = 0.1,
        tau: float = 0.1,
        freeze_llm: bool = True,
    ):
        super().__init__()
        self.d      = d
        self.d_a    = d_a
        self.d_q    = d_q
        self.H_a    = H_a

        # ── LLM backbone (Qwen2.5) ───────────────────────────────────
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)
        # Use flash_attention_2 on Ampere+ (sm_80+); fall back gracefully
        _can_flash = (
            torch.cuda.is_available()
            and torch.cuda.get_device_capability()[0] >= 8
        )
        _attn_impl   = "flash_attention_2" if _can_flash else "eager"
        _torch_dtype = torch.bfloat16       if _can_flash else torch.float32
        try:
            self.llm = AutoModel.from_pretrained(
                llm_path,
                trust_remote_code=True,
                attn_implementation=_attn_impl,
                torch_dtype=_torch_dtype,
            )
        except Exception:
            self.llm = AutoModel.from_pretrained(llm_path, trust_remote_code=True)
        d_lang = self.llm.config.hidden_size

        if freeze_llm:
            for p in self.llm.parameters():
                p.requires_grad = False

        # ── SGMTS visual encoder ─────────────────────────────────────
        self.sgmts = SGMTSEncoder(
            d_f=d_visual,
            d_lang=d_lang,
            d_visual=d_visual,
            patch_size=patch_size,
            d_state=d_state,
        )

        # ── Semantic embedding projection (visual → s) ───────────────
        # Bridge between d_visual (SGMTS mean-pool) and d (semantic space)
        self.s_proj = nn.Sequential(
            nn.Linear(d_visual, d),
            nn.LayerNorm(d),
        )

        # ── Memory Tree SSM readout ───────────────────────────────────
        self.tree_ssm = TreeSSMReadout(
            d_node=d,      # z_v / s stored in d dims after s_proj
            d_a=d_a,
            d_q=d_q,
            d_ssm=d_ssm,
            d_state=d_state,
        )

        # ── MLPElevation (operation ③) ───────────────────────────────
        self.mlp_elev = MLPElevation(d=d)

        # ── CrossModal Fusion ────────────────────────────────────────
        self.fusion = CrossModalFusion(
            d_ssm=d_ssm,
            d_visual=d_visual,
            d_lang=d_lang,
            d_q=d_q,
            d=d,
        )

        # ── Action head ──────────────────────────────────────────────
        self.action_head = FlowMatchingActionHead(
            d_a=d_a,
            H_a=H_a,
            d_model=d,
            d_ctx=d,
            N_ode=n_ode,
        )

        # ── Auxiliary reconstruction decoder (phases 1-2) ───────────────
        self.recon_decoder = NodeReconDecoder(d=d)

        # ── One tree per batch sample (managed externally per episode) ─
        self._tree_pool: Dict[int, HierarchicalMemoryTree] = {}
        self._tree_cfg = dict(
            d=d, d_a=d_a, d_q=d_q,
            theta_fuse=theta_fuse, K_elev=K_elev,
            delta_w=delta_w, tau=tau,
        )

    # ---------------------------------------------------------------- #
    #  Tree management helpers                                          #
    # ---------------------------------------------------------------- #

    def get_tree(self, batch_idx: int) -> HierarchicalMemoryTree:
        if batch_idx not in self._tree_pool:
            self._tree_pool[batch_idx] = HierarchicalMemoryTree(**self._tree_cfg)
        return self._tree_pool[batch_idx]

    def reset_trees(self, batch_size: int = 1):
        """Call at the start of every new episode."""
        self._tree_pool.clear()
        for i in range(batch_size):
            self._tree_pool[i] = HierarchicalMemoryTree(**self._tree_cfg)

    # ---------------------------------------------------------------- #
    #  Step-level forward (single timestep, sequential inference)      #
    # ---------------------------------------------------------------- #

    @torch.no_grad()
    def step(
        self,
        image: torch.Tensor,           # (1, C, H, W)
        instruction: str,
        q: torch.Tensor,               # (1, d_q)
        a_prev: Optional[torch.Tensor] = None,   # (1, d_a)
    ) -> torch.Tensor:
        """
        Inference step: returns predicted action sequence (1, H_a, d_a).
        Call reset_trees() at the beginning of each episode.
        """
        device = image.device
        tree   = self.get_tree(0)

        # ── Language encoding ────────────────────────────────────────
        lang_hidden, lang_g = self._encode_language([instruction], device)

        # ── Visual encoding ──────────────────────────────────────────
        Z_V      = self.sgmts(image, lang_g)            # (1, P, d_visual)
        z_v_mean = Z_V.mean(dim=1).squeeze(0)           # (d_visual,)

        # s = semantic projection of mean visual feature
        s = self.s_proj(z_v_mean.unsqueeze(0)).squeeze(0)   # (d,)
        # z_v stored in memory tree also uses s_proj'd space
        z_v_node = s

        # Action to store
        a_node = a_prev if a_prev is not None else torch.zeros(1, self.d_a, device=device)
        a_node = a_node.squeeze(0)

        # ── Update memory tree ───────────────────────────────────────
        tree.insert(z_v_node, a_node, q.squeeze(0), s)

        # Elevation check
        if tree.elevation_pending_parent is not None:
            semantic_elevation(tree, tree.elevation_pending_parent, self.mlp_elev, device)
            tree.elevation_pending_parent = None

        # ── Tree SSM readout ─────────────────────────────────────────
        Z_M = self.tree_ssm(tree, device=device)        # (N_M, d_ssm)
        Z_M = Z_M.unsqueeze(0)                          # (1, N_M, d_ssm)

        # ── Fusion ───────────────────────────────────────────────────
        Z_fused = self.fusion(Z_M, Z_V, lang_g, q)     # (1, N_ctx, d)

        # ── Action prediction ─────────────────────────────────────────
        a_pred = self.action_head.sample(Z_fused, device=device)   # (1, H_a, d_a)
        return a_pred

    # ---------------------------------------------------------------- #
    #  Training forward (trajectory batch)                             #
    # ---------------------------------------------------------------- #

    def forward(
        self,
        images: torch.Tensor,          # (B, T, C, H, W)
        instructions: List[str],       # length B
        states: torch.Tensor,          # (B, T, d_q)
        actions: torch.Tensor,         # (B, T, d_a)
        subtask_ids: Optional[torch.Tensor] = None,  # (B, T) int subtask index
    ) -> Dict[str, torch.Tensor]:
        """
        Full trajectory forward.  Returns dict of scalar losses:
            L_flow  — main flow matching action loss
            L_recon — (optional) node visual reconstruction
            L_sem   — (optional) language–visual alignment
            L_prog  — (optional) progression ordering
        """
        B, T, C, H, W = images.shape
        device = images.device

        # ── Reset trees for this batch ───────────────────────────────
        self.reset_trees(B)

        # ── Language encoding (shared across time steps) ─────────────
        lang_hidden, lang_g = self._encode_language(instructions, device)
        # lang_g: (B, d_lang)

        # ── Trajectory loop ──────────────────────────────────────────
        all_Z_fused, all_mu = [], []

        for t in range(T):
            imgs_t  = images[:, t]                # (B, C, H, W)
            q_t     = states[:, t]                # (B, d_q)
            a_t     = actions[:, t]               # (B, d_a)

            # Visual features
            Z_V_t   = self.sgmts(imgs_t, lang_g) # (B, P, d_visual)

            # Per-sample tree update
            Z_M_list, mu_list = [], []
            for b in range(B):
                z_v_bar = Z_V_t[b].mean(0)                                     # (d_visual,)
                s_b     = self.s_proj(z_v_bar.unsqueeze(0)).squeeze(0)          # (d,)
                z_v_b   = s_b

                tree_b  = self.get_tree(b)
                mu_t_b, _ = tree_b.insert(z_v_b, a_t[b], q_t[b], s_b)
                mu_list.append(mu_t_b)

                # Trigger elevation if pending
                if hasattr(tree_b, '_elevation_pending') and tree_b._elevation_pending is not None:
                    semantic_elevation(tree_b, tree_b._elevation_pending, self.mlp_elev, device)
                    tree_b._elevation_pending = None

                Z_M_b = self.tree_ssm(tree_b, device=device)  # (N_M, d_ssm)
                Z_M_list.append(Z_M_b)

            # Pad Z_M to same length
            N_max  = max(z.shape[0] for z in Z_M_list)
            d_ssm  = Z_M_list[0].shape[1]
            Z_M_pad = torch.zeros(B, N_max, d_ssm, device=device)
            for b, z in enumerate(Z_M_list):
                Z_M_pad[b, :z.shape[0]] = z

            # Fusion
            Z_fused_t = self.fusion(Z_M_pad, Z_V_t, lang_g, q_t)   # (B, N_ctx, d)
            all_Z_fused.append(Z_fused_t)
            all_mu.append(sum(mu_list) / B)

        # ── Action loss over sliding H_a windows ─────────────────────
        L_flow = self._compute_flow_loss(all_Z_fused, actions, device)

        # ── Progression loss (anchor-based, at final timestep) ───────
        L_prog = self._compute_prog_loss(B, device)

        losses = {
            "L_flow": L_flow,
            "L_prog": L_prog,
        }
        return losses

    # ---------------------------------------------------------------- #
    #  Loss helpers                                                     #
    # ---------------------------------------------------------------- #

    def _compute_flow_loss(
        self,
        all_Z_fused: List[torch.Tensor],   # list of T × (B, N_ctx, d)
        actions: torch.Tensor,             # (B, T, d_a)
        device: torch.device,
    ) -> torch.Tensor:
        """Average flow-matching loss over all timestep windows."""
        B, T, d_a = actions.shape
        losses_t = []
        for t, Z_fused_t in enumerate(all_Z_fused):
            t_end = min(t + self.H_a, T)
            a_gt  = actions[:, t:t_end]            # (B, ≤H_a, d_a)
            # Pad to H_a
            pad   = self.H_a - a_gt.shape[1]
            if pad > 0:
                a_gt = torch.cat([a_gt, a_gt[:, -1:].expand(-1, pad, -1)], dim=1)
            losses_t.append(self.action_head.flow_loss(a_gt, Z_fused_t))
        return torch.stack(losses_t).mean()

    def _compute_prog_loss(self, B: int, device: torch.device) -> torch.Tensor:
        """
        Progression loss: ancestor node embedding should have lower
        norm than descendant when projected to a ranking head.
        Uses cosine distance between ancestor-descendant pairs.
        Approximated here as a margin loss.
        """
        total_loss = torch.tensor(0.0, device=device)
        count = 0
        for b in range(B):
            tree = self.get_tree(b)
            pairs = tree.ancestor_descendant_pairs()
            if not pairs:
                continue
            for anc_id, desc_id in pairs:
                anc_node  = tree.nodes[anc_id]
                desc_node = tree.nodes[desc_id]
                s_anc  = anc_node.s.to(device)
                s_desc = desc_node.s.to(device)
                # Hinge: cosine(anc, desc) should be > other pairs
                # Simple proxy: parent s should align with child s (they are on the same branch)
                cos_sim = torch.nn.functional.cosine_similarity(
                    s_anc.unsqueeze(0), s_desc.unsqueeze(0)
                )
                total_loss = total_loss + (1.0 - cos_sim)
                count += 1
        if count == 0:
            return total_loss
        return total_loss / count

    # ---------------------------------------------------------------- #
    #  Language encoding                                                #
    # ---------------------------------------------------------------- #

    def _encode_language(
        self,
        instructions: List[str],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            lang_hidden : (B, L, d_lang)
            lang_g      : (B, d_lang) mean-pooled CLS
        """
        enc = self.tokenizer(
            instructions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(device)

        with torch.set_grad_enabled(not all(not p.requires_grad for p in self.llm.parameters())):
            out = self.llm(**enc)

        lang_hidden = out.last_hidden_state          # (B, L, d_lang)
        # Mean-pool over non-padding tokens
        mask = enc["attention_mask"].unsqueeze(-1).float()
        lang_g = (lang_hidden * mask).sum(1) / mask.sum(1).clamp(min=1)  # (B, d_lang)
        return lang_hidden, lang_g
