"""
DualTreeAdapter_Evo1 — 双树增强模块 Evo-1 适配器 (CONSTRUCTION.md §3.2)

对骨架（Evo-1/InternVL3-1B）的最小侵入：
  1. 在 vision_model 末层注册只读 hook，提取 patch 特征 P_t
  2. 运行 SGMTS + GateFusion，将增强后的 V_t' 重新投影后替换 vit_embeds
  3. 在 LLM 输入序列末尾拼接 HMT 读出的记忆 token

骨架权重加载：与原版 Evo-1 完全相同，直接调用
    AutoModel.from_pretrained("OpenGVLab/InternVL3-1B")
无任何障碍。

训练阶段控制（mode 参数）：
  pretrain : L_boundary + L_sem（骨架全冻结）
  phase1   : L_flow only（骨架全冻结）
  phase2   : L_flow only（骨架 LLM 以 0.1× LR 解冻）
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_adapter import BaseDualTreeAdapter
from ..model.sgmts.sgmts import SGMTS
from ..model.gate_fusion import GateFusion
from ..model.memory_tree import (
    HierarchicalMemoryTree,
    MLPElevation,
    TreeSSMReadout,
    merge,
    branch,
    propagate_elevation_to_root,
)
from ..model.action_head.jump_aware_head import JumpAwareHead
from ..losses.tree_losses import l_boundary, l_sem, l_elev


class DualTreeAdapter_Evo1(BaseDualTreeAdapter):
    """
    双树增强模块 — Evo-1 适配器（InternVL3-1B 骨架）

    Parameters
    ----------
    backbone       : EVO1 实例（含已加载的 InternVL3-1B 权重）
    d_vit          : ViT 隐藏维度，InternVL3-1B 为 896
    d_a            : 动作维度，默认 7
    d_ssm          : TreeSSM 输出维度
    d_state        : SSM 状态维度
    mount_tau      : HMT 语义挂载阈值
    max_tree_depth : HMT 最大树深度（超出剪枝）
    """

    def __init__(
        self,
        backbone,                    # EVO1 实例
        d_vit: int = 896,
        d_a: int = 7,
        d_ssm: int = 256,
        d_state: int = 16,
        mount_tau: float = 0.4,
        max_tree_depth: int = 4,
        alpha: float = 0.5,
        delta_w: float = 0.1,
    ):
        super().__init__()
        self.backbone       = backbone
        self.d_vit          = d_vit
        self.d_a            = d_a
        self.d_ssm          = d_ssm
        self.max_tree_depth = max_tree_depth

        # ── 检测 ViT 原始输出维度（可能与 LLM 字符空间维度不同）─────
        # InternVL3-1B: ViT hidden = 1024， LLM hidden = 896
        try:
            d_patch = backbone.embedder.model.vision_model.config.hidden_size
        except AttributeError:
            d_patch = d_vit   # 回退到 d_vit（单模型测试时）

        # ── 双树模块组件 ────────────────────────────────────
        self.sgmts     = SGMTS(d_vit=d_vit, d_patch=d_patch, d_state=d_state, alpha=alpha)
        self.gate_fuse = GateFusion(d_vit=d_patch)
        self.jump_head = JumpAwareHead(d_a=d_a, d_state=d_state)
        self.tree_ssm  = TreeSSMReadout(d_node=d_vit, d_ssm=d_ssm, d_state=d_state)
        self.mlp_elev  = MLPElevation(d=d_vit)
        self.mem_proj  = nn.Linear(d_ssm, d_vit)

        # L_sem 用：将抽象节点语义嵌入投影到与 LLM 任务嵌入对齐的空间
        self.sem_proj  = nn.Linear(d_vit, d_vit, bias=False)

        # ── 树实例池（每个 batch index 对应一棵 HMT）─────────────────
        self._tree_pool: Dict[int, HierarchicalMemoryTree] = {}
        self._mount_tau = mount_tau
        self._delta_w   = delta_w

        # ── ViT 特征缓存（由 hook 填充）──────────────────────────────
        self._P_t_raw: Optional[torch.Tensor] = None  # 含 CLS

        # 注册只读 hook：ViT 最后一层输出 → 保存到 self._P_t_raw
        self._register_vit_hook()

    # ---------------------------------------------------------------- #
    #  HMT CPU helpers                                                  #
    # ---------------------------------------------------------------- #

    def _mlp_elev_cpu(self, z: torch.Tensor) -> torch.Tensor:
        """
        在 mlp_elev 所在设备上运行，返回 CPU tensor。
        HMT 树节点全部存 CPU tensor，而 mlp_elev 权重在 CUDA，需要此桥接。

        Args:
            z : (d_vit,) CPU float tensor
        Returns:
            s : (d_vit,) CPU float tensor
        """
        dev = next(self.mlp_elev.parameters()).device
        with torch.no_grad():
            return self.mlp_elev(z.float().to(dev)).detach().cpu()

    # ---------------------------------------------------------------- #
    #  Hook                                                             #
    # ---------------------------------------------------------------- #

    def _register_vit_hook(self):
        embedder = self.backbone.embedder
        vit_last = embedder.model.vision_model.encoder.layers[-1]

        def _hook(module, input, output):
            # InternVL3 ViT 层输出为 tuple (hidden_states, ...) 或直接 tensor
            hs = output[0] if isinstance(output, tuple) else output
            self._P_t_raw = hs  # (total_tiles, N_vit_tokens, d_vit) 含 CLS

        vit_last.register_forward_hook(_hook)

    # ---------------------------------------------------------------- #
    #  Tree helpers                                                     #
    # ---------------------------------------------------------------- #

    def get_tree(self, batch_idx: int) -> HierarchicalMemoryTree:
        if batch_idx not in self._tree_pool:
            self._tree_pool[batch_idx] = HierarchicalMemoryTree(
                d=self.d_vit,
                d_a=self.d_a,
                mount_tau=self._mount_tau,
                delta_w=self._delta_w,
            )
        return self._tree_pool[batch_idx]

    def reset(self, batch_size: int = 1) -> None:
        self._tree_pool = {}

    def _get_top_abstract_nodes(self, tree: HierarchicalMemoryTree) -> Optional[torch.Tensor]:
        """返回 HMT 顶层（最浅）两个抽象节点语义嵌入的均值，作为 s_top。"""
        abstract_nodes = [
            n for n in tree.nodes.values()
            if not n.is_leaf() and n.s is not None
        ]
        if not abstract_nodes:
            return None
        # 按深度升序取最浅的 2 个
        abstract_nodes.sort(key=lambda n: tree.depth(n.node_id))
        top2 = abstract_nodes[:2]
        s_stack = torch.stack([n.s for n in top2]).float()
        return s_stack.mean(0)   # (d_vit,)

    def _get_m_ctx(self, tree: HierarchicalMemoryTree, device: torch.device) -> torch.Tensor:
        """用 TreeSSMReadout 读出记忆 token m_ctx (d_ssm,)。无抽象节点时返回零向量。"""
        abstract_nodes = [n for n in tree.nodes.values() if not n.is_leaf()]
        if not abstract_nodes:
            return torch.zeros(self.d_ssm, device=device)
        Y = self.tree_ssm(tree)   # (N_abs, d_ssm)
        return Y[-1]              # 最后一个 BFS 节点的输出

    def _fit_image_mask(self, image_mask: Optional[torch.Tensor], num_views: int) -> torch.Tensor:
        if image_mask is None:
            return torch.ones(num_views, dtype=torch.bool)
        mask = torch.as_tensor(image_mask, dtype=torch.bool).flatten()
        if mask.numel() == num_views:
            return mask
        if mask.numel() == 1:
            return mask.repeat(num_views)
        fitted = torch.zeros(num_views, dtype=torch.bool)
        fitted[: min(mask.numel(), num_views)] = mask[:num_views]
        return fitted

    def _normalize_batch_images_and_masks(
        self,
        images,
        image_mask: Optional[torch.Tensor],
        mode: str,
    ) -> tuple[list, list[torch.Tensor]]:
        if isinstance(images, torch.Tensor):
            if images.ndim == 4:
                batch_images = [[images[b]] for b in range(images.shape[0])]
            elif images.ndim == 5:
                if mode in ("phase1", "phase2"):
                    batch_images = [[images[b, v] for v in range(images.shape[1])] for b in range(images.shape[0])]
                else:
                    batch_images = [[images[b, 0]] for b in range(images.shape[0])]
            elif images.ndim == 6:
                batch_images = [[images[b, 0, v] for v in range(images.shape[2])] for b in range(images.shape[0])]
            else:
                raise ValueError(f"Unsupported image tensor shape: {tuple(images.shape)}")
        elif isinstance(images, list):
            if images and isinstance(images[0], (list, tuple)):
                batch_images = [list(sample_views) for sample_views in images]
            else:
                batch_images = [[img] for img in images]
        else:
            raise TypeError(f"Unsupported images type: {type(images)!r}")

        if isinstance(image_mask, torch.Tensor) and image_mask.ndim == 2:
            batch_masks = [self._fit_image_mask(image_mask[b], len(batch_images[b])) for b in range(len(batch_images))]
        elif isinstance(image_mask, list) and len(image_mask) == len(batch_images):
            batch_masks = [self._fit_image_mask(image_mask[b], len(batch_images[b])) for b in range(len(batch_images))]
        else:
            batch_masks = [self._fit_image_mask(image_mask, len(sample_views)) for sample_views in batch_images]
        return batch_images, batch_masks

    # ---------------------------------------------------------------- #
    #  Task encoding                                                    #
    # ---------------------------------------------------------------- #

    @torch.no_grad()
    def _encode_task(self, prompt: str, device: torch.device) -> torch.Tensor:
        """
        将任务描述编码为均值池化嵌入 g_task (1, d_vit)。
        使用骨架 LLM 的词嵌入层（冻结），不做 transformer 推理。
        """
        embedder = self.backbone.embedder
        tokens   = embedder.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=64
        ).to(device)
        embeds  = embedder.model.language_model.get_input_embeddings()(tokens.input_ids)
        g_task  = embeds.mean(dim=1).to(torch.float32)  # (1, d_vit)
        return g_task

    # ---------------------------------------------------------------- #
    #  Batched VLM forward (Phase 1 / Phase 2)                         #
    # ---------------------------------------------------------------- #

    def _embed_batch_flow(
        self,
        images: list,                    # List[List[Tensor(C,H,W)]], outer len=B
        instructions: list,              # List[str] len=B
        image_masks: list[torch.Tensor], # len=B, each (V,)
        device: torch.device,
    ) -> torch.Tensor:
        """
        Phase 1/2 专用批量 VLM forward。

        与逐样本 _embed_with_dual_tree 等价，但只做一次 ViT forward、
        一次 LLM forward，速度约为串行版本的 B 倍。

        HMT 记忆 token 在 phase1/2 训练中从不被写入（仅 pretrain 写树），
        始终为零向量，所以不需要查询树池。

        Returns
        -------
        vl_feat : (B, d_vit) float32  — CLS token 输出，直接传给 predict_action
        """
        B = len(images)
        embedder = self.backbone.embedder
        flat_images = [view for sample_views in images for view in sample_views]
        if not flat_images:
            raise ValueError("No images provided for batched flow embedding.")

        # ── Step 1: 所有图像共享一次 ViT forward ──────────────────────
        # _preprocess_images 接受 List[Tensor/PIL]，tiles 沿 dim=0 拼接
        pixel_values, num_tiles_list = embedder._preprocess_images(flat_images)
        # extract_feature 触发 hook → self._P_t_raw = (sum_tiles, N_vit, d_patch)
        vit_embeds_all = embedder.model.extract_feature(pixel_values)
        # P_t: 去除 CLS token
        P_t_raw = self._P_t_raw[:, 1:, :].to(torch.float32)   # (sum_tiles, N_p, d_patch)

        # 按图像分割
        vit_split = list(vit_embeds_all.split(num_tiles_list, dim=0))  # List[(n_i, N_tok, d_vit)]
        P_t_split = list(P_t_raw.split(num_tiles_list, dim=0))        # List[(n_i, N_p, d_patch)]

        # ── Step 2: 每视角独立 SGMTS + GateFusion，随后按样本重组 ───────
        sample_num_tiles = []
        sample_view_embeds = []
        flat_offset = 0
        for b in range(B):
            g_task_b = self._encode_task(instructions[b] if b < len(instructions) else "", device)
            sample_tiles = []
            sample_embeds = []
            for _ in images[b]:
                P_t_view = P_t_split[flat_offset]
                n_tv = P_t_view.shape[0]
                g_task_tiled = g_task_b.expand(n_tv, -1)
                Z_v_view = self.sgmts(P_t_view, g_task_tiled, [None] * n_tv)
                V_t_prime_view = self.gate_fuse(Z_v_view, P_t_view)

                T_v, N_p_v, d_p_v = V_t_prime_view.shape
                h = w = int(math.isqrt(N_p_v))
                V_ps = V_t_prime_view.reshape(T_v, h, w, d_p_v)
                V_ps = embedder.model.pixel_shuffle(
                    V_ps.to(torch.bfloat16),
                    scale_factor=embedder.model.downsample_ratio,
                )
                V_ps = V_ps.reshape(T_v, -1, V_ps.shape[-1])
                emb_new = embedder.model.mlp1(V_ps)
                sample_embeds.append(emb_new.reshape(-1, emb_new.shape[-1]))
                sample_tiles.append(num_tiles_list[flat_offset])
                flat_offset += 1
            sample_num_tiles.append(sample_tiles)
            sample_view_embeds.append(torch.cat(sample_embeds, dim=0))

        # ── Step 3: 构建每图像的 input_embeds，再堆叠做单次 LLM forward ─
        all_embeds = []
        all_masks  = []
        zero_mem = torch.zeros(self.d_ssm, device=device)
        mem_vec  = self.mem_proj(zero_mem.float())             # (d_vit,)

        for b in range(B):
            instr_b  = instructions[b] if b < len(instructions) else ""
            num_tiles_b = sample_num_tiles[b]
            prompt_b = embedder._build_multimodal_prompt(num_tiles_b, instr_b)
            input_embeds_b, attn_mask_b = embedder._prepare_and_fuse_embeddings(
                prompt_b,
                sample_view_embeds[b],
                image_masks[b].to(device),
                num_tiles_b,
            )
            # 拼接零记忆 token
            mem_tok = mem_vec.to(input_embeds_b.dtype).unsqueeze(0).unsqueeze(0)  # (1,1,d)
            input_embeds_b = torch.cat([input_embeds_b, mem_tok], dim=1)
            attn_mask_b    = torch.cat(
                [attn_mask_b, torch.ones(1, 1, dtype=attn_mask_b.dtype, device=device)], dim=1
            )
            all_embeds.append(input_embeds_b)   # (1, L+1, d_vit)
            all_masks.append(attn_mask_b)        # (1, L+1)

        # ── Step 4: 单次 LLM forward ──────────────────────────────────
        input_embeds_batch = torch.cat(all_embeds, dim=0)   # (B, L+1, d_vit)
        attn_mask_batch    = torch.cat(all_masks,  dim=0)   # (B, L+1)

        outputs = embedder.model.language_model(
            inputs_embeds=input_embeds_batch,
            attention_mask=attn_mask_batch,
            output_hidden_states=True,
            return_dict=True,
        )
        fused_hidden = outputs.hidden_states[-1].to(torch.float32)  # (B, L+1, d_vit)
        return fused_hidden[:, 0, :]   # (B, d_vit)

    # ---------------------------------------------------------------- #
    #  Core forward                                                     #
    # ---------------------------------------------------------------- #

    def _embed_with_dual_tree(
        self,
        images: list,
        image_mask: torch.Tensor,
        prompt: str,
        batch_idx: int = 0,
    ):
        """
        完整的视觉-语言融合前向（含双树增强）。

        Returns:
            fused_hidden : (1, seq_len+1, d_vit) float32，含记忆 token
            attn_mask    : (1, seq_len+1)
        """
        embedder = self.backbone.embedder
        device   = torch.device(embedder.device)

        # ── Step 1: 图像预处理 ─────────────────────────────────────
        pixel_values, num_tiles_list = embedder._preprocess_images(images)
        # 触发 ViT forward hook → 填充 self._P_t_raw
        vit_embeds_orig = embedder.model.extract_feature(pixel_values)
        # P_t: 剤除 CLS token → (total_tiles, N_p, d_patch=1024) — 供 SGMTS 平面扫描
        P_t = self._P_t_raw[:, 1:, :].to(torch.float32)
        # z_v_feat: mlp1 投影后的 LLM 空间特征 (total_tiles, 256, d_vit=896) → 均値 → (total_tiles, d_vit)
        # 用于 HMT 存储和 MLPElevation（必须是 896-dim）
        z_v_feat = vit_embeds_orig.mean(dim=1).to(torch.float32)  # (total_tiles, d_vit=896)

        # ── Step 2: 任务编码 & s_top ────────────────────────────────
        tree  = self.get_tree(batch_idx)
        g_task = self._encode_task(prompt, device)  # (1, d_vit)
        s_top  = self._get_top_abstract_nodes(tree)

        total_tiles = P_t.shape[0]
        # 广播 g_task 到 total_tiles
        g_task_tiled = g_task.expand(total_tiles, -1)  # (total_tiles, d_vit)
        s_top_list   = [s_top] * total_tiles

        # ── Step 3: SGMTS ────────────────────────────────────────────
        Z_v = self.sgmts(P_t, g_task_tiled, s_top_list)  # (total_tiles, N_p, d_vit)

        # ── Step 4: GateFusion ───────────────────────────────────────
        V_t_prime = self.gate_fuse(Z_v, P_t)             # (total_tiles, N_p, d_vit)

        # ── Step 5: pixel_shuffle → MLP projector ───────────────────
        # V_t_prime: (total_tiles, N_p, d_patch)  例如 (1, 1024, 1024)
        # InternVL3 的 mlp1 期望输入 (N, num_tokens, d_patch / ratio^2)
        # 需先做与 extract_feature 相同的 pixel_shuffle，再送 mlp1
        T, N_p, d_p = V_t_prime.shape
        h = w = int(math.isqrt(N_p))
        # reshape → (T, h, w, d_p) 供 pixel_shuffle
        V_ps = V_t_prime.reshape(T, h, w, d_p)
        V_ps = embedder.model.pixel_shuffle(
            V_ps.to(torch.bfloat16),
            scale_factor=embedder.model.downsample_ratio,
        )  # (T, h', w', d_p / ratio^2)
        V_ps = V_ps.reshape(T, -1, V_ps.shape[-1])  # (T, num_tokens, 4096)
        vit_embeds_new = embedder.model.mlp1(V_ps)  # (T, num_tokens, d_vit=896)
        # 展平为 _prepare_and_fuse_embeddings 需要的形状
        vit_embeds_new = vit_embeds_new.reshape(-1, vit_embeds_new.shape[-1])

        # ── Step 6: 构建多模态 Prompt & 融合嵌入 ─────────────────────
        prompt_str     = embedder._build_multimodal_prompt(num_tiles_list, prompt)
        input_embeds, attn_mask = embedder._prepare_and_fuse_embeddings(
            prompt_str, vit_embeds_new, image_mask, num_tiles_list
        )

        # ── Step 7: 拼接记忆 token ───────────────────────────────────
        m_ctx      = self._get_m_ctx(tree, device)                # (d_ssm,)
        mem_token  = self.mem_proj(m_ctx.float())                  # (d_vit,)
        mem_token  = mem_token.to(input_embeds.dtype)
        mem_token  = mem_token.unsqueeze(0).unsqueeze(0)           # (1, 1, d_vit)
        input_embeds = torch.cat([input_embeds, mem_token], dim=1)
        attn_mask    = torch.cat(
            [attn_mask, torch.ones(1, 1, dtype=attn_mask.dtype, device=device)],
            dim=1
        )

        # ── Step 8: LLM 前向 ─────────────────────────────────────────
        outputs = embedder.model.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attn_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        fused_hidden = outputs.hidden_states[-1].to(torch.float32)  # (1, seq_len+1, d_vit)

        return fused_hidden, attn_mask, P_t, Z_v, tree, z_v_feat

    # ---------------------------------------------------------------- #
    #  BaseDualTreeAdapter interface                                    #
    # ---------------------------------------------------------------- #

    def forward(
        self,
        images,                                      # List[PIL/Tensor] 或 (B,C,H,W) Tensor
        instructions: list = None,                   # List[str]，训练脚本传入
        states: Optional[torch.Tensor] = None,       # (B, d_q)，训练脚本传入
        actions: Optional[torch.Tensor] = None,      # (B, T, d_a)，训练脚本传入
        subtask_ids: Optional[torch.Tensor] = None,  # (B, T) pretrain 用
        mode: str = "phase1",
        # --- 旧参数名兼容 ---
        image_mask: Optional[torch.Tensor] = None,
        prompt: str = None,
        state: Optional[torch.Tensor] = None,
        actions_gt: Optional[torch.Tensor] = None,
        # --- 预提取特征缓存（pretrain 快速路径）---
        precomputed_vit: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        训练 forward — 支持批量训练脚本调用格式。

        训练脚本约定
        ------------
        images       : (B, C, H, W) 张量 或 List[PIL.Image/Tensor]
        instructions : List[str]，长度 B
        states       : (B, d_q)
        actions      : (B, T, d_a)
        subtask_ids  : (B, T) — pretrain 阶段需要

        Returns
        -------
        losses : dict with keys depending on mode
        """
        # 兼容旧参数名
        if instructions is None and prompt is not None:
            instructions = [prompt]
        if states is None and state is not None:
            states = state
        if actions is None and actions_gt is not None:
            actions = actions_gt

        batch_images, batch_image_masks = self._normalize_batch_images_and_masks(images, image_mask, mode)
        B = len(batch_images)
        if instructions is None:
            instructions = [""] * B

        # 安全拦截：images 含 None 元素 = 缓存模式激活
        # 无论 forward 是否有 precomputed_vit 参数，都路由到缓存路径
        if any(any(img is None for img in sample_views) for sample_views in batch_images):
            if precomputed_vit is not None:
                return self._forward_pretrain_cached(
                    precomputed_vit=precomputed_vit,
                    instructions=instructions,
                    actions=actions,
                    subtask_ids=subtask_ids,
                )
            raise RuntimeError(
                "[DualTreeVLA] images 中含 None，但 precomputed_vit 未设置。\n"
                "请先运行: python scripts/extract_pretrain_features.py\n"
                "然后在 pretrain.yaml 的 data 节中设置 feat_cache_dir。"
            )

        # ── Phase 1 / Phase 2：批量 VLM forward（比串行快约 B 倍）────
        if mode in ("phase1", "phase2"):
            assert actions is not None, "actions required for phase1/phase2"
            device = next(self.backbone.action_head.parameters()).device
            vl_feat = self._embed_batch_flow(batch_images, instructions, batch_image_masks, device)
            # (B, d_vit)  —— 单次 ViT + 单次 LLM forward
            loss_flow, _ = self.backbone.predict_action(
                vl_feat, states, actions_gt=actions
            )
            return {"L_flow": loss_flow, "total": loss_flow}

        # ── Pretrain：保留原逐样本串行逻辑（HMT 树需要逐帧更新）────────
        loss_keys: Dict[str, torch.Tensor] = {}

        for b in range(B):
            img_b   = batch_images[b]
            instr_b = instructions[b] if b < len(instructions) else ""
            st_b    = states[b:b+1] if states is not None else None
            act_b   = actions[b:b+1] if actions is not None else None
            sid_b   = subtask_ids[b:b+1] if subtask_ids is not None else None

            fused_hidden, attn_mask, P_t, Z_v, tree, z_v_feat_b = self._embed_with_dual_tree(
                img_b, batch_image_masks[b].to(next(self.mem_proj.parameters()).device), instr_b, batch_idx=b
            )
            vl_feat = fused_hidden[:, 0, :]  # (1, d_vit)

            if mode == "pretrain":
                step_losses = self._pretrain_losses(tree, P_t, z_v_feat_b, act_b, sid_b, instr_b)
            else:
                assert act_b is not None, "actions required for phase1/phase2"
                loss_flow, _ = self.backbone.predict_action(
                    vl_feat, st_b, actions_gt=act_b
                )
                step_losses = {"L_flow": loss_flow, "total": loss_flow}

            for k, v in step_losses.items():
                loss_keys[k] = loss_keys.get(k, torch.zeros((), device=v.device)) + v / B

        if "total" not in loss_keys:
            loss_keys["total"] = sum(v for k, v in loss_keys.items()
                                     if k not in ("total",))
        return loss_keys

    def _forward_pretrain_cached(
        self,
        precomputed_vit: Dict[str, torch.Tensor],
        instructions: Optional[list],
        actions: Optional[torch.Tensor],
        subtask_ids: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Pretrain 快速路径：跳过 ViT + SGMTS + LLM forward。

        precomputed_vit:
            P_t_raw  : (B, tiles, 1025, 1024) float32
            z_v_feat : (B, tiles, 896)         float32
        """
        P_t_raw_b_all  = precomputed_vit["P_t_raw"]   # (B, tiles, 1025, 1024)
        z_v_feat_b_all = precomputed_vit["z_v_feat"]  # (B, tiles, 896)
        B = z_v_feat_b_all.shape[0]

        if instructions is None:
            instructions = [""] * B

        loss_keys: Dict[str, torch.Tensor] = {}
        for b in range(B):
            z_v_feat_b = z_v_feat_b_all[b]   # (tiles, 896)
            tree       = self.get_tree(b)
            instr_b    = instructions[b] if b < len(instructions) else ""
            act_b      = actions[b:b+1]     if actions is not None else None
            sid_b      = subtask_ids[b:b+1] if subtask_ids is not None else None

            step_losses = self._pretrain_losses(
                tree, None, z_v_feat_b, act_b, sid_b, instr_b
            )
            for k, v in step_losses.items():
                loss_keys[k] = loss_keys.get(k, torch.zeros((), device=v.device)) + v / B

        if "total" not in loss_keys:
            loss_keys["total"] = sum(v for k, v in loss_keys.items()
                                     if k not in ("total",))
        return loss_keys

    def _pretrain_losses(
        self,
        tree: HierarchicalMemoryTree,
        P_t: torch.Tensor,
        z_v_feat: torch.Tensor,          # (total_tiles, d_vit=896) — mlp1 投影空间
        actions_gt: Optional[torch.Tensor],
        subtask_ids: Optional[torch.Tensor],
        prompt: str,
    ) -> Dict[str, torch.Tensor]:
        """
        预训练损失：L_boundary + 0.5×L_sem + 0.2×L_elev

        同时在线更新记忆树（逐帧 Merge/Branch/Elevate），使 HMT
        在预训练阶段就能正确积累子任务结构，供 SGMTS 通过 s_top 引导。
        """
        device = z_v_feat.device if P_t is None else P_t.device

        if actions_gt is None:
            zero = torch.zeros(1, device=device)
            return {"L_boundary": zero, "L_sem": zero, "L_elev": zero, "total": zero}

        B, T, _ = actions_gt.shape
        logits_list, labels_list = [], []

        # 每帧逐步更新 HMT（使树随轨迹演进）
        # z_v_feat: (total_tiles, d_vit=896) — mlp1 投影空间特征，与 HMT 维度匹配
        z_v_seq = z_v_feat  # (total_tiles, d_vit=896)

        for t in range(T):
            a_new = actions_gt[:, t, :]               # (B, d_a)
            if t == 0:
                A_act = a_new.unsqueeze(1)             # (B, 1, d_a)
            else:
                A_act = actions_gt[:, max(0, t - self.jump_head.max_len):t, :]  # (B, L, d_a)

            _, logit = self.jump_head(A_act, a_new)   # (B,)
            logits_list.append(logit)

            # 边界标签：优先使用 subtask_ids（干净的子任务切换点），
            # 回退到基于动作统计的自监督标签
            if t > 0:
                if subtask_ids is not None and subtask_ids.shape[-1] > t:
                    label = (subtask_ids[:, t] != subtask_ids[:, t - 1]).float().to(device)
                else:
                    a_prev  = actions_gt[:, :t, :]
                    a_mean  = a_prev.mean(dim=1)
                    diff    = (a_new - a_mean).norm(dim=-1)
                    sigma_a = (a_prev - a_mean.unsqueeze(1)).norm(dim=-1).std(dim=-1, correction=0).clamp(min=1e-6)
                    label   = (diff > 2.0 * sigma_a).float()
            else:
                label = torch.zeros(B, device=device)
            labels_list.append(label)

            # ── 在线 HMT 更新（仅第一个 batch 维，避免树爆炸） ──────
            with torch.no_grad():
                a_cpu    = a_new[0].cpu()
                z_cpu    = z_v_seq[0].cpu() if z_v_seq.shape[0] > 0 else torch.zeros(self.d_vit)
                s_cpu    = self._mlp_elev_cpu(z_cpu)
                force_b  = bool(logit[0].item() >= 0.0)   # logit > 0 → p_jump ≥ 0.5
                tree.insert(z_v=z_cpu, a=a_cpu, force_branch=force_b, s_current=s_cpu)
                if force_b and tree.elevation_pending_parent is not None:
                    from ..model.memory_tree.operations import semantic_elevation
                    _elev_dev = next(self.mlp_elev.parameters()).device
                    semantic_elevation(
                        tree, tree.elevation_pending_parent,
                        self.mlp_elev, device=_elev_dev
                    )
                    propagate_elevation_to_root(
                        tree, tree.elevation_pending_parent,
                        self.mlp_elev, device=_elev_dev
                    )

        logits_t = torch.stack(logits_list, dim=1).reshape(-1)   # (B*T,)
        labels_t = torch.stack(labels_list, dim=1).reshape(-1)   # (B*T,)
        loss_b   = l_boundary(logits_t, labels_t)

        # ── L_sem：分两部分 ────────────────────────────────────────────
        g_task = self._encode_task(prompt, device)   # (1, d_vit)

        # (a) 抽象节点语义对齐（trains sem_proj；节点 s 为 detach，但参数有梯度）
        abstract_nodes = [n for n in tree.nodes.values() if not n.is_leaf() and n.s is not None]
        if len(abstract_nodes) >= 1:
            s_nodes     = torch.stack([n.s.to(device).float() for n in abstract_nodes])
            s_proj      = self.sem_proj(s_nodes)
            g_task_exp  = g_task.expand(len(abstract_nodes), -1)
            loss_s_node = l_sem(s_proj, g_task_exp)
        else:
            loss_s_node = torch.zeros(1, device=device).squeeze()

        # (b) 当前帧视觉特征对齐（z_v_feat 在计算图中，梯度经 sem_proj → mlp1
        #     → pixel_shuffle → gate_fuse → sgmts 完整传播）
        z_v_mean   = z_v_feat.mean(dim=0, keepdim=True)  # (1, d_vit)
        z_v_proj   = self.sem_proj(z_v_mean)              # (1, d_vit)
        loss_s_vis = l_sem(z_v_proj, g_task)              # cosine alignment

        loss_s = 0.5 * loss_s_node + 0.5 * loss_s_vis

        # ── L_elev：语义提升一致性 ──────────────────────────────────
        # 树节点中 n.s / n.z_v 均为 detach 的 CPU 张量，直接用作 loss 无梯度。
        # 需在 gradient context 中重新调用 mlp_elev，使其获得真实梯度信号。
        loss_e_sum = torch.zeros(1, device=device).squeeze()
        elev_count = 0
        for n in list(tree.nodes.values()):
            if n.is_leaf() or not n.children_ids:
                continue
            child_embeds, child_ws = [], []
            for cid in n.children_ids:
                if cid not in tree.nodes:
                    continue
                child = tree.nodes[cid]
                emb   = child.z_v if child.is_leaf() else child.s
                if emb is not None:
                    child_embeds.append(emb.float())
                    child_ws.append(child.w)
            if not child_embeds:
                continue
            wt     = torch.tensor(child_ws, dtype=torch.float, device=device)
            wt     = wt / wt.sum().clamp(min=1e-6)
            z_pool = (torch.stack([e.to(device) for e in child_embeds]) * wt.unsqueeze(1)).sum(0)
            # 在梯度上下文中重新计算 s_abs，使 mlp_elev 获得真实梯度
            s_abs_recomputed = self.mlp_elev(z_pool.float())
            loss_e_sum = loss_e_sum + l_elev(
                s_abs_recomputed,
                [e.to(device) for e in child_embeds],
                child_ws,
            )
            elev_count += 1
        loss_e = loss_e_sum / max(elev_count, 1)

        total = loss_b + 0.5 * loss_s + 0.2 * loss_e
        return {
            "L_boundary": loss_b,
            "L_sem":      loss_s,
            "L_elev":     loss_e,
            "total":      total,
        }

    @torch.no_grad()
    def inference(
        self,
        images: list,
        image_mask: torch.Tensor,
        prompt: str,
        state: torch.Tensor,
        batch_idx: int = 0,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        单步推理。

        Returns
        -------
        action : (1, horizon, d_a)
        """
        fused_hidden, _, P_t, _, tree, z_v_feat = self._embed_with_dual_tree(
            images, image_mask, prompt, batch_idx=batch_idx
        )
        vl_feat = fused_hidden[:, 0, :]   # (1, d_vit)
        action  = self.backbone.predict_action(
            vl_feat, state, action_mask=action_mask
        )

        # ── 更新 HMT ────────────────────────────────────────────────
        # z_v_feat: (tiles, d_vit=896) — mlp1 投影空间，与 MLPElevation(d=896) 匹配
        z_v_mean = z_v_feat.mean(dim=0).detach().cpu()  # (d_vit=896,)
        a_scalar  = action.detach().cpu()                       # keep on CPU for HMT
        if a_scalar.ndim == 3:
            a_scalar = a_scalar[0, 0]   # first step, first batch
        elif a_scalar.ndim == 2:
            a_scalar = a_scalar[0]

        # JumpAwareHead 判定边界
        if tree.active_id is not None:
            active_node = tree.nodes[tree.active_id]
            if active_node.a_hist:
                A_hist_t = torch.stack(active_node.a_hist[-64:]).unsqueeze(0).to(vl_feat.device)
                a_new_t  = a_scalar.unsqueeze(0).to(vl_feat.device)
                p_jump, _ = self.jump_head(A_hist_t, a_new_t)
                force_branch = (p_jump.item() >= 0.5)
            else:
                force_branch = False
        else:
            force_branch = False

        if force_branch:
            s_current = self._get_top_abstract_nodes(tree)
            branch(tree, z_v_mean, a_scalar, s_current)
            _elev_dev = next(self.mlp_elev.parameters()).device
            propagate_elevation_to_root(
                tree, tree.active_id, self.mlp_elev,
                device=_elev_dev
            )
        else:
            merge(tree, z_v_mean, a_scalar)

        return action

    # ── 参数冻结便利方法 ───────────────────────────────────────────── #

    def freeze_backbone_for_pretrain(self):
        """预训练：骨架全冻结。"""
        self.freeze_backbone(freeze_llm=True, freeze_vit=True)

    def set_phase2_lr(self, optimizer, backbone_lr_scale: float = 0.1):
        """Phase 2：骨架 LLM 使用 0.1× 学习率。"""
        for pg in optimizer.param_groups:
            if pg.get("is_backbone_llm", False):
                pg["lr"] = pg.get("base_lr", pg["lr"]) * backbone_lr_scale
