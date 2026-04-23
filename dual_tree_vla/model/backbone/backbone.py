"""
InternVL3Backbone — EVO1 的自包含替代实现

消除对外部 Evo-1 包的依赖，将 InternVL3Embedder + FlowMatchingActionHead
封装为与 EVO1 接口完全兼容的主干网络。

用法
----
    from dual_tree_vla.model.backbone import InternVL3Backbone

    backbone = InternVL3Backbone(config={
        "vlm_name":  "model_weights/InternVL3-1B",
        "device":    "cuda",
        "action_horizon": 16,
        "per_action_dim": 7,
        "embed_dim":  896,      # VLM 输出维度 / action head context dim
        "state_dim":  7,
        "num_inference_timesteps": 50,
    })

    # 训练
    loss, _ = backbone.predict_action(vl_feat, state, actions_gt=a_gt)

    # 推理
    action = backbone.predict_action(vl_feat, state, action_mask=mask)
    # action: (B, H_a, d_a)
"""
from __future__ import annotations

from typing import List, Tuple, Union

from PIL import Image
import torch
import torch.nn as nn

from .internvl3_embedder import InternVL3Embedder
from ..action_head.flow_matching import FlowMatchingActionHead


class InternVL3Backbone(nn.Module):
    """
    EVO1 的等价替换，只依赖 DualTreeVLA 内部代码。

    Parameters
    ----------
    config : dict
        支持以下键（均可省略，有合理默认值）：

        vlm_name / vlm_path      : VLM 本地路径或 HF model ID（必填之一）
        device                   : 推理设备，默认 "cuda"
        action_horizon / H_a     : 动作预测步数，默认 16
        per_action_dim / d_a     : 单步动作维度，默认 7
        embed_dim / d_vit        : VLM 隐藏维度（= action head context dim），默认 896
        d_model                  : FlowMatchingActionHead 内部维度，默认 512
        n_heads                  : 注意力头数，默认 8
        n_layers                 : FlowBlock 层数，默认 6
        num_inference_timesteps  : 推理 ODE 步数，默认 50
        state_dim                : 状态维度（当前版本备用），默认 7
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = dict(config)

        # ── 基础参数 ─────────────────────────────────────────────────
        device    = config.get("device", "cuda")
        self._device = device
        self.return_cls_only = config.get("return_cls_only", True)

        vlm_name  = config.get("vlm_name") or config.get("vlm_path")
        if not vlm_name:
            raise ValueError(
                "InternVL3Backbone: config 必须含 'vlm_name' 或 'vlm_path' 键，"
                "指向 InternVL3 模型目录。"
            )

        self.horizon       = config.get("action_horizon", config.get("H_a", 16))
        self.per_action_dim = config.get("per_action_dim", config.get("d_a", 7))
        d_ctx              = config.get("embed_dim",  config.get("d_vit", 896))
        d_model            = config.get("d_model", 512)
        n_heads            = config.get("n_heads",  8)
        n_layers           = config.get("n_layers", 6)
        N_ode              = config.get("num_inference_timesteps", 50)

        # ── 子模块 ───────────────────────────────────────────────────
        self.embedder = InternVL3Embedder(
            model_name=vlm_name,
            device=device,
        )

        self.action_head = FlowMatchingActionHead(
            d_a     = self.per_action_dim,
            H_a     = self.horizon,
            d_model = d_model,
            n_layers= n_layers,
            n_heads = n_heads,
            d_ctx   = d_ctx,
            N_ode   = N_ode,
        ).to(device)

    def get_vl_embeddings(
        self,
        images: List[Union[Image.Image, torch.Tensor]],
        image_mask: torch.Tensor,
        prompt: str = "",
        return_cls_only: Union[bool, None] = None,
    ) -> torch.Tensor:
        if return_cls_only is None:
            return_cls_only = self.return_cls_only
        if images is None or len(images) == 0:
            raise ValueError("Must provide at least one image.")
        return self.embedder.get_fused_image_text_embedding_from_tensor_images(
            image_tensors=images,
            image_mask=image_mask,
            text_prompt=prompt,
            return_cls_only=return_cls_only,
        )

    def prepare_state(self, state_input: Union[list, torch.Tensor]) -> torch.Tensor:
        if isinstance(state_input, list):
            state_tensor = torch.tensor(state_input)
        elif isinstance(state_input, torch.Tensor):
            state_tensor = state_input
        else:
            raise TypeError("Unsupported state input type")

        if state_tensor.ndim == 1:
            state_tensor = state_tensor.unsqueeze(0)
        return state_tensor.to(self._device)

    @torch.no_grad()
    def run_inference(
        self,
        images: List[Union[Image.Image, torch.Tensor]],
        image_mask: torch.Tensor,
        prompt: str,
        state_input: Union[list, torch.Tensor],
        return_cls_only: Union[bool, None] = None,
        action_mask: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:
        fused_tokens = self.get_vl_embeddings(
            images=images,
            image_mask=image_mask,
            prompt=prompt,
            return_cls_only=return_cls_only,
        )
        state_tensor = self.prepare_state(state_input)
        return self.predict_action(fused_tokens, state_tensor, action_mask=action_mask)

    # ---------------------------------------------------------------- #
    #  Action prediction (EVO1-compatible interface)                    #
    # ---------------------------------------------------------------- #

    def predict_action(
        self,
        fused_tokens: torch.Tensor,
        state: torch.Tensor,
        actions_gt: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
    ):
        """
        参数
        ----
        fused_tokens : (B, d_vit)   — VLM 输出的 CLS token（来自 evo1_adapter）
        state        : (B, d_q)     — 当前关节状态（暂不注入 action head）
        actions_gt   : (B, T, d_a)  — 训练时的 ground-truth 动作序列
        action_mask  : (B, d_a)     — 推理时各维度有效位（0=无效关节）

        返回
        ----
        训练 (actions_gt 不为 None): (loss_flow, None)
        推理 (actions_gt 为 None)  : action (B, H_a, d_a)
        """
        # 将 CLS token 变为 1-token 上下文序列 (B, 1, d_vit)
        ctx = fused_tokens.unsqueeze(1).to(self.action_head.a_in.weight.device)

        if actions_gt is not None:
            # ── 训练 ─────────────────────────────────────────────────
            a_gt = self._align_horizon(actions_gt)  # → (B, H_a, d_a)
            loss = self.action_head.flow_loss(a_gt, ctx)
            return loss, None

        else:
            # ── 推理 ─────────────────────────────────────────────────
            action = self.action_head.sample(ctx)   # (B, H_a, d_a)
            if action_mask is not None:
                # action_mask: (B, d_a) → broadcast to (B, H_a, d_a)
                mask = action_mask.unsqueeze(1).expand_as(action).to(
                    dtype=action.dtype, device=action.device
                )
                action = action * mask
            return action

    def forward(
        self,
        fused_tokens: torch.Tensor,
        state: torch.Tensor | None = None,
        actions_gt: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, None] | torch.Tensor:
        return self.predict_action(
            fused_tokens=fused_tokens,
            state=state,
            actions_gt=actions_gt,
            action_mask=action_mask,
        )

    # ---------------------------------------------------------------- #
    #  Helpers                                                          #
    # ---------------------------------------------------------------- #

    def _align_horizon(self, actions_gt: torch.Tensor) -> torch.Tensor:
        """
        对 actions_gt (B, T, d_a) 裁剪或填充至 (B, H_a, d_a)。

        - T > H_a : 取前 H_a 步
        - T < H_a : 用最后一帧重复填充
        - T = H_a : 原样返回
        """
        B, T, d_a = actions_gt.shape
        if T == self.horizon:
            return actions_gt
        if T > self.horizon:
            return actions_gt[:, : self.horizon, :]
        # T < H_a：用最后一帧填充
        pad = actions_gt[:, -1:, :].expand(B, self.horizon - T, d_a)
        return torch.cat([actions_gt, pad], dim=1)
