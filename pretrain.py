"""
DualTreeVLA 预训练脚本 — 语义边界结构学习

预训练阶段目标（RoboCerebra）:
  1. 训练 JumpAwareHead：纯动作突变 → 分支点检测（L_boundary）
  2. 检验分支点语义是否接近真实子任务描述（L_sem，InfoNCE）
  3. SGMTS 视觉扫描的语义引导能力同步优化

冻结: LLM backbone, CrossModalFusion, FlowMatchingActionHead
可训练: SGMTS, s_proj, JumpAwareHead, TreeSSMReadout, MLPElevation

损失（完全不包含 FlowMatching）:
  L_boundary  — 动作突变边界 BCE（自监督或 RoboCerebra 标注）
  L_sem       — 分支点语义 InfoNCE（需 RoboCerebra 子任务描述标注）

用法:
  单卡: python pretrain.py --config configs/pretrain.yaml
  多卡: accelerate launch --config_file configs/accelerate_zero2.yaml \\
            pretrain.py --config configs/pretrain.yaml
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    _ACCELERATE = True
except ImportError:
    _ACCELERATE = False

try:
    import wandb
    _WANDB = True
except ImportError:
    _WANDB = False

import sys
_project_root = Path(__file__).parent
sys.path.insert(0, str(_project_root))

# ── 主干网络（自包含，无外部 Evo-1 依赖）────────────────────────────
from dual_tree_vla.model.backbone import InternVL3Backbone as EVO1

from dual_tree_vla.adapter import DualTreeAdapter_Evo1
from dual_tree_vla.dataset import RoboCerebraDataset, robocerebra_collate

DualTreeVLA = DualTreeAdapter_Evo1  # 局部别名


# ================================================================
#  工具函数
# ================================================================

def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def is_main(accel=None) -> bool:
    if accel is not None:
        return accel.is_main_process
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return True


def log_msg(msg: str, accel=None):
    if is_main(accel):
        print(msg, flush=True)


# ================================================================
#  冻结策略 — 预训练阶段
# ================================================================

def freeze_for_pretrain(model: DualTreeAdapter_Evo1):
    """
    预训练：骨架全冻结。
    可训练: SGMTS, GateFusion, JumpAwareHead, TreeSSMReadout, MLPElevation, sem_proj, mem_proj
    """
    model.freeze_backbone(freeze_llm=True, freeze_vit=True)
    trainable_modules = [
        model.sgmts,
        model.gate_fuse,
        model.sem_proj,
        model.jump_head,
        model.tree_ssm,
        model.mlp_elev,
        model.mem_proj,
    ]
    for m in trainable_modules:
        for p in m.parameters():
            p.requires_grad = True
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    return n_train, n_total


def inspect_parameters(model: DualTreeAdapter_Evo1, accel=None):
    log_msg("\n── 参数统计 ──────────────────────", accel)
    embedder = model.backbone.embedder
    groups = {
        "VLM (ViT+LLM)":     embedder.model,
        "SGMTS encoder":     model.sgmts,
        "GateFusion":        model.gate_fuse,
        "sem_proj":          model.sem_proj,
        "JumpAwareHead":     model.jump_head,
        "TreeSSMReadout":    model.tree_ssm,
        "MLPElevation":      model.mlp_elev,
        "mem_proj":          model.mem_proj,
        "ActionHead":        model.backbone.action_head,
    }
    total_all, train_all = 0, 0
    for name, mod in groups.items():
        total = sum(p.numel() for p in mod.parameters())
        train = sum(p.numel() for p in mod.parameters() if p.requires_grad)
        status = "TRAIN" if train > 0 else "frozen"
        log_msg(f"  {name:<24}: {total/1e6:6.2f}M total  {train/1e6:6.2f}M trainable  [{status}]", accel)
        total_all += total
        train_all  += train
    log_msg(f"  {'─'*60}", accel)
    log_msg(f"  {'TOTAL':<24}: {total_all/1e6:6.2f}M total  {train_all/1e6:6.2f}M trainable", accel)
    log_msg("", accel)


# ================================================================
#  单步训练（一条轨迹）
# ================================================================

def pretrain_step(
    model: DualTreeAdapter_Evo1,
    batch: Dict,
    device: torch.device,
    loss_cfg: dict,
) -> Dict[str, torch.Tensor]:
    """
    调用 model(..., mode='pretrain')：仅计算 L_boundary + L_sem。
    """
    frames      = batch["frames"]                    # (B, T, C, H, W) 或 None
    actions     = batch["actions"].to(device)        # (B, T, d_a)
    states      = batch["states"].to(device)         # (B, d_q)

    B = actions.shape[0]
    unwrapped = model.module if hasattr(model, "module") else model
    unwrapped.reset(B)

    subtask_ids = batch.get("subtask_ids")
    if subtask_ids is not None:
        subtask_ids = subtask_ids.to(device)
    instructions: List[str] = batch["instructions"]

    # 缓存快速路径：将预提取特征移到 GPU
    precomputed_vit: Optional[Dict] = None
    if batch.get("P_t_raw") is not None:
        precomputed_vit = {
            "P_t_raw":  batch["P_t_raw"].to(device),   # (B, tiles, 1025, 1024)
            "z_v_feat": batch["z_v_feat"].to(device),  # (B, tiles, 896)
        }

    losses = model(
        images=frames if frames is not None else [None] * B,
        instructions=instructions,
        states=states,
        actions=actions,
        subtask_ids=subtask_ids,
        mode="pretrain",
        precomputed_vit=precomputed_vit,
    )

    return {
        "loss":       losses.get("total", torch.zeros((), device=device)),
        "L_boundary": losses.get("L_boundary", torch.zeros((), device=device)).detach(),
        "L_sem":      losses.get("L_sem",      torch.zeros((), device=device)).detach(),
        "L_elev":     losses.get("L_elev",     torch.zeros((), device=device)).detach(),
    }


# ================================================================
#  Checkpoint 保存
# ================================================================

def save_ckpt(model, optimizer, epoch, step, path: Path, accel=None):
    if accel is not None and hasattr(accel, "unwrap_model"):
        state = accel.unwrap_model(model).state_dict()
    else:
        state = model.state_dict()
    torch.save({"model": state, "optimizer": optimizer.state_dict(),
                "epoch": epoch, "step": step}, path)
    print(f"[pretrain] Saved checkpoint → {path}", flush=True)


# ================================================================
#  Pretrain Eval（保存记忆树 JSON + 视觉树 heatmap）
# ================================================================

def _run_pretrain_eval(
    model,
    dataset,
    epoch: int,
    device: torch.device,
    cfg: dict,
    accel=None,
    n_eval_samples: int = None,
):
    """
    在训练中间评估：
      - 对 n_eval_samples 条轨迹运行双树模块
      - 记忆树最终状态 → result/trees/ep{epoch}/{traj_id}.json
      - 每帧语义重要性 → result/heatmaps/ep{epoch}/{traj_id}/frame_{t}.png

    由主进程独占执行（is_main 已在外部保证）。
    """
    from dual_tree_vla.model.memory_tree.operations import (
        semantic_elevation, propagate_elevation_to_root,
    )

    tc        = cfg.get("train", {})
    result_dir = Path(tc.get("result_dir", "results/pretrain_eval"))
    heatmap_step = tc.get("eval_heatmap_step", 4)   # 每 4 帧保存一张 heatmap
    n_eval     = n_eval_samples or tc.get("eval_samples", 3)
    n_eval     = min(n_eval, len(dataset))

    tree_dir    = result_dir / f"ep{epoch:03d}" / "trees"
    heatmap_root = result_dir / f"ep{epoch:03d}" / "heatmaps"
    tree_dir.mkdir(parents=True, exist_ok=True)

    # 切换为 eval 模式
    raw_model = accel.unwrap_model(model) if (accel is not None and hasattr(accel, "unwrap_model")) else model
    raw_model.eval()

    try:
        import cv2 as _cv2; _CV2_OK = True
    except ImportError:
        _CV2_OK = False
    try:
        from PIL import Image as _PilImage; _PIL_OK = True
    except ImportError:
        _PIL_OK = False

    all_metrics: List[Dict] = []

    with torch.no_grad():
        for idx in range(n_eval):
            sample      = dataset[idx]
            traj_id     = f"traj_{idx:04d}"
            frames      = sample["frames"]      # (T, C, H, W)
            actions     = sample["actions"]     # (T, d_a)
            instruction = sample["instruction"]
            boundary_gt = sample.get("boundary_mask")

            T    = frames.shape[0]
            import math as _math
            N_p  = 256   # post-pixel_shuffle LLM token 数（16×16）

            raw_model.reset(batch_size=1)
            tree = raw_model.get_tree(0)

            traj_heatmap_dir = heatmap_root / traj_id
            traj_heatmap_dir.mkdir(parents=True, exist_ok=True)

            all_logits:     List[float] = []
            all_gt:         List[int]   = []
            branch_frames:  List[int]   = []

            for t in range(T):
                img_t = frames[t].unsqueeze(0).to(device)
                act_t = actions[t].to(device)

                embedder    = raw_model.backbone.embedder
                pixel_values, num_tiles_list = embedder._preprocess_images([img_t.squeeze(0)])
                _   = embedder.model.extract_feature(pixel_values)
                P_t = raw_model._P_t_raw[:, 1:, :].to(torch.float32)

                g_task   = raw_model._encode_task(instruction, device)
                s_top_v  = raw_model._get_top_abstract_nodes(tree)
                tot_tiles = P_t.shape[0]
                g_t_exp   = g_task.expand(tot_tiles, -1)
                s_top_lst = [s_top_v] * tot_tiles

                Z_v, sigma_maps = raw_model.sgmts(
                    P_t, g_t_exp, s_top_lst, return_attn=True
                )
                sigma_np = sigma_maps[0].cpu().float().numpy()

                # 保存 heatmap
                if t % heatmap_step == 0:
                    frame_rgb = (frames[t].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
                    H_img, W_img = frame_rgb.shape[:2]
                    _nH = _nW = int(_math.isqrt(sigma_np.shape[0]))  # SGMTS 在原始 ViT patch 上运行
                    hmap_small   = sigma_np.reshape(_nH, _nW).astype(np.float32)
                    if _CV2_OK:
                        hmap_up = _cv2.resize(hmap_small, (W_img, H_img), interpolation=_cv2.INTER_LINEAR)
                        hmin, hmax = hmap_up.min(), hmap_up.max()
                        hu8 = ((hmap_up - hmin) / (hmax - hmin + 1e-6) * 255).astype(np.uint8)
                        colored = _cv2.applyColorMap(hu8, _cv2.COLORMAP_JET)
                        frame_bgr = _cv2.cvtColor(frame_rgb, _cv2.COLOR_RGB2BGR)
                        overlay   = _cv2.addWeighted(frame_bgr, 0.5, colored, 0.5, 0)
                        _cv2.imwrite(str(traj_heatmap_dir / f"frame_{t:04d}.png"), overlay)
                    elif _PIL_OK:
                        _PilImage.fromarray(frame_rgb).save(
                            str(traj_heatmap_dir / f"frame_{t:04d}_orig.png")
                        )
                        np.save(str(traj_heatmap_dir / f"sigma_{t:04d}.npy"), sigma_np)

                # JumpAwareHead
                active_node = tree.nodes.get(tree.active_id) if tree.active_id else None
                if active_node and active_node.a_hist:
                    A_act = torch.stack(active_node.a_hist[-raw_model.jump_head.max_len:]).unsqueeze(0).to(device)
                else:
                    A_act = act_t.new_zeros(1, 1, raw_model.d_a)
                p_jump, logit = raw_model.jump_head(A_act, act_t.unsqueeze(0))
                force_branch  = bool(p_jump.item() >= 0.5)
                all_logits.append(float(logit.item()))
                if boundary_gt is not None:
                    all_gt.append(int(boundary_gt[t].item()))

                # ── 更新 HMT（使用 mlp1 投影后的 896-dim 特征）────────────────
                # P_t 是 ViT 原始 1024-dim，需要先做 pixel_shuffle + mlp1 得到 896-dim
                _T, _N, _dp = P_t.shape                 # (_T, 1024, 1024) 或 (_T, N, 1024)
                _h = _w = int(_math.isqrt(_N))
                _V_ps = P_t.reshape(_T, _h, _w, _dp)
                _V_ps = embedder.model.pixel_shuffle(
                    _V_ps.to(torch.bfloat16), scale_factor=embedder.model.downsample_ratio
                )
                _V_ps = _V_ps.reshape(_T, -1, _V_ps.shape[-1])
                _vit_proj = embedder.model.mlp1(_V_ps).float()  # (_T, 256, 896)
                z_v_c = _vit_proj.mean(dim=0).mean(dim=0).cpu()  # (896,)
                s_cur = raw_model._mlp_elev_cpu(z_v_c)
                tree.insert(z_v=z_v_c, a=act_t.cpu(), force_branch=force_branch, s_current=s_cur)
                if force_branch:
                    branch_frames.append(t)
                    pend = tree.elevation_pending_parent
                    if pend is not None:
                        _elev_dev = next(raw_model.mlp_elev.parameters()).device
                        semantic_elevation(tree, pend, raw_model.mlp_elev, device=_elev_dev)
                        propagate_elevation_to_root(tree, pend, raw_model.mlp_elev, device=_elev_dev)

            # 保存记忆树 JSON
            tree_json = tree.to_json_dict()
            tree_json.update({
                "traj_id": traj_id,
                "instruction": instruction,
                "T": T,
                "branch_frames": branch_frames,
                "subtask_ids": sample["subtask_ids"].tolist(),
                "epoch": epoch,
            })
            with open(tree_dir / f"{traj_id}.json", "w", encoding="utf-8") as fp:
                json.dump(tree_json, fp, ensure_ascii=False, indent=2)

            # 指标
            m: Dict = {"traj_id": traj_id, "n_branches": len(branch_frames)}
            if all_gt and all_logits:
                preds  = [int(l >= 0.0) for l in all_logits]
                tp     = sum(p == 1 and g == 1 for p, g in zip(preds, all_gt))
                fp     = sum(p == 1 and g == 0 for p, g in zip(preds, all_gt))
                fn     = sum(p == 0 and g == 1 for p, g in zip(preds, all_gt))
                prec   = tp / max(tp + fp, 1)
                rec    = tp / max(tp + fn, 1)
                m["boundary_f1"] = round(2 * prec * rec / max(prec + rec, 1e-6), 4)
            all_metrics.append(m)

    # 汇总
    summary_path = result_dir / f"ep{epoch:03d}" / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as fp:
        json.dump({"epoch": epoch, "n_trajs": n_eval, "per_traj": all_metrics}, fp, indent=2)

    f1_vals = [m.get("boundary_f1") for m in all_metrics if "boundary_f1" in m]
    f1_mean  = round(float(np.mean(f1_vals)), 4) if f1_vals else None
    print(f"[pretrain eval] ep={epoch}  boundary_F1_mean={f1_mean}  "
          f"trees→{tree_dir}  heatmaps→{heatmap_root}", flush=True)

    # 恢复训练模式
    raw_model.train()


# ================================================================
#  主训练循环
# ================================================================

def train(cfg: dict):
    # ── Accelerator ─────────────────────────────────────────────────
    if _ACCELERATE:
        from accelerate import DistributedDataParallelKwargs
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accel = Accelerator(
            mixed_precision=cfg.get("mixed_precision", "bf16"),
            gradient_accumulation_steps=cfg.get("grad_accum", 1),
            kwargs_handlers=[ddp_kwargs],
        )
        device = accel.device
    else:
        accel  = None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if is_main(accel):
        if _ACCELERATE:
            set_seed(cfg.get("seed", 42))

    log_msg(f"[pretrain] device={device}  mixed_precision={cfg.get('mixed_precision', 'bf16')}", accel)

    # ── Dataset ──────────────────────────────────────────────────────
    data_cfg = cfg["data"]
    dataset  = RoboCerebraDataset(
        root       = data_cfg["root"],
        scenes     = data_cfg.get("scenes"),
        img_h      = data_cfg.get("img_h", 224),
        img_w      = data_cfg.get("img_w", 224),
        subsample  = data_cfg.get("subsample", 4),
        max_seqlen = data_cfg.get("max_seqlen", 64),
        feat_cache_dir = data_cfg.get("feat_cache_dir"),  # None 则回落到视频加载
    )
    _cache_mode = data_cfg.get("feat_cache_dir") is not None
    log_msg(f"[pretrain] Dataset: {len(dataset)} 条轨迹  {'[缓存快速路径 ✓]' if _cache_mode else '[视频读帧模式]'}", accel)

    loader = DataLoader(
        dataset,
        batch_size  = cfg["train"]["batch_size"],
        shuffle     = True,
        num_workers = cfg["train"].get("num_workers", 4),
        collate_fn  = robocerebra_collate,
        pin_memory  = True,
        drop_last   = True,
    )

    # ── 构建模型 ────────────────────────────────────────────────────
    assert EVO1 is not None, "InternVL3Backbone 未能导入，请检查 dual_tree_vla 包是否正确安装。"
    mc = cfg["model"]
    _vlm_path = mc.get("vlm_path") or mc.get("llm_path")
    if not _vlm_path:
        raise ValueError("config[model] 必须含 'vlm_path' 键，指向 InternVL3 模型目录。")
    backbone = EVO1(config={
        "vlm_name":           _vlm_path,
        "device":             str(device),
        "action_horizon":     mc.get("H_a", 16),
        "per_action_dim":     mc.get("d_a", 7),
        "embed_dim":          mc.get("d_vit", 896),
        "state_dim":          mc.get("d_q", 7),
        "num_inference_timesteps": mc.get("n_ode", 50),
    })
    model = DualTreeAdapter_Evo1(
        backbone       = backbone,
        d_vit          = mc.get("d_vit", 896),
        d_a            = mc.get("d_a", 7),
        d_ssm          = mc.get("d_ssm", 256),
        d_state        = mc.get("d_state", 16),
        mount_tau      = mc.get("mount_tau", 0.4),
        max_tree_depth = mc.get("max_tree_depth", 4),
        alpha          = mc.get("alpha", 0.5),
        delta_w        = mc.get("delta_w", 0.1),
    )

    n_train, n_total = freeze_for_pretrain(model)
    log_msg(f"[pretrain] 可训练参数: {n_train:,} / {n_total:,}", accel)
    inspect_parameters(model, accel)

    # 可选：从已有断点恢复
    tc = cfg["train"]
    resume = tc.get("resume_from")
    if resume and os.path.isfile(str(resume)):
        ckpt = torch.load(resume, map_location="cpu")
        missing, unexp = model.load_state_dict(ckpt["model"], strict=False)
        log_msg(f"[pretrain] 恢复自 {resume}  missing={len(missing)}  unexpected={len(unexp)}", accel)

    # ── 优化器（参考 Evo-1 train.py 风格）────────────────────────────
    lr = float(tc.get("lr", 3e-4))
    wd = float(tc.get("weight_decay", 1e-4))

    decay_p, no_decay_p = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(k in name for k in ("norm", "bias", "A_log")):
            no_decay_p.append(p)
        else:
            decay_p.append(p)

    optimizer = torch.optim.AdamW(
        [{"params": decay_p, "weight_decay": wd},
         {"params": no_decay_p, "weight_decay": 0.0}],
        lr=lr,
    )

    total_steps  = tc["epochs"] * len(loader)
    warmup_steps = int(tc.get("warmup_ratio", 0.05) * total_steps)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        prog = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.01, 0.5 * (1.0 + math.cos(math.pi * prog)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Accelerate 包装 ──────────────────────────────────────────────
    if _ACCELERATE and accel is not None:
        model, optimizer, loader, scheduler = accel.prepare(
            model, optimizer, loader, scheduler
        )

    model.train()

    # ── W&B 初始化 ────────────────────────────────────────────────────
    if _WANDB and is_main(accel):
        wandb.init(
            project=cfg.get("wandb_project", "DualTreeVLA-pretrain"),
            name=cfg.get("wandb_run",    f"pretrain_{int(time.time())}"),
            config=cfg,
            mode="offline",
        )

    # ── 训练循环 ─────────────────────────────────────────────────────
    ckpt_dir    = Path(tc["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_every  = tc.get("save_every", 5)
    global_step = 0
    best_loss   = float("inf")
    loss_cfg    = cfg.get("loss", {})

    for epoch in tqdm(range(1, tc["epochs"] + 1), desc="[pretrain] epochs", unit="ep", disable=not is_main(accel)):
        epoch_sums: Dict[str, float] = {}

        pbar = tqdm(loader, desc=f"  ep {epoch:03d}", leave=False, unit="batch", disable=not is_main(accel))
        for batch in pbar:
            if not _ACCELERATE:
                batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                         for k, v in batch.items()}

            step_losses = pretrain_step(model, batch, device, loss_cfg)
            loss = step_losses["loss"]

            if _ACCELERATE and accel is not None:
                accel.backward(loss)
                accel.clip_grad_norm_(model.parameters(), 1.0)
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            for k, v in step_losses.items():
                epoch_sums[k] = epoch_sums.get(k, 0.0) + (
                    v.item() if isinstance(v, torch.Tensor) else float(v))

            if is_main(accel):
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "L_b":  f"{step_losses.get('L_boundary', 0):.4f}" if hasattr(step_losses.get('L_boundary', 0), 'item') else f"{float(step_losses.get('L_boundary', 0)):.4f}",
                    "L_s":  f"{step_losses.get('L_sem', 0):.4f}" if hasattr(step_losses.get('L_sem', 0), 'item') else f"{float(step_losses.get('L_sem', 0)):.4f}",
                    "lr":   f"{scheduler.get_last_lr()[0]:.1e}",
                })

            if global_step % 50 == 0 and is_main(accel):
                lr_now = scheduler.get_last_lr()[0]
                detail = "  ".join(f"{k}={v:.4f}" for k, v in step_losses.items() if k != "loss")
                log_msg(
                    f"[pretrain] ep={epoch}/{tc['epochs']}  step={global_step}"
                    f"  loss={loss.item():.4f}  lr={lr_now:.2e}  {detail}",
                    accel,
                )
                if _WANDB:
                    wandb.log({"train/"+k: (v.item() if isinstance(v, torch.Tensor) else v)
                               for k, v in step_losses.items()}, step=global_step)
                    wandb.log({"lr": lr_now}, step=global_step)

        n_batch  = max(len(loader), 1)
        avg_loss = epoch_sums.get("loss", 0.0) / n_batch
        avg_detail = "  ".join(
            f"{k}={epoch_sums.get(k,0.0)/n_batch:.4f}"
            for k in ("L_boundary", "L_sem", "L_elev")
        )
        log_msg(f"[pretrain] Epoch {epoch}/{tc['epochs']}  avg_loss={avg_loss:.4f}  {avg_detail}", accel)

        if is_main(accel) and epoch % save_every == 0:
            save_ckpt(model, optimizer, epoch, global_step,
                      ckpt_dir / f"pretrain_ep{epoch:03d}.pt", accel)

        if is_main(accel) and avg_loss < best_loss:
            best_loss = avg_loss
            save_ckpt(model, optimizer, epoch, global_step,
                      ckpt_dir / "pretrain_best.pt", accel)

        # ── Eval：每 eval_every epoch 保存记忆树 JSON + 视觉树 heatmap ──
        eval_every = tc.get("eval_every", 5)
        if is_main(accel) and eval_every > 0 and epoch % eval_every == 0:
            _run_pretrain_eval(
                model       = model,
                dataset     = dataset,
                epoch       = epoch,
                device      = device,
                cfg         = cfg,
                accel       = accel,
            )

    log_msg("[pretrain] 完成。", accel)


# ================================================================
#  入口
# ================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to pretrain.yaml")
    args = parser.parse_args()
    cfg  = load_cfg(args.config)
    train(cfg)


if __name__ == "__main__":
    main()

