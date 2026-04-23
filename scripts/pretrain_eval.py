"""
pretrain_eval.py — 预训练模型评估脚本

对 RoboCerebra 数据集的每条轨迹运行预训练好的双树模块，输出：

  1. 记忆树（HMT）JSON：
       result/trees/{traj_id}.json
     包含树的拓扑结构、每个节点的类型/权重/动作统计

  2. 视觉树（SGMTS）语义重要性 heatmap（蒙版形式叠加在原图上）：
       result/heatmaps/{traj_id}/frame_{t:04d}.png
     每帧原图 + σ 热力图叠加（Jet colormap，alpha=0.5）

用法:
  python scripts/pretrain_eval.py \\
      --config dual_tree_vla/config/pretrain.yaml \\
      --ckpt   data/outputs/pretrain_best.pt \\
      --out    results/eval_pretrain \\
      [--max_traj 20]  [--device cuda]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── 自动推断 Evo-1 路径（常见位置）─────────────────────────────────
_project_root = Path(__file__).parent.parent
_evo1_candidates = [
    _project_root.parent / "Evo-1" / "Evo-1",
    _project_root.parent / "evo-1" / "evo-1",
    _project_root / "Evo-1" / "Evo-1",
    _project_root / "third_party" / "Evo-1" / "Evo-1",
]
for _p in _evo1_candidates:
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
        break

# ------------------------------------------------------------------ #
#  可选依赖                                                              #
# ------------------------------------------------------------------ #
try:
    import cv2 as _cv2
    _CV2_OK = True
except ImportError:
    _CV2_OK = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    _MPL_OK = True
except ImportError:
    _MPL_OK = False

try:
    from PIL import Image as PilImage
    _PIL_OK = True
except ImportError:
    _PIL_OK = False

from dual_tree_vla.adapter import DualTreeAdapter_Evo1
from dual_tree_vla.dataset import RoboCerebraDataset, robocerebra_collate
from dual_tree_vla.model.memory_tree.operations import (
    semantic_elevation,
    propagate_elevation_to_root,
)
from dual_tree_vla.losses.tree_losses import l_boundary, l_sem


# ================================================================= #
#  Heatmap 工具                                                       #
# ================================================================= #

def sigma_to_heatmap_overlay(
    frame_hwc: np.ndarray,
    sigma: np.ndarray,
    nH: int,
    nW: int,
    alpha: float = 0.50,
    colormap: int = None,
) -> np.ndarray:
    """
    将语义重要性分数 σ (N_p,) 渲染为与原图相同分辨率的热力图，
    并以半透明方式叠加到原图上（BGR）。

    Parameters
    ----------
    frame_hwc : (H, W, 3) uint8 RGB
    sigma     : (N_p,) float, 已归一化到 [0,1]
    nH, nW    : patch grid 尺寸
    alpha     : 热力图不透明度（0=仅原图，1=仅热力图）
    colormap  : cv2 colormap id（默认 COLORMAP_JET）

    Returns
    -------
    overlay : (H, W, 3) uint8 BGR
    """
    if colormap is None and _CV2_OK:
        colormap = _cv2.COLORMAP_JET
    H, W = frame_hwc.shape[:2]

    # σ → (nH, nW) → resize 到原图分辨率
    heatmap_small = sigma.reshape(nH, nW).astype(np.float32)
    if _CV2_OK:
        heatmap_resized = _cv2.resize(
            heatmap_small, (W, H), interpolation=_cv2.INTER_LINEAR
        )
        # 归一化到 [0, 255]
        hmin, hmax = heatmap_resized.min(), heatmap_resized.max()
        if hmax - hmin > 1e-6:
            heatmap_u8 = ((heatmap_resized - hmin) / (hmax - hmin) * 255).astype(np.uint8)
        else:
            heatmap_u8 = np.zeros((H, W), dtype=np.uint8)
        colored = _cv2.applyColorMap(heatmap_u8, colormap)  # (H,W,3) BGR

        # 原图 RGB → BGR
        frame_bgr = _cv2.cvtColor(frame_hwc, _cv2.COLOR_RGB2BGR)
        overlay   = _cv2.addWeighted(frame_bgr, 1 - alpha, colored, alpha, 0)
    elif _MPL_OK:
        # fallback: matplotlib
        heatmap_resized = np.array(
            PilImage.fromarray(heatmap_small).resize((W, H), PilImage.BILINEAR)
        ) if _PIL_OK else heatmap_small
        hmin, hmax = heatmap_resized.min(), heatmap_resized.max()
        norm = (heatmap_resized - hmin) / (hmax - hmin + 1e-6)
        cmap  = cm.get_cmap("jet")
        colored_rgba = (cmap(norm) * 255).astype(np.uint8)[:, :, :3]  # RGB
        overlay_rgb  = (
            (1 - alpha) * frame_hwc.astype(np.float32)
            + alpha * colored_rgba.astype(np.float32)
        ).clip(0, 255).astype(np.uint8)
        # 转 BGR 以便 cv2 / PIL 兼容保存
        overlay = overlay_rgb[:, :, ::-1]
    else:
        overlay = frame_hwc[:, :, ::-1].copy()  # 原图 BGR

    return overlay


def save_heatmap_frame(
    save_path: str,
    frame_hwc: np.ndarray,
    sigma_np: np.ndarray,
    nH: int,
    nW: int,
):
    """保存单帧热力图叠加。"""
    overlay_bgr = sigma_to_heatmap_overlay(frame_hwc, sigma_np, nH, nW)
    if _CV2_OK:
        _cv2.imwrite(save_path, overlay_bgr)
    elif _PIL_OK:
        # BGR → RGB 然后用 PIL 保存
        PilImage.fromarray(overlay_bgr[:, :, ::-1]).save(save_path)
    else:
        # 最后兼容：用 numpy 保存为 npy
        np.save(save_path.replace(".png", ".npy"), sigma_np)


# ================================================================= #
#  单轨迹 Eval                                                        #
# ================================================================= #

def eval_trajectory(
    model: DualTreeAdapter_Evo1,
    sample: Dict,
    traj_id: str,
    out_dir: Path,
    device: torch.device,
    save_heatmaps: bool = True,
    heatmap_step: int = 1,
) -> Dict:
    """
    对一条轨迹运行预训练 eval：
      - 逐帧调用双树模块
      - 收集记忆树最终状态 → JSON
      - 收集每帧 σ → heatmap PNG

    Parameters
    ----------
    model        : DualTreeAdapter_Evo1（已加载预训练权重）
    sample       : RoboCerebraDataset 单条轨迹 dict
    traj_id      : 轨迹标识符字符串
    out_dir      : 输出根目录（result/）
    device       : torch.device
    save_heatmaps: 是否保存 heatmap
    heatmap_step : 每隔多少帧保存一次 heatmap（节约磁盘）

    Returns
    -------
    metrics : dict  包含边界检测统计
    """
    frames      = sample["frames"]       # (T, C, H, W) float32 [0,1]
    actions     = sample["actions"]      # (T, d_a)
    states      = sample["states"]       # (T, d_q)
    instruction = sample["instruction"]
    subtask_ids = sample["subtask_ids"]  # (T,)
    boundary_gt = sample.get("boundary_mask")  # (T,)  optional

    T = frames.shape[0]
    _, C, H, W = frames.shape
    import math
    # nH/nW 延迟计算：SGMTS 在原始 ViT patch（如 1024=32×32）上运行，
    # 不能用 N_p=256（pixel_shuffle 后的 token 数）

    # ── 初始化记忆树 ─────────────────────────────────────────────────
    model.reset(batch_size=1)
    tree = model.get_tree(0)

    # ── 输出目录 ────────────────────────────────────────────────────
    tree_dir    = out_dir / "trees"
    heatmap_dir = out_dir / "heatmaps" / traj_id
    tree_dir.mkdir(parents=True, exist_ok=True)
    if save_heatmaps:
        heatmap_dir.mkdir(parents=True, exist_ok=True)

    all_logits:     List[float] = []
    all_boundary_gt: List[int]   = []
    branch_frames:  List[int]   = []

    with torch.no_grad():
        for t in range(T):
            img_t   = frames[t].unsqueeze(0).to(device)    # (1, C, H, W)
            act_t   = actions[t].to(device)                # (d_a,)

            # ── 调用 SGMTS + GateFusion（带 sigma 输出）──────────────
            embedder = model.backbone.embedder
            pixel_values, num_tiles_list = embedder._preprocess_images(
                [img_t.squeeze(0)]
            )
            _ = embedder.model.extract_feature(pixel_values)   # 触发 ViT hook
            P_t = model._P_t_raw[:, 1:, :].to(torch.float32)  # (tiles, N_p, d_vit)

            g_task    = model._encode_task(instruction, device)         # (1, d_vit)
            s_top_val = model._get_top_abstract_nodes(tree)             # (d_vit,) or None

            total_tiles = P_t.shape[0]
            g_task_t  = g_task.expand(total_tiles, -1)
            s_top_lst = [s_top_val] * total_tiles

            # SGMTS with sigma
            Z_v, sigma_maps = model.sgmts(
                P_t, g_task_t, s_top_lst, return_attn=True
            )    # sigma_maps: list of (N_p,) tensors

            # 取第一个 tile 的 sigma（单张图像）
            sigma_np = sigma_maps[0].cpu().float().numpy()
            # 动态推导 patch grid 尺寸（SGMTS 在原始 ViT patch 上运行，如 1024=32×32）
            _nH = _nW = int(math.isqrt(sigma_np.shape[0]))

            # ── 保存 heatmap ──────────────────────────────────────────
            if save_heatmaps and t % heatmap_step == 0:
                # frame: (C, H, W) float [0,1] → (H, W, 3) uint8 RGB
                frame_rgb = (frames[t].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
                save_path = str(heatmap_dir / f"frame_{t:04d}.png")
                save_heatmap_frame(save_path, frame_rgb, sigma_np, _nH, _nW)

            # ── JumpAwareHead 推断跳变概率 ───────────────────────────
            active_node = tree.nodes.get(tree.active_id) if tree.active_id is not None else None
            if active_node is not None and active_node.a_hist:
                a_hist = active_node.a_hist[-model.jump_head.max_len:]
                A_act  = torch.stack(a_hist).unsqueeze(0).to(device)     # (1, L, d_a)
            else:
                A_act  = act_t.new_zeros(1, 1, model.d_a)
            a_new  = act_t.unsqueeze(0)                                   # (1, d_a)
            p_jump, logit = model.jump_head(A_act, a_new)                 # (1,), (1,)
            force_branch  = bool(p_jump.item() >= 0.5)

            all_logits.append(float(logit.item()))
            if boundary_gt is not None:
                all_boundary_gt.append(int(boundary_gt[t].item()))

            # ── 更新记忆树 ────────────────────────────────────────────
            # P_t 是 ViT 原始 1024-dim，需先做 pixel_shuffle + mlp1 得到 896-dim
            _T, _N, _dp = P_t.shape
            _h = _w = int(math.isqrt(_N))
            _V_ps = P_t.reshape(_T, _h, _w, _dp)
            _V_ps = embedder.model.pixel_shuffle(
                _V_ps.to(torch.bfloat16), scale_factor=embedder.model.downsample_ratio
            )
            _V_ps = _V_ps.reshape(_T, -1, _V_ps.shape[-1])
            _vit_proj = embedder.model.mlp1(_V_ps).float()   # (_T, 256, 896)
            z_v_mean = _vit_proj.mean(dim=0).mean(dim=0).cpu()  # (896,)
            s_cur    = model._mlp_elev_cpu(z_v_mean)

            tree.insert(
                z_v=z_v_mean,
                a=act_t.cpu(),
                force_branch=force_branch,
                s_current=s_cur,
            )

            # ── 语义提升 ─────────────────────────────────────────────
            if force_branch:
                branch_frames.append(t)
                pending = tree.elevation_pending_parent
                if pending is not None:
                    _elev_dev = next(model.mlp_elev.parameters()).device
                    semantic_elevation(tree, pending, model.mlp_elev, device=_elev_dev)
                    propagate_elevation_to_root(tree, pending, model.mlp_elev, device=_elev_dev)

    # ── 保存记忆树 JSON ──────────────────────────────────────────────
    tree_json = tree.to_json_dict()
    tree_json["traj_id"]      = traj_id
    tree_json["instruction"]  = instruction
    tree_json["T"]            = T
    tree_json["branch_frames"] = branch_frames
    tree_json["subtask_ids"]  = subtask_ids.tolist() if hasattr(subtask_ids, "tolist") else list(subtask_ids)

    tree_path = tree_dir / f"{traj_id}.json"
    with open(tree_path, "w", encoding="utf-8") as fp:
        json.dump(tree_json, fp, ensure_ascii=False, indent=2)

    # ── 边界检测指标 ─────────────────────────────────────────────────
    metrics: Dict = {"traj_id": traj_id, "T": T, "n_branches": len(branch_frames)}
    if all_boundary_gt and all_logits:
        import torch as _torch
        logits_t  = _torch.tensor(all_logits)
        labels_t  = _torch.tensor(all_boundary_gt, dtype=_torch.float32)
        preds     = (logits_t >= 0.0).long()
        tp = int(((preds == 1) & (labels_t == 1)).sum())
        fp = int(((preds == 1) & (labels_t == 0)).sum())
        fn = int(((preds == 0) & (labels_t == 1)).sum())
        precision = tp / max(tp + fp, 1)
        recall    = tp / max(tp + fn, 1)
        f1        = 2 * precision * recall / max(precision + recall, 1e-6)
        metrics.update({
            "boundary_precision": round(precision, 4),
            "boundary_recall":    round(recall, 4),
            "boundary_f1":        round(f1, 4),
            "n_gt_boundaries":    int(labels_t.sum()),
            "n_pred_boundaries":  int(preds.sum()),
        })

    return metrics


# ================================================================= #
#  主流程                                                              #
# ================================================================= #

def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="DualTreeVLA 预训练 Eval")
    parser.add_argument("--config",    required=True,  help="pretrain.yaml 路径")
    parser.add_argument("--ckpt",      required=True,  help="预训练权重 .pt 路径")
    parser.add_argument("--out",       default="results/eval_pretrain", help="输出根目录")
    parser.add_argument("--max_traj",  type=int, default=None, help="最多评估多少条轨迹")
    parser.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--heatmap_step", type=int, default=1, help="每隔多少帧保存一次 heatmap")
    parser.add_argument("--no_heatmap",   action="store_true", help="跳过 heatmap 生成")
    args = parser.parse_args()

    cfg    = load_cfg(args.config)
    device = torch.device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[eval] 输出目录: {out_dir}")
    print(f"[eval] 设备:     {device}")

    # ── 构建模型 ─────────────────────────────────────────────────────
    mc = cfg["model"]
    _vlm_path = mc.get("vlm_path") or mc.get("llm_path")
    from dual_tree_vla.model.backbone import InternVL3Backbone as EVO1
    backbone = EVO1(config={
        "vlm_name":                _vlm_path,
        "device":                  str(device),
        "action_horizon":          mc.get("H_a", 16),
        "per_action_dim":          mc.get("d_a", 7),
        "embed_dim":               mc.get("d_vit", 896),
        "state_dim":               mc.get("d_q", 7),
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

    # ── 加载预训练权重 ────────────────────────────────────────────────
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state_dict = ckpt.get("model", ckpt)
    model_sd   = model.state_dict()
    compatible = {k: v for k, v in state_dict.items()
                  if k in model_sd and v.shape == model_sd[k].shape}
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    print(f"[eval] 权重加载完毕  compatible={len(compatible)}  "
          f"missing={len(missing)}  unexpected={len(unexpected)}")

    model.to(device)
    model.eval()

    # ── 数据集 ────────────────────────────────────────────────────────
    dc      = cfg["data"]
    dataset = RoboCerebraDataset(
        root       = dc["root"],
        scenes     = dc.get("scenes"),
        img_h      = dc.get("img_h", 224),
        img_w      = dc.get("img_w", 224),
        subsample  = dc.get("subsample", 4),
        max_seqlen = dc.get("max_seqlen", 64),
    )
    n_eval = len(dataset) if args.max_traj is None else min(args.max_traj, len(dataset))
    print(f"[eval] 数据集: {len(dataset)} 条轨迹，本次评估 {n_eval} 条")

    all_metrics: List[Dict] = []
    for idx in range(n_eval):
        sample   = dataset[idx]
        traj_id  = f"traj_{idx:04d}"

        print(f"[eval] {idx+1}/{n_eval}  {traj_id}  T={sample['frames'].shape[0]}", end="  ", flush=True)

        metrics = eval_trajectory(
            model         = model,
            sample        = sample,
            traj_id       = traj_id,
            out_dir       = out_dir,
            device        = device,
            save_heatmaps = not args.no_heatmap,
            heatmap_step  = args.heatmap_step,
        )
        all_metrics.append(metrics)
        print(f"branches={metrics['n_branches']}  "
              + (f"F1={metrics.get('boundary_f1', 'N/A')}" if 'boundary_f1' in metrics else ""), flush=True)

    # ── 汇总指标 ─────────────────────────────────────────────────────
    summary_path = out_dir / "summary.json"
    summary: Dict = {
        "n_trajs": n_eval,
        "per_traj": all_metrics,
    }
    if any("boundary_f1" in m for m in all_metrics):
        f1_vals = [m["boundary_f1"] for m in all_metrics if "boundary_f1" in m]
        pr_vals = [m["boundary_precision"] for m in all_metrics if "boundary_precision" in m]
        rc_vals = [m["boundary_recall"] for m in all_metrics if "boundary_recall" in m]
        summary["mean_boundary_f1"]        = round(float(np.mean(f1_vals)), 4)
        summary["mean_boundary_precision"] = round(float(np.mean(pr_vals)), 4)
        summary["mean_boundary_recall"]    = round(float(np.mean(rc_vals)), 4)
        print(f"\n[eval] 平均边界检测  "
              f"F1={summary['mean_boundary_f1']:.4f}  "
              f"P={summary['mean_boundary_precision']:.4f}  "
              f"R={summary['mean_boundary_recall']:.4f}")

    with open(summary_path, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)
    print(f"[eval] 汇总指标已保存 → {summary_path}")
    print(f"[eval] 完成。记忆树 JSON → {out_dir}/trees/  "
          + (f"热力图 → {out_dir}/heatmaps/" if not args.no_heatmap else ""))


if __name__ == "__main__":
    main()
