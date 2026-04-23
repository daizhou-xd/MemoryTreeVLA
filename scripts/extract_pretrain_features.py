"""
离线预提取 RoboCerebra 数据集的冻结 ViT 特征。

运行一次即可，训练时直接加载缓存，跳过视频读取 + ViT + LLM forward：
    python scripts/extract_pretrain_features.py \
        --config dual_tree_vla/config/pretrain.yaml \
        [--cache_dir data/RoboCerebra/_feat_cache] \
        [--frame_idx 0]          # 取第 0 帧（也可以 "mid" 取中间帧）
        [--device cuda:0]

每条轨迹保存一个 .pt 文件，内容：
    {
        "P_t_raw":  (1025, 1024) float16  — ViT 最后一层含 CLS
        "z_v_feat": (1,    896)  float16  — mlp1 投影后 LLM 空间特征
    }

存储量估算：~2.1 MB/轨迹 × 253 轨迹 ≈ 530 MB，可接受。
"""
from __future__ import annotations

import argparse
import os
import sys
from math import isqrt
from pathlib import Path

# ── 保证能 import dual_tree_vla ──────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

import h5py
import numpy as np
import torch
import yaml
from tqdm import tqdm

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


# ================================================================
#  Helper: 读取单帧
# ================================================================

def _load_one_frame(video_path: str, frame_idx: int | str, h: int, w: int) -> np.ndarray:
    """Returns (H, W, 3) uint8 numpy array."""
    if not _CV2_AVAILABLE or not os.path.isfile(video_path):
        return np.zeros((h, w, 3), dtype=np.uint8)

    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if isinstance(frame_idx, str) and frame_idx == "mid":
        frame_idx = n_frames // 2

    cap.set(cv2.CAP_PROP_POS_FRAMES, min(frame_idx, n_frames - 1))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return np.zeros((h, w, 3), dtype=np.uint8)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (w, h))
    return frame


# ================================================================
#  Main
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="Extract pretrain ViT features")
    parser.add_argument("--config", default="dual_tree_vla/config/pretrain.yaml")
    parser.add_argument("--cache_dir", default=None,
                        help="缓存保存根目录，默认与数据集同级 _feat_cache/")
    parser.add_argument("--frame_idx", default="mid",
                        help="取第几帧：整数 or 'mid'（默认中间帧）")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true",
                        help="重新计算已存在的缓存")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_cfg  = cfg["data"]
    model_cfg = cfg["model"]
    device    = torch.device(args.device)

    root_path = Path(data_cfg["root"])
    cache_dir = Path(args.cache_dir) if args.cache_dir else (root_path.parent / "_feat_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    img_h = data_cfg.get("img_h", 224)
    img_w = data_cfg.get("img_w", 224)

    frame_idx: int | str
    try:
        frame_idx = int(args.frame_idx)
    except ValueError:
        frame_idx = "mid"

    # ── 加载骨架 (只需要 VLM backbone) ────────────────────────────
    print(f"[extract] 加载 InternVL3 骨架: {model_cfg['vlm_path']} ...")
    from dual_tree_vla.model.backbone import InternVL3Backbone

    backbone = InternVL3Backbone(config={
        "vlm_name":           model_cfg["vlm_path"],
        "device":             str(device),
        "action_horizon":     model_cfg.get("H_a", 16),
        "per_action_dim":     model_cfg.get("d_a", 7),
        "embed_dim":          model_cfg.get("d_vit", 896),
        "state_dim":          model_cfg.get("d_q", 7),
        "num_inference_timesteps": model_cfg.get("n_ode", 50),
    })
    embedder = backbone.embedder
    embedder.model.eval()

    # ── 注册 ViT hook 捕获 P_t_raw ────────────────────────────────
    _P_t_raw_holder: list = []

    def _hook(module, input, output):
        hs = output[0] if isinstance(output, tuple) else output
        _P_t_raw_holder.clear()
        _P_t_raw_holder.append(hs.detach().cpu().to(torch.float16))

    vit_last = embedder.model.vision_model.encoder.layers[-1]
    _hook_handle = vit_last.register_forward_hook(_hook)

    # ── 遍历所有轨迹 ────────────────────────────────────────────
    all_scenes = sorted(d.name for d in root_path.iterdir() if d.is_dir())
    scenes = data_cfg.get("scenes")
    if scenes:
        all_scenes = [s for s in all_scenes if s in scenes]

    trajectories = []
    for scene in all_scenes:
        scene_path = root_path / scene
        for case_dir in sorted(scene_path.iterdir()):
            if not case_dir.is_dir():
                continue
            hdf5_files = list(case_dir.glob("demo.hdf5"))
            if not hdf5_files:
                continue
            hdf5_path = str(hdf5_files[0])
            case_name = case_dir.name
            video_path = str(case_dir / f"{case_name}.mp4")
            task_json  = str(case_dir / "task_description.json")
            if not os.path.isfile(task_json):
                continue
            try:
                with h5py.File(hdf5_path, "r") as f:
                    demo_keys = list(f.get("data", {}).keys())
            except Exception:
                continue
            for dk in demo_keys:
                trajectories.append((scene, case_dir.name, hdf5_path, video_path, dk))

    print(f"[extract] 共 {len(trajectories)} 条轨迹，缓存目录: {cache_dir}")

    n_new = 0
    with torch.no_grad():
        for scene, case_name, hdf5_path, video_path, demo_key in tqdm(
            trajectories, desc="提取特征", unit="traj"
        ):
            out_dir  = cache_dir / scene / case_name
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{demo_key}.pt"

            if out_path.exists() and not args.overwrite:
                continue

            # 读取 1 帧
            frame_np = _load_one_frame(video_path, frame_idx, img_h, img_w)
            frame_t  = torch.from_numpy(frame_np.astype(np.float32) / 255.0)
            frame_t  = frame_t.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)

            # 走 embedder 预处理 + ViT forward
            # _preprocess_images 期望 PIL.Image 或 (C,H,W) tensor
            img_list = [frame_t.squeeze(0)]  # (3, H, W)
            try:
                pixel_values, _ = embedder._preprocess_images(img_list)
            except Exception:
                # 回退：直接当 pixel_values
                pixel_values = frame_t.to(device)

            # ViT + mlp1 → vit_embeds_orig (z_v_feat 来源)
            pixel_values = pixel_values.to(device)
            vit_embeds_orig = embedder.model.extract_feature(pixel_values)
            # z_v_feat: mlp1 输出均值 → (tiles, d_vit=896)
            z_v_feat = vit_embeds_orig.mean(dim=1).to(torch.float16).cpu()  # (tiles, 896)

            # P_t_raw: 由 hook 填充 (tiles, 1025, 1024)
            if _P_t_raw_holder:
                P_t_raw = _P_t_raw_holder[0]  # (tiles, 1025, 1024) fp16
            else:
                # Fallback: 重建 P_t_raw shape（不应该到这里）
                P_t_raw = torch.zeros(1, 1025, 1024, dtype=torch.float16)

            torch.save({"P_t_raw": P_t_raw, "z_v_feat": z_v_feat}, out_path)
            n_new += 1

    _hook_handle.remove()

    n_skip = len(trajectories) - n_new
    print(f"[extract] 完成。新建: {n_new}，跳过: {n_skip}，缓存目录: {cache_dir}")
    print("[extract] 训练时在 pretrain.yaml 中添加:")
    print(f"  data:")
    print(f"    feat_cache_dir: \"{cache_dir}\"")


if __name__ == "__main__":
    main()
