"""
MemoryTreeVLA — Offline Evaluation Script
==========================================

Benchmarks
----------
  robocerebra  RoboCerebra long-horizon (NeurIPS 2025, arXiv:2506.06677)
  libero       LIBERO-SPATIAL / OBJECT / GOAL / LONG (NeurIPS 2023, arXiv:2306.03310)

Metrics
-------
  Action quality (all benchmarks)
    action_l1            Mean per-step L1 error: |a_pred[0] − a_gt|
    action_l2            Mean per-step L2 error: ||a_pred[0] − a_gt||₂

  Memory tree quality (all benchmarks)
    tree_nodes           Avg node count at trajectory end
    tree_depth           Avg max-depth at trajectory end
    tree_branches        Avg branch-creation events per trajectory
    tree_elevations      Avg elevation events per trajectory

  Subtask / progress  (RoboCerebra only — requires GT subtask labels)
    subtask_boundary_f1  Branch-creation F1 vs GT subtask boundaries (±tol steps)
    prog_monotone_rate   Fraction of (ancestor, descendant) pairs where
                         ancestor is farther from the task-goal embedding
    subtask_sr           Fraction of GT subtask boundaries that have ≥1
                         branch within ±boundary_tol timesteps

Usage
-----
  # RoboCerebra evaluation
  python eval.py \\
      --ckpt  checkpoints/runs/phase3_epoch0030.pt \\
      --config configs/default.yaml \\
      --dataset robocerebra \\
      --data_root dataset/RoboCerebra/RoboCerebra_trainset \\
      --out results/robocerebra_eval.json

  # LIBERO-LONG evaluation
  python eval.py \\
      --ckpt  checkpoints/runs/phase3_epoch0030.pt \\
      --config configs/default.yaml \\
      --dataset libero \\
      --data_root dataset/LIBERO \\
      --libero_split long \\
      --out results/libero_long_eval.json

  # GPU selection, batch size, subsample
  python eval.py --ckpt ... --dataset libero --device cuda:0 --subsample 2 --max_traj 50
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import yaml


# ================================================================
#  CLI
# ================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MemoryTreeVLA offline evaluation")

    # Checkpoint & config
    p.add_argument("--ckpt",    required=True,  help="Path to .pt model checkpoint")
    p.add_argument("--config",  default="configs/default.yaml",
                   help="YAML config used during training")

    # Dataset
    p.add_argument("--dataset",  choices=["robocerebra", "libero"],
                   required=True, help="Which benchmark to evaluate")
    p.add_argument("--data_root", required=True,
                   help="Root directory of the dataset")
    p.add_argument("--libero_split",
                   choices=["spatial", "object", "goal", "long"],
                   default="long",
                   help="LIBERO subset (only used when --dataset libero)")
    p.add_argument("--scenes",   nargs="*", default=None,
                   help="(RoboCerebra) scene sub-folders to include, e.g. "
                        "coffee_table kitchen_table")
    p.add_argument("--subsample", type=int, default=None,
                   help="Temporal subsample (overrides config)")
    p.add_argument("--max_traj",  type=int, default=None,
                   help="Limit number of trajectories evaluated (for quick testing)")
    p.add_argument("--max_seqlen", type=int, default=None,
                   help="Truncate trajectories to this many steps (overrides config)")

    # Runtime
    p.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=1,
                   help="Number of trajectories per evaluation batch "
                        "(set >1 only if GPU memory allows)")

    # Boundary detection tolerance (subtask-boundary F1 / SR)
    p.add_argument("--boundary_tol", type=int, default=5,
                   help="Timestep window for branch-vs-GT-boundary matching "
                        "(RoboCerebra only)")

    # Output
    p.add_argument("--out", default=None,
                   help="JSON file to write results (optional)")

    return p.parse_args()


# ================================================================
#  Config & model loading
# ================================================================

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_model(ckpt_path: str, cfg: dict, device: torch.device):
    """
    Load MemoryTreeVLA from checkpoint.
    Handles plain .pt (state_dict or full training ckpt) and
    DeepSpeed tag-file directory checkpoints.
    """
    from models import MemoryTreeVLA

    m_cfg = cfg.get("model", {})
    model = MemoryTreeVLA(
        llm_path   = m_cfg.get("llm_path", "checkpoints/Qwen2.5-1.5B-Instruct"),
        d          = m_cfg.get("d", 256),
        d_a        = m_cfg.get("d_a", 7),
        d_q        = m_cfg.get("d_q", 84),
        d_visual   = m_cfg.get("d_visual", 256),
        d_ssm      = m_cfg.get("d_ssm", 256),
        d_state    = m_cfg.get("d_state", 16),
        patch_size = m_cfg.get("patch_size", 16),
        H_a        = m_cfg.get("H_a", 16),
        n_ode      = m_cfg.get("n_ode", 20),
        theta_fuse = m_cfg.get("theta_fuse", 0.4),
        K_elev     = m_cfg.get("K_elev", 4),
        delta_w    = m_cfg.get("delta_w", 0.1),
        tau        = m_cfg.get("tau", 0.1),
        freeze_llm = False,   # load full weights regardless
    )

    ckpt_path = str(ckpt_path)

    # DeepSpeed checkpoint directory
    if os.path.isdir(ckpt_path):
        # Gather sharded weight files and merge
        import glob
        shard_files = sorted(glob.glob(
            os.path.join(ckpt_path, "**", "*model_states.pt"), recursive=True
        ))
        if not shard_files:
            raise FileNotFoundError(
                f"No model_states.pt found in DeepSpeed checkpoint dir: {ckpt_path}"
            )
        # Load first shard (ZeRO-2: all model params on each rank)
        state = torch.load(shard_files[0], map_location="cpu")
        sd = state.get("module", state)
    else:
        state = torch.load(ckpt_path, map_location="cpu")
        # Training checkpoint wraps state_dict under 'model_state_dict'
        sd = (
            state.get("model_state_dict")
            or state.get("module")
            or state
        )

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[WARN] Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing)>5 else ''}")
    if unexpected:
        print(f"[WARN] Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")

    model.to(device)
    model.eval()
    return model


# ================================================================
#  Metrics helpers
# ================================================================

def tree_max_depth(tree) -> int:
    """Maximum node depth across all nodes."""
    if tree.root_id is None:
        return 0
    return max(tree.depth(nid) for nid in tree.nodes)


def tree_elevation_events(tree) -> int:
    """Count nodes that are 'abstract' (inserted by elevation).
    These are identified as non-leaf nodes whose children were re-parented
    — approximated here by counting nodes with node_id gaps, or by the
    elevation_counter attribute if tracked externally.
    Falls back to the tree's own counter if available."""
    return getattr(tree, "_elevation_count", 0)


def compute_prog_monotone_rate(tree) -> float:
    """
    Fraction of (ancestor, descendant) pairs where ancestor is farther from
    the mean leaf-node semantic embedding (= task goal proxy).
    Section 3.6: L_prog corresponds to dist(ancestor, goal) > dist(descendant, goal).
    """
    if tree.root_id is None or len(tree.nodes) < 2:
        return 1.0

    # Collect leaf semantics as task-goal proxy
    leaf_s = [
        tree.nodes[nid].s.float()
        for nid in tree.nodes
        if tree.nodes[nid].is_leaf()
    ]
    if not leaf_s:
        return 1.0
    goal = F.normalize(torch.stack(leaf_s).mean(0), dim=-1)   # (d,)

    pairs = tree.ancestor_descendant_pairs()
    if not pairs:
        return 1.0

    correct = 0
    for anc_id, desc_id in pairs:
        if anc_id not in tree.nodes or desc_id not in tree.nodes:
            continue
        s_a = F.normalize(tree.nodes[anc_id].s.float(),  dim=-1)
        s_d = F.normalize(tree.nodes[desc_id].s.float(), dim=-1)
        dist_a = 1.0 - (s_a * goal).sum().item()
        dist_d = 1.0 - (s_d * goal).sum().item()
        # Correct if ancestor is farther (or equal within epsilon)
        if dist_a >= dist_d - 1e-4:
            correct += 1

    return correct / len(pairs)


def compute_boundary_matches(
    branch_steps: List[int],
    gt_boundaries: List[int],
    tol: int,
) -> Tuple[int, int, int]:
    """
    Compute TP, FP, FN for boundary detection with ±tol tolerance.

    Returns (tp, fp, fn).
    """
    matched_gt  = set()
    matched_pred = set()

    for bi, gt in enumerate(gt_boundaries):
        for pi, pr in enumerate(branch_steps):
            if abs(pr - gt) <= tol and bi not in matched_gt and pi not in matched_pred:
                matched_gt.add(bi)
                matched_pred.add(pi)
                break

    tp = len(matched_gt)
    fp = len(branch_steps) - len(matched_pred)
    fn = len(gt_boundaries) - tp
    return tp, fp, fn


def f1_from_counts(tp: int, fp: int, fn: int) -> float:
    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    denom     = precision + recall
    return 2 * precision * recall / denom if denom > 0 else 0.0


# ================================================================
#  Single-trajectory evaluation
# ================================================================

@torch.no_grad()
def evaluate_trajectory(
    model,
    frames: torch.Tensor,     # (T, 3, H, W)
    actions: torch.Tensor,    # (T, d_a)  — ground-truth actions
    states: torch.Tensor,     # (T, d_q)
    instruction: str,
    device: torch.device,
    has_subtask_labels: bool = False,
    subtask_ids: Optional[torch.Tensor] = None,   # (T,) int
) -> Dict:
    """
    Run one trajectory through model.step(), collecting per-step metrics.
    Returns a dict of scalar metric values for this trajectory.
    """
    T, d_a = actions.shape
    if T == 0:
        return {}

    model.reset_trees(batch_size=1)
    tree = model.get_tree(0)
    # Attach elevation counter
    tree._elevation_count = 0

    l1_errors, l2_errors = [], []
    branch_steps: List[int] = []   # timesteps where a branch was created
    prev_size = 0
    a_prev = None

    for t in range(T):
        img_t = frames[t].unsqueeze(0).to(device)     # (1, 3, H, W)
        q_t   = states[t].unsqueeze(0).to(device)     # (1, d_q)

        # Snapshot tree size before step
        size_before = tree.size()

        # Predict action chunk
        a_chunk = model.step(img_t, instruction, q_t, a_prev)   # (1, H_a, d_a)

        # Track new branches (size increased by 1 → branch, not elevation)
        size_after = tree.size()
        if size_after > size_before:
            branch_steps.append(t)

        # Track elevations via elevation_pending_parent
        # (elevation may add an extra node, total size can jump by 2)
        if tree.elevation_pending_parent is not None:
            tree._elevation_count += 1
            # The model's step() doesn't clear elevation_pending_parent
            # (bug in original code — we log it but don't re-trigger it)
            tree.elevation_pending_parent = None

        # Action L1 / L2  (compare first predicted step vs GT action)
        a_pred_first = a_chunk[0, 0].cpu()    # (d_a,)
        a_gt_t       = actions[t]              # (d_a,)
        l1_errors.append((a_pred_first - a_gt_t).abs().mean().item())
        l2_errors.append((a_pred_first - a_gt_t).norm().item())

        # Teacher-force: feed GT action as a_prev
        a_prev = actions[t].unsqueeze(0).to(device)

    # --- Tree stats -------------------------------------------------------
    result: Dict = {
        "action_l1":      float(sum(l1_errors) / max(len(l1_errors), 1)),
        "action_l2":      float(sum(l2_errors) / max(len(l2_errors), 1)),
        "tree_nodes":     float(tree.size()),
        "tree_depth":     float(tree_max_depth(tree)),
        "tree_branches":  float(len(branch_steps)),
        "tree_elevations":float(tree._elevation_count),
    }

    # --- Subtask-specific metrics (RoboCerebra) ---------------------------
    if has_subtask_labels and subtask_ids is not None:
        # GT subtask boundaries: timesteps where subtask label changes
        gt_bounds: List[int] = []
        for t in range(1, T):
            if subtask_ids[t].item() != subtask_ids[t - 1].item():
                gt_bounds.append(t)

        # Boundary F1
        tp, fp, fn = compute_boundary_matches(branch_steps, gt_bounds, tol=5)
        result["subtask_boundary_f1"] = f1_from_counts(tp, fp, fn)

        # Subtask success rate: fraction of GT boundaries with ≥1 matched branch
        sr = tp / max(len(gt_bounds), 1)
        result["subtask_sr"] = float(sr)

        # Progress monotonicity
        result["prog_monotone_rate"] = compute_prog_monotone_rate(tree)

    return result


# ================================================================
#  Main evaluation loop
# ================================================================

def run_evaluation(args: argparse.Namespace):
    cfg    = load_config(args.config)
    device = torch.device(args.device)

    # Override config with CLI args
    data_cfg = cfg.get("data", {})
    if args.subsample  is not None: data_cfg["subsample"]  = args.subsample
    if args.max_seqlen is not None: data_cfg["max_seqlen"] = args.max_seqlen

    subsample  = data_cfg.get("subsample", 4)
    max_seqlen = data_cfg.get("max_seqlen", 256)

    # ── Load model ───────────────────────────────────────────────────
    print(f"Loading model from {args.ckpt} ...")
    model = load_model(args.ckpt, cfg, device)

    # ── Build dataset ────────────────────────────────────────────────
    print(f"Building dataset ({args.dataset}) ...")
    has_subtask_labels = (args.dataset == "robocerebra")

    if args.dataset == "robocerebra":
        from dataset import RoboCerebraDataset
        ds = RoboCerebraDataset(
            root       = args.data_root,
            scenes     = args.scenes,
            img_h      = data_cfg.get("img_h", 224),
            img_w      = data_cfg.get("img_w", 224),
            subsample  = subsample,
            max_seqlen = max_seqlen,
        )
    else:   # libero
        from dataset.libero import LIBERODataset
        ds = LIBERODataset(
            root       = args.data_root,
            split      = args.libero_split,
            img_h      = data_cfg.get("img_h", 224),
            img_w      = data_cfg.get("img_w", 224),
            subsample  = subsample,
            max_seqlen = max_seqlen,
            d_q        = cfg.get("model", {}).get("d_q", 84),
        )

    n_traj = len(ds)
    if args.max_traj is not None:
        n_traj = min(n_traj, args.max_traj)
    print(f"Evaluating {n_traj} trajectories ...")

    # ── Per-trajectory evaluation ─────────────────────────────────────
    all_results: List[Dict] = []
    t0 = time.time()

    for idx in range(n_traj):
        sample = ds[idx]

        frames      = sample["frames"]          # (T, 3, H, W)
        actions     = sample["actions"]         # (T, 7)
        states      = sample["states"]          # (T, d_q)
        instruction = sample["instruction"]
        subtask_ids = sample.get("subtask_ids") # (T,) or None

        traj_result = evaluate_trajectory(
            model         = model,
            frames        = frames,
            actions       = actions,
            states        = states,
            instruction   = instruction,
            device        = device,
            has_subtask_labels = has_subtask_labels,
            subtask_ids   = subtask_ids,
        )
        traj_result["trajectory_idx"] = idx
        traj_result["instruction"]    = instruction
        all_results.append(traj_result)

        # Progress print
        if (idx + 1) % 10 == 0 or idx == n_traj - 1:
            elapsed = time.time() - t0
            fps = (idx + 1) / elapsed
            print(
                f"  [{idx+1:>4}/{n_traj}]  "
                f"L1={traj_result.get('action_l1', 0):.4f}  "
                f"L2={traj_result.get('action_l2', 0):.4f}  "
                f"nodes={traj_result.get('tree_nodes', 0):.1f}  "
                f"depth={traj_result.get('tree_depth', 0):.1f}  "
                f"({fps:.2f} traj/s)"
            )

    # ── Aggregate ────────────────────────────────────────────────────
    summary = aggregate_results(all_results, has_subtask_labels)

    print("\n" + "=" * 60)
    print(f"{'Metric':<35} {'Value':>10}")
    print("-" * 60)
    for k, v in summary.items():
        print(f"  {k:<33} {v:>10.4f}")
    print("=" * 60)

    # ── Save ─────────────────────────────────────────────────────────
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config":          args.config,
            "checkpoint":      args.ckpt,
            "dataset":         args.dataset,
            "libero_split":    args.libero_split if args.dataset == "libero" else None,
            "n_trajectories":  n_traj,
            "summary":         summary,
            "per_trajectory":  all_results,
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nResults saved to {out_path}")

    return summary


# ================================================================
#  Result aggregation
# ================================================================

def aggregate_results(
    results: List[Dict],
    has_subtask_labels: bool,
) -> Dict[str, float]:
    """Compute mean ± std for all numeric metrics across trajectories."""
    if not results:
        return {}

    # Collect numeric keys (exclude string & index fields)
    ignore = {"trajectory_idx", "instruction"}
    numeric_keys = [
        k for k in results[0]
        if k not in ignore and isinstance(results[0][k], (int, float))
    ]

    summary: Dict[str, float] = {}
    for k in numeric_keys:
        vals = [r[k] for r in results if k in r and not math.isnan(r[k])]
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        std  = math.sqrt(sum((v - mean) ** 2 for v in vals) / max(len(vals) - 1, 1))
        summary[k]           = mean
        summary[f"{k}_std"]  = std

    return summary


# ================================================================
#  Entry point
# ================================================================

if __name__ == "__main__":
    args = parse_args()
    run_evaluation(args)
