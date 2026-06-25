#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run five-seed evaluation for SGC, GST, PSO, and BI-PPO.

This script keeps the trained BI-PPO model fixed and changes only the test
environment seed. It saves one test log per method/seed and writes a summary
CSV with mean/std metrics for Table 2.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


CODE_DIR = Path(r"E:\毕业论文\2deployment\code")
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

METHODS = ("SGC", "GST", "PSO", "BIS-PPO")
DEFAULT_SEEDS = (42, 123, 2024, 2025, 3407)
DEFAULT_OUT_DIR = r"E:\毕业论文\2deployment\code\multi_seed"

SHAPEFILE_PATH = r"E:\毕业论文\1planning\数据\ChibiUAVbase\chibi_UAVbase_boustrophedon_paths_energy_with_batch_and_uav_point.shp"
AIR_AIR_CSV = r"E:\毕业论文\2deployment\data\air_air_data.csv"
AIR_GROUND_CSV = r"E:\毕业论文\2deployment\data\air_ground_data.csv"

INIT_RELAY_POS = [
    [29.788655458031535, 113.9091413860207, 80.0],
    [29.794116735397676, 113.91302969947397, 100.0],
]
GROUND_POS = [29.789405942654724, 113.91183988045603, 10.0]

N_RELAYS = 2
TOTAL_BW_HZ = 20e6
DEMAND_BPS = 12e6
MAX_SPEED_M_PER_STEP = 20.0
MAX_V_SPEED_M_PER_STEP = 4.0
MIN_ALT_M = 50.0
MAX_ALT_M = 300.0


def make_raw_env(seed: int, episode_len: int, fixed_start: bool = True) -> RelayMoveEnvShp:
    """Create the unvectorized relay environment used by baselines."""
    from Env_matplot_relaymove_shp_v4 import RelayMoveEnvShp

    init_pos = np.array(INIT_RELAY_POS, dtype=np.float64) if fixed_start else None
    env = RelayMoveEnvShp(
        shapefile_path=SHAPEFILE_PATH,
        n_relays=N_RELAYS,
        n_survey=None,
        batch_list=None,
        total_bandwidth_hz=TOTAL_BW_HZ,
        demand_bps=DEMAND_BPS,
        max_speed_m_per_step=MAX_SPEED_M_PER_STEP,
        max_v_speed_m_per_step=MAX_V_SPEED_M_PER_STEP,
        min_alt_m=MIN_ALT_M,
        max_alt_m=MAX_ALT_M,
        episode_len=episode_len,
        seed=seed,
        init_relay_pos=init_pos,
        ground_pos=np.array(GROUND_POS, dtype=np.float64),
        use_meas_fit=True,
        air_air_csv=AIR_AIR_CSV,
        air_ground_csv=AIR_GROUND_CSV,
    )
    env.test_mode = True
    return env


def make_vec_env(seed: int, episode_len: int):
    """Create the vectorized environment used by the trained BI-PPO model."""
    from stable_baselines3.common.vec_env import DummyVecEnv

    def _init():
        return make_raw_env(seed=seed, episode_len=episode_len, fixed_start=True)

    return DummyVecEnv([_init])


def controller_for(method: str):
    from chibi_baselines_relay_shp_V3 import GSTController, PSOController, SGCController

    method = method.upper()
    if method == "SGC":
        return SGCController()
    if method == "GST":
        return GSTController()
    if method == "PSO":
        return PSOController()
    raise ValueError(f"Unknown baseline method: {method}")


def run_baseline(method: str, seed: int, episode_len: int) -> pd.DataFrame:
    env = make_raw_env(seed=seed, episode_len=episode_len, fixed_start=True)
    controller = controller_for(method)
    obs, _ = env.reset()
    controller.reset(env)

    try:
        for _ in range(episode_len):
            action = controller.act(env, obs)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        return pd.DataFrame(env.log_data)
    finally:
        env.close()


def run_bippo(seed: int, episode_len: int) -> pd.DataFrame:
    from chibi_PPO_matplot_shp_v4_B_I import BetaICMPPO, MODEL_PATH, VECNORM_PATH
    from stable_baselines3.common.vec_env import VecNormalize

    if not (os.path.exists(MODEL_PATH) and os.path.exists(VECNORM_PATH)):
        raise FileNotFoundError(
            f"Missing BI-PPO model or VecNormalize stats: {MODEL_PATH}, {VECNORM_PATH}"
        )

    # DummyVecEnv automatically resets the wrapped env when it reaches done.
    # Use a slightly longer env horizon than the measured rollout so reset()
    # does not clear env.log_data on the final requested step.
    eval_env = make_vec_env(seed=seed, episode_len=episode_len + 100)
    eval_env = VecNormalize.load(VECNORM_PATH, eval_env)
    eval_env.training = False
    eval_env.norm_reward = False

    custom_objects = {
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
    }

    model = BetaICMPPO.load(MODEL_PATH, env=eval_env, custom_objects=custom_objects)
    eval_env.set_attr("test_mode", True)

    obs = eval_env.reset()
    try:
        for _ in range(episode_len):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = eval_env.step(action)
            if bool(done[0]):
                break

        log_data = eval_env.get_attr("log_data")[0]
        return pd.DataFrame(log_data)
    finally:
        eval_env.close()


def trajectory_smoothness(df: pd.DataFrame) -> float:
    """Match the paper's smoothness cost: sum of action-vector changes."""
    relay_cols = sorted(
        {
            int(c[len("relay") : -len("_lat")])
            for c in df.columns
            if c.startswith("relay") and c.endswith("_lat")
        }
    )
    if len(df) < 3 or not relay_cols:
        return 0.0

    actions = []
    for _, row in df.iterrows():
        row_actions = []
        for r in relay_cols:
            row_actions.extend(
                [
                    float(row[f"relay{r}_lat"]),
                    float(row[f"relay{r}_lon"]),
                    float(row[f"relay{r}_alt_m"]),
                ]
            )
        actions.append(row_actions)

    pos = np.asarray(actions, dtype=np.float64)
    vel = np.diff(pos, axis=0)
    acc = np.diff(vel, axis=0)
    return float(np.sum(np.linalg.norm(acc, axis=1)))


def link_weak_coverage_risk(
    df: pd.DataFrame,
    rssi_threshold_dbm: float = -110.0,
    rate_threshold_bps: float = 1e6,
) -> float:
    """Strict link-level risk over all survey-link time slots."""
    rate_cols = sorted(c for c in df.columns if c.startswith("rate_") and c.endswith("_bps"))
    if not rate_cols:
        return float("nan")
    total = 0
    weak = 0
    for rate_col in rate_cols:
        idx = rate_col[len("rate_") : -len("_bps")]
        rssi_col = f"rssi_{idx}_dbm"
        rates = df[rate_col].to_numpy(dtype=float)
        if rssi_col in df.columns:
            rssis = df[rssi_col].to_numpy(dtype=float)
            weak_mask = (rates < rate_threshold_bps) | (rssis < rssi_threshold_dbm)
        else:
            weak_mask = rates < rate_threshold_bps
        weak += int(np.sum(weak_mask))
        total += len(rates)
    return 100.0 * weak / max(total, 1)


def compute_metrics(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty or "sum_rate_bps" not in df.columns:
        raise ValueError(
            "Expected a non-empty test log with column 'sum_rate_bps'. "
            f"Got columns: {list(df.columns)}"
        )
    return {
        "avg_sum_rate_mbps": float(df["sum_rate_bps"].mean() / 1e6),
        "weak_coverage_risk_pct": link_weak_coverage_risk(df),
        "trajectory_smoothness": trajectory_smoothness(df),
    }


def summarize(records: List[Dict[str, float]]) -> pd.DataFrame:
    rows = []
    grouped: Dict[str, List[Dict[str, float]]] = {}
    for rec in records:
        grouped.setdefault(str(rec["method"]), []).append(rec)

    for method in METHODS:
        vals = grouped.get(method, [])
        if not vals:
            continue
        row = {"method": method, "n_seeds": len(vals)}
        for metric in ("avg_sum_rate_mbps", "weak_coverage_risk_pct", "trajectory_smoothness"):
            arr = np.asarray([float(v[metric]) for v in vals], dtype=float)
            row[f"{metric}_mean"] = float(np.mean(arr))
            row[f"{metric}_std"] = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def parse_seeds(seed_text: str) -> Tuple[int, ...]:
    return tuple(int(x.strip()) for x in seed_text.split(",") if x.strip())


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    parser.add_argument("--episode-len", type=int, default=1210)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--methods",
        default=",".join(METHODS),
        help="Comma-separated subset of SGC,GST,PSO,BIS-PPO",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    seeds = parse_seeds(args.seeds)
    methods = tuple(m.strip() for m in args.methods.split(",") if m.strip())
    out_dir = Path(args.out_dir)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, float]] = []

    for seed in seeds:
        for method in methods:
            print(f"Running {method} with seed={seed} ...")
            if method.upper() == "BIS-PPO":
                df = run_bippo(seed=seed, episode_len=args.episode_len)
                method_label = "BIS-PPO"
            else:
                method_label = method.upper()
                df = run_baseline(method=method_label, seed=seed, episode_len=args.episode_len)

            log_path = logs_dir / f"{method_label}_test_log_seed{seed}.csv"
            df.to_csv(log_path, index=False)

            try:
                metrics = compute_metrics(df)
            except ValueError as exc:
                raise ValueError(
                    f"Failed to compute metrics for method={method_label}, seed={seed}, "
                    f"log_path={log_path}"
                ) from exc
            metrics.update({"method": method_label, "seed": seed, "steps": len(df)})
            records.append(metrics)

    per_seed = pd.DataFrame(records)
    per_seed.to_csv(out_dir / "per_seed_metrics.csv", index=False)

    summary = summarize(records)
    summary.to_csv(out_dir / "summary_mean_std.csv", index=False)
    print("\nPer-seed metrics:")
    print(per_seed.to_string(index=False))
    print("\nSummary:")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
