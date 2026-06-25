#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Measure per-step upper-level decision time for SGC, GST, PSO, and BI-PPO.

The timer only covers the deployment decision:
  - SGC/GST/PSO: controller.act(env, obs)
  - BI-PPO: model.predict(obs, deterministic=True)

It does not include env.step(), bandwidth allocation inside env.step(), logging,
CSV writing, or plotting. Results are written to:
E:\\毕业论文\\2deployment\\code\\multi_seed\\inference_time.csv
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


CODE_DIR = Path(r"E:\毕业论文\2deployment\code")
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

DEFAULT_OUT_CSV = r"E:\毕业论文\2deployment\code\multi_seed\inference_time.csv"
DEFAULT_METHODS = ("SGC", "GST", "PSO", "BIS-PPO")

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


def make_raw_env(seed: int, episode_len: int):
    from Env_matplot_relaymove_shp_v4 import RelayMoveEnvShp

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
        init_relay_pos=np.array(INIT_RELAY_POS, dtype=np.float64),
        ground_pos=np.array(GROUND_POS, dtype=np.float64),
        use_meas_fit=True,
        air_air_csv=AIR_AIR_CSV,
        air_ground_csv=AIR_GROUND_CSV,
    )
    env.test_mode = False
    return env


def make_vec_env(seed: int, episode_len: int):
    from stable_baselines3.common.vec_env import DummyVecEnv

    def _init():
        return make_raw_env(seed=seed, episode_len=episode_len)

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


def cuda_synchronize_if_available() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        return


def summarize_times(method: str, times_ms: List[float]) -> Dict[str, float | str | int]:
    arr = np.asarray(times_ms, dtype=np.float64)
    return {
        "method": method,
        "n_steps": int(arr.size),
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "median_ms": float(np.median(arr)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
    }


def measure_baseline(method: str, seed: int, episode_len: int, warmup_steps: int) -> Dict[str, float | str | int]:
    env = make_raw_env(seed=seed, episode_len=episode_len)
    controller = controller_for(method)
    obs, _ = env.reset()
    controller.reset(env)

    times_ms: List[float] = []
    try:
        for step in range(episode_len):
            t0 = time.perf_counter()
            action = controller.act(env, obs)
            t1 = time.perf_counter()

            if step >= warmup_steps:
                times_ms.append((t1 - t0) * 1000.0)

            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
    finally:
        env.close()

    return summarize_times(method.upper(), times_ms)


def measure_bippo(seed: int, episode_len: int, warmup_steps: int) -> Dict[str, float | str | int]:
    from chibi_PPO_matplot_shp_v4_B_I import BetaICMPPO, MODEL_PATH, VECNORM_PATH
    from stable_baselines3.common.vec_env import VecNormalize

    if not (os.path.exists(MODEL_PATH) and os.path.exists(VECNORM_PATH)):
        raise FileNotFoundError(f"Missing BI-PPO model or VecNormalize stats: {MODEL_PATH}, {VECNORM_PATH}")

    eval_env = make_vec_env(seed=seed, episode_len=episode_len)
    eval_env = VecNormalize.load(VECNORM_PATH, eval_env)
    eval_env.training = False
    eval_env.norm_reward = False

    custom_objects = {
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
    }
    model = BetaICMPPO.load(MODEL_PATH, env=eval_env, custom_objects=custom_objects)

    obs = eval_env.reset()
    times_ms: List[float] = []
    try:
        for step in range(episode_len):
            cuda_synchronize_if_available()
            t0 = time.perf_counter()
            action, _ = model.predict(obs, deterministic=True)
            cuda_synchronize_if_available()
            t1 = time.perf_counter()

            if step >= warmup_steps:
                times_ms.append((t1 - t0) * 1000.0)

            obs, _, done, _ = eval_env.step(action)
            if bool(done[0]):
                break
    finally:
        eval_env.close()

    return summarize_times("BIS-PPO", times_ms)


def parse_methods(text: str) -> List[str]:
    methods = [m.strip() for m in text.split(",") if m.strip()]
    valid = set(DEFAULT_METHODS)
    unknown = [m for m in methods if m not in valid]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Valid methods: {sorted(valid)}")
    return methods


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-csv", default=DEFAULT_OUT_CSV)
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--episode-len", type=int, default=1310)
    parser.add_argument("--warmup-steps", type=int, default=20)
    args = parser.parse_args(list(argv) if argv is not None else None)

    methods = parse_methods(args.methods)
    rows: List[Dict[str, float | str | int]] = []

    for method in methods:
        print(f"Measuring {method} ...")
        if method == "BIS-PPO":
            row = measure_bippo(seed=args.seed, episode_len=args.episode_len, warmup_steps=args.warmup_steps)
        else:
            row = measure_baseline(
                method=method,
                seed=args.seed,
                episode_len=args.episode_len,
                warmup_steps=args.warmup_steps,
            )
        rows.append(row)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"Saved inference timing table to: {out_csv}")
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
