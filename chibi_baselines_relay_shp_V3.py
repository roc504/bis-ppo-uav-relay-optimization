# -*- coding: utf-8 -*-
"""
chibi_baselines_relay_shp_V3.py

Baselines for RelayMoveEnvShp (v4) - ALIGNED VERSION & FIXED PLOTTING
- SGC: Static Geometric Center
- GST: Greedy Signal Tracking (Corrected to use SA Bandwidth Allocation)
- PSO: Particle Swarm Optimization (Corrected to use SA Bandwidth Allocation)

Updates:
1. Aligned Physical Constraints & SA Resource Allocation with PPO Env.
2. Added individual Link Rate curves to Throughput plot.
3. Added RSSI plot.
4. Added a horizontal mean-fit line for total throughput.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import re
from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd

# ---------------------------------------------------------
# 引用你的新环境
# ---------------------------------------------------------
from Env_matplot_relaymove_shp_v4 import RelayMoveEnvShp, haversine_m
from DR_SA_g import simulated_annealing, SAConfig

# -----------------------------
# User config (配置区)
# -----------------------------
METHOD = "PSO"  # 可选: "SGC" | "GST" | "PSO"

# 路径需保持一致
# SHAPEFILE_PATH = r"E:\毕业论文\1planning\数据\boustrophedon_paths_with_batch_uav.shp"
SHAPEFILE_PATH = r"E:\毕业论文\1planning\数据\ChibiUAVbase\chibi_UAVbase_boustrophedon_paths_energy_with_batch_and_uav_point.shp"
AIR_AIR_CSV = r"E:\毕业论文\2deployment\data\air_air_data.csv"
AIR_GROUND_CSV = r"E:\毕业论文\2deployment\data\air_ground_data.csv"

# INIT_RELAY_POS = [
#     [30.52131148155365, 114.370137061499, 80.0],  # 研究区右下角
#     [30.527200552833573, 114.3556190247159, 100.0]
# ]
# GROUND_POS = [30.516004194419658, 114.38530939375414, 0.0]

INIT_RELAY_POS = [
    # [30.52131148155365, 114.370137061499, 80.0],  # 研究区右下角
    # [30.527200552833573, 114.3556190247159, 100.0]#武大
    [29.788655458031535 , 113.9091413860207, 80.0],
    [29.794116735397676 , 113.91302969947397, 100.0]#赤壁
]

GROUND_POS = [29.789405942654724,  113.91183988045603, 10.0]#赤壁基地

N_RELAYS = 2
TOTAL_BW_HZ = 20e6
DEMAND_BPS = 12e6

# === [关键修改] 约束条件必须与 PPO 环境完全一致 ===
MAX_SPEED_M_PER_STEP = 20.0
MAX_V_SPEED_M_PER_STEP = 4.0
MIN_ALT_M = 50.0
MAX_ALT_M = 300.0

EPISODE_LEN = 1210
SEED = 2024
OUT_DIR = "runs_baselines_aligned_chibi"


# -----------------------------
# Utilities
# -----------------------------
def _infer_indices(cols, prefix: str, suffix: str):
    pat = re.compile(rf"^{re.escape(prefix)}(\d+){re.escape(suffix)}$")
    idx = set()
    for c in cols:
        m = pat.match(c)
        if m:
            idx.add(int(m.group(1)))
    return sorted(idx)


# -----------------------------
# 核心预测逻辑
# -----------------------------
def estimate_sum_rate_nextstep(
        env: RelayMoveEnvShp,
        relay_pos_next: np.ndarray,
        step_idx_next: int,
        use_sa_bw: bool,
        sa_cfg: Optional[SAConfig] = None,
) -> float:
    """
    预测下一步的 Sum Rate。
    逻辑与 Environment v4 完全一致。
    """
    nR = env.n_relays
    nS = env.n_survey

    # 1. 预测 Survey 位置
    survey_pos_next = np.zeros((nS, 3), dtype=float)
    for s in range(nS):
        path = env.survey_paths.get(s, [])
        j = min(step_idx_next, len(path) - 1)
        lat, lon, h = path[j]
        survey_pos_next[s] = np.array([lat, lon, h], dtype=float)

    # 2. 计算 Relay -> Ground 链路
    pr_rg_all = np.zeros(nR, dtype=float)
    for r in range(nR):
        d_rg_h = haversine_m(relay_pos_next[r, 0], relay_pos_next[r, 1], env.ground_pos[0], env.ground_pos[1])
        dz_rg = abs(float(relay_pos_next[r, 2] - env.ground_pos[2]))

        if env.use_meas_fit:
            c_ref = env._poly2_eval_bps(env.beta_ag, d_rg_h, dz_rg, env.bounds_ag)
            pr_rg_all[r] = env._pr_w_from_fitted_rate(c_ref)
        else:
            d_rg = env._distance_3d_m(relay_pos_next[r], env.ground_pos)
            pr_rg_all[r] = env._rx_power_w(d_rg)

    # 3. 计算 Survey -> Relay 链路并选择最佳中继
    base_bw = env.total_bandwidth_hz / max(nS, 1)
    sel = np.full(nS, -1, dtype=int)
    pr_sr_sel = np.zeros(nS, dtype=float)
    pr_rg_sel = np.zeros(nS, dtype=float)

    for s in range(nS):
        best_val = -1.0
        best_r = -1
        best_pr_sr = 0.0
        best_pr_rg = 0.0

        for r in range(nR):
            d_sr_h = haversine_m(survey_pos_next[s, 0], survey_pos_next[s, 1], relay_pos_next[r, 0],
                                 relay_pos_next[r, 1])
            dz_sr = abs(float(survey_pos_next[s, 2] - relay_pos_next[r, 2]))

            if env.use_meas_fit:
                c_ref = env._poly2_eval_bps(env.beta_aa, d_sr_h, dz_sr, env.bounds_aa)
                pr_sr = env._pr_w_from_fitted_rate(c_ref)
            else:
                d_sr = env._distance_3d_m(survey_pos_next[s], relay_pos_next[r])
                pr_sr = env._rx_power_w(d_sr)

            pr_rg = pr_rg_all[r]

            val = env._two_hop_rate_bps(base_bw, pr_sr, pr_rg)
            if val > best_val:
                best_val = val
                best_r = r
                best_pr_sr = pr_sr
                best_pr_rg = pr_rg

        sel[s] = best_r
        pr_sr_sel[s] = best_pr_sr
        pr_rg_sel[s] = best_pr_rg

    # 4. 带宽分配 (SA vs 均分)
    if not use_sa_bw:
        bw = np.full(nS, env.total_bandwidth_hz / max(nS, 1), dtype=float)
    else:
        # 使用与 PPO 环境一致的 SA 逻辑
        demand = np.full(nS, env.demand_per_link_bps, dtype=float)
        pr_eff = env._pr_eff_for_sa(pr_sr_sel, pr_rg_sel)

        cfg = sa_cfg if sa_cfg is not None else env.sa_cfg
        init = np.full(nS, env.total_bandwidth_hz / max(nS, 1), dtype=float)

        bw = simulated_annealing(
            total_bandwidth_hz=env.total_bandwidth_hz,
            throughput_demand_bps=demand,
            rx_power_w=pr_eff,
            noise_density_w_per_hz=env.radio.noise_density_w_per_hz,
            init_solution_hz=init,
            config=cfg,
            seed=0,
        )

    # 5. 汇总速率
    rates = np.zeros(nS, dtype=float)
    for s in range(nS):
        if sel[s] >= 0 and pr_sr_sel[s] > 0 and pr_rg_sel[s] > 0:
            rates[s] = env._two_hop_rate_bps(float(bw[s]), float(pr_sr_sel[s]), float(pr_rg_sel[s]))

    return float(np.sum(rates))


def action_to_reach_target(env: RelayMoveEnvShp, target_latlonalt: np.ndarray) -> np.ndarray:
    """计算归一化动作"""
    target = np.asarray(target_latlonalt, dtype=np.float64)
    act = np.zeros((env.n_relays, 3), dtype=np.float64)

    for r in range(env.n_relays):
        cur_lat, cur_lon, cur_alt = map(float, env.relay_pos[r])
        tgt_lat, tgt_lon, tgt_alt = map(float, target[r])

        m_per_deg_lat = 111320.0
        m_per_deg_lon = 111320.0 * float(np.cos(np.deg2rad(cur_lat)))

        dy_n = (tgt_lat - cur_lat) * m_per_deg_lat
        dx_e = (tgt_lon - cur_lon) * max(m_per_deg_lon, 1e-6)
        dz_u = (tgt_alt - cur_alt)

        act[r, 0] = np.clip(dx_e / env.max_speed, -1.0, 1.0)
        act[r, 1] = np.clip(dy_n / env.max_speed, -1.0, 1.0)
        act[r, 2] = np.clip(dz_u / env.max_v_speed, -1.0, 1.0)

    return act.reshape(-1).astype(np.float32)


# -----------------------------
# Controllers
# -----------------------------
class BaseController:
    def reset(self, env): pass

    def act(self, env, obs): raise NotImplementedError


class SGCController(BaseController):
    def act(self, env, obs):
        return np.zeros(env.n_relays * 3, dtype=np.float32)


class GSTController(BaseController):
    """Greedy Signal Tracking - NOW USES SA"""

    def __init__(self, max_joint: int = 2000):
        xy_dirs = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (0.7, 0.7), (0.7, -0.7), (-0.7, 0.7), (-0.7, -0.7)]
        z_dirs = [-1.0, 0.0, 1.0]

        self.per_relay_dirs = [(dx, dy, dz) for (dx, dy) in xy_dirs for dz in z_dirs]
        self.max_joint = max_joint
        self.joint_actions_cache = {}

        # [修改] 使用与 Env 中完全一致的 SA 参数
        self.sa_cfg = SAConfig(t_init=5.0, t_min=1e-3, alpha=0.97, n_iter=120, step_scale=0.12, bw_min=1e6)

    def _joint_actions(self, n_relays):
        if n_relays in self.joint_actions_cache:
            return self.joint_actions_cache[n_relays]
        combos = list(product(self.per_relay_dirs, repeat=n_relays))
        if len(combos) > self.max_joint:
            rng = np.random.default_rng(0)
            sel = rng.choice(len(combos), size=self.max_joint, replace=False)
            combos = [combos[i] for i in sel]
        acts = np.array([[v for pair in combo for v in pair] for combo in combos], dtype=np.float32)
        self.joint_actions_cache[n_relays] = acts
        return acts

    def act(self, env, obs):
        cand = self._joint_actions(env.n_relays)
        best_val = -1e30
        best_a = cand[0]
        step_next = int(env.t + 1)
        cur = env.relay_pos.copy()

        for a in cand:
            raw = a.astype(np.float64).reshape(env.n_relays, 3)
            clipped = np.clip(raw, -1.0, 1.0)
            relay_next = cur.copy()
            for r in range(env.n_relays):
                dx_e = float(clipped[r, 0]) * env.max_speed
                dy_n = float(clipped[r, 1]) * env.max_speed
                dz_u = float(clipped[r, 2]) * env.max_v_speed

                dlat, dlon = env._enu_m_to_latlon_deg(float(relay_next[r, 0]), dx_e, dy_n)
                relay_next[r, 0] += dlat
                relay_next[r, 1] += dlon
                relay_next[r, 2] += dz_u

            relay_next[:, 0] = np.clip(relay_next[:, 0], env.lat_min, env.lat_max)
            relay_next[:, 1] = np.clip(relay_next[:, 1], env.lon_min, env.lon_max)
            relay_next[:, 2] = np.clip(relay_next[:, 2], env.min_alt, env.max_alt)

            # [确认] 启用 use_sa_bw=True
            val = estimate_sum_rate_nextstep(
                env,
                relay_next,
                step_next,
                use_sa_bw=True,
                sa_cfg=self.sa_cfg
            )

            if val > best_val:
                best_val = val
                best_a = a
        return best_a.astype(np.float32)


@dataclass
class PSOParams:
    n_particles: int = 15
    n_iters: int = 10
    w: float = 0.6
    c1: float = 1.6
    c2: float = 1.6
    vmax_deg: float = 0.0004
    vmax_alt_m: float = 5.0


class PSOController(BaseController):
    def __init__(self, params: PSOParams = PSOParams()):
        self.p = params
        self.sa_cfg = SAConfig(t_init=5.0, t_min=1e-3, alpha=0.97, n_iter=120, step_scale=0.12, bw_min=1e6)

    def act(self, env, obs):
        rng = np.random.default_rng(0 + int(env.t))
        step_next = int(env.t + 1)

        x = np.repeat(env.relay_pos[None, :, :], self.p.n_particles, axis=0)
        x[:, :, 0] += rng.normal(0, 0.001, size=(self.p.n_particles, env.n_relays))
        x[:, :, 1] += rng.normal(0, 0.001, size=(self.p.n_particles, env.n_relays))
        x[:, :, 2] += rng.normal(0, 5.0, size=(self.p.n_particles, env.n_relays))

        x[:, :, 0] = np.clip(x[:, :, 0], env.lat_min, env.lat_max)
        x[:, :, 1] = np.clip(x[:, :, 1], env.lon_min, env.lon_max)
        x[:, :, 2] = np.clip(x[:, :, 2], env.min_alt, env.max_alt)

        v = np.zeros_like(x)
        pbest = x.copy()
        pbest_val = np.full(self.p.n_particles, -1e30, dtype=float)

        def fitness(pos):
            return estimate_sum_rate_nextstep(env, pos, step_next, use_sa_bw=True, sa_cfg=self.sa_cfg)

        for i in range(self.p.n_particles):
            pbest_val[i] = fitness(x[i])

        g_idx = int(np.argmax(pbest_val))
        gbest = pbest[g_idx].copy()
        gbest_val = float(pbest_val[g_idx])

        for _ in range(self.p.n_iters):
            r1 = rng.random(size=x.shape)
            r2 = rng.random(size=x.shape)
            v = (self.p.w * v + self.p.c1 * r1 * (pbest - x) + self.p.c2 * r2 * (gbest[None, :, :] - x))

            v[:, :, 0] = np.clip(v[:, :, 0], -self.p.vmax_deg, self.p.vmax_deg)
            v[:, :, 1] = np.clip(v[:, :, 1], -self.p.vmax_deg, self.p.vmax_deg)
            v[:, :, 2] = np.clip(v[:, :, 2], -self.p.vmax_alt_m, self.p.vmax_alt_m)

            x = x + v
            x[:, :, 0] = np.clip(x[:, :, 0], env.lat_min, env.lat_max)
            x[:, :, 1] = np.clip(x[:, :, 1], env.lon_min, env.lon_max)
            x[:, :, 2] = np.clip(x[:, :, 2], env.min_alt, env.max_alt)

            for i in range(self.p.n_particles):
                val = fitness(x[i])
                if val > pbest_val[i]:
                    pbest_val[i] = val
                    pbest[i] = x[i].copy()

            g_idx = int(np.argmax(pbest_val))
            if pbest_val[g_idx] > gbest_val:
                gbest_val = float(pbest_val[g_idx])
                gbest = pbest[g_idx].copy()

        return action_to_reach_target(env, gbest)


# -----------------------------
# Main Run & Plotting
# -----------------------------
def make_env(init_relay_pos: np.ndarray) -> RelayMoveEnvShp:
    print(f"DEBUG: Initializing Env with Meas Fit = True")
    print(f"DEBUG: Aligning Constraints with PPO Training Config...")
    return RelayMoveEnvShp(
        shapefile_path=SHAPEFILE_PATH,
        n_relays=N_RELAYS,
        n_survey=None,
        batch_list=None,
        total_bandwidth_hz=TOTAL_BW_HZ,
        demand_bps=DEMAND_BPS,
        max_speed_m_per_step=MAX_SPEED_M_PER_STEP,  # 20.0
        max_v_speed_m_per_step=MAX_V_SPEED_M_PER_STEP,  # 4.0
        min_alt_m=MIN_ALT_M,
        max_alt_m=MAX_ALT_M,
        episode_len=EPISODE_LEN,
        seed=SEED,
        init_relay_pos=init_relay_pos,
        ground_pos=np.array(GROUND_POS, dtype=np.float64),
        use_meas_fit=True,
        air_air_csv=AIR_AIR_CSV,
        air_ground_csv=AIR_GROUND_CSV
    )


def run_once(method: str):
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"\n====== Running {method.upper()} Baseline (ALIGNED) ======")

    if method.upper() == "SGC":
        controller = SGCController()
    elif method.upper() == "GST":
        controller = GSTController()
    elif method.upper() == "PSO":
        controller = PSOController()
    else:
        raise ValueError("Unknown method")

    env = make_env(np.array(INIT_RELAY_POS))
    env.test_mode = True

    obs, _ = env.reset()
    controller.reset(env)

    # 运行仿真
    for t in range(EPISODE_LEN):
        action = controller.act(env, obs)
        obs, reward, term, trunc, info = env.step(action)

        if (t + 1) % 50 == 0:
            rate = info.get('sum_rate_bps', 0)
            print(f"Step {t + 1}/{EPISODE_LEN} | SumRate: {rate / 1e6:.2f} Mbps")
        if term or trunc:
            break

    df = pd.DataFrame(env.log_data)
    csv_path = os.path.join(OUT_DIR, f"{method.upper()}_test_log.csv")
    df.to_csv(csv_path, index=False)
    print(f"[{method.upper()}] Saved CSV: {csv_path}")

    relay_ids = _infer_indices(df.columns, "relay", "_lat")
    survey_ids = _infer_indices(df.columns, "survey", "_lat")

    # =========================================================
    # 绘图部分 (Updated)
    # =========================================================

    # 1. 2D 轨迹
    plt.figure(figsize=(10, 6))
    for i in relay_ids:
        plt.plot(df[f"relay{i}_lon"], df[f"relay{i}_lat"], label=f"Relay {i}", linewidth=2)
        plt.scatter(df[f"relay{i}_lon"].iloc[0], df[f"relay{i}_lat"].iloc[0], marker='o', s=100)
    for s in survey_ids:
        plt.plot(df[f"survey{s}_lon"], df[f"survey{s}_lat"], "--", alpha=0.5, label=f"Survey {s}")
    plt.scatter([GROUND_POS[1]], [GROUND_POS[0]], marker="*", s=300, c='gold', edgecolors='k', label="Ground Station")
    plt.title(f"{method.upper()} (Aligned) - 2D Trajectories")
    plt.xlabel("Longitude");
    plt.ylabel("Latitude")
    plt.legend();
    plt.grid(True)
    plt.savefig(os.path.join(OUT_DIR, f"{method.upper()}_trajectory.png"), dpi=300)
    plt.close()

    # 2. 高度变化
    plt.figure(figsize=(10, 4))
    for i in relay_ids:
        col = f"relay{i}_alt_m"
        if col in df.columns:
            plt.plot(df["step"], df[col], label=f"Relay {i} Altitude", linewidth=2)
    plt.title(f"{method.upper()} (Aligned) - Relay Altitude")
    plt.xlabel("Step");
    plt.ylabel("Altitude (m)")
    plt.legend();
    plt.grid(True)
    plt.savefig(os.path.join(OUT_DIR, f"{method.upper()}_altitude.png"), dpi=300)
    plt.close()

    # 3. 吞吐量 (Fixed: Added Individual Rates)
    plt.figure(figsize=(10, 4))
    # 先画各条链路
    for s in survey_ids:
        col_rate = f"rate_{s}_bps"
        if col_rate in df.columns:
            # 加上透明度 alpha=0.6 防止遮挡 Sum Rate
            plt.plot(df["step"], df[col_rate] / 1e6, label=f"Rate Survey {s}", alpha=0.6)

    # 再画总速率
    if "sum_rate_bps" in df.columns:
        total_rate_mbps = df["sum_rate_bps"].to_numpy(dtype=float) / 1e6
        plt.plot(df["step"], total_rate_mbps, 'r-', linewidth=2.5, label="Total Sum Rate")
        finite_rates = total_rate_mbps[np.isfinite(total_rate_mbps)]
        if finite_rates.size > 0:
            mean_rate_mbps = float(np.mean(finite_rates))
            plt.axhline(
                y=mean_rate_mbps,
                color='r',
                linestyle='--',
                linewidth=2.0,
                alpha=0.85,
                label=f"Total Mean Fit ({mean_rate_mbps:.2f} Mbps)",
            )

    plt.title(f"{method.upper()} (Aligned) - Throughput (Mbps)")
    plt.xlabel("Step");
    plt.ylabel("Mbps")
    plt.legend(loc='best', fontsize='small');
    plt.grid(True)
    plt.savefig(os.path.join(OUT_DIR, f"{method.upper()}_throughput.png"), dpi=300)
    plt.close()

    # 4. RSSI (Added)
    plt.figure(figsize=(10, 4))
    for s in survey_ids:
        col_rssi = f"rssi_{s}_dbm"
        if col_rssi in df.columns:
            plt.plot(df["step"], df[col_rssi], label=f"RSSI Survey {s}")

    # 画一条灵敏度参考线
    # plt.axhline(y=-110, color='k', linestyle=':', label="Sensitivity Limit")
    plt.title(f"{method.upper()} (Aligned) - Signal Strength (RSSI)")
    plt.xlabel("Step");
    plt.ylabel("RSSI (dBm)")
    plt.legend(fontsize='small');
    plt.grid(True)
    plt.savefig(os.path.join(OUT_DIR, f"{method.upper()}_rssi.png"), dpi=300)
    plt.close()

    print(f"Results saved to {OUT_DIR}")
    env.close()


if __name__ == "__main__":
    run_once(METHOD)
