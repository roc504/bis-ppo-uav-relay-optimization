# -*- coding: utf-8 -*-
"""
Env_matplot_relaymove_shp_v4.py

Shapefile-driven multi-survey-UAV environment (Gymnasium).

REWARD FUNCTION UPDATE (Fix for Problem 4):
- Removed "Differential Reward" (Zero-Sum Trap).
- Implemented "Absolute Value Reward" based on physical limits.
- Components:
  1. Rate Reward: Normalized by estimated max capacity.
  2. RSSI Reward: Normalized [-110, -40] -> [0, 1].
  3. Action Penalty: Penalize large movements to prevent jitter.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import geopandas as gpd
import pandas as pd
from DR_SA_g import simulated_annealing, SAConfig

'''
"uav_id"字段
单无人机带宽判定，<=1时把全部带宽给唯一用户
绝对值奖励机制
基于模型计算通信容量，使用实测数据
'''
@dataclass
class RadioParams:
    tx_power_dbm: float = 30.0
    noise_density_w_per_hz: float = 1e-20
    pathloss_exp: float = 2.2
    pl0_db: float = 32.4
    shadow_std_db: float = 2.0


def dbm_to_w(dbm: float) -> float:
    return 10 ** ((dbm - 30.0) / 10.0)


def path_loss_db(d_m: float, pl0_db: float, n: float) -> float:
    d = max(float(d_m), 1.0)
    return float(pl0_db + 10.0 * n * np.log10(d))


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6371000.0
    phi1 = np.deg2rad(lat1)
    phi2 = np.deg2rad(lat2)
    dphi = np.deg2rad(lat2 - lat1)
    dlmb = np.deg2rad(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlmb / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return float(R * c)


def _read_gdf(shapefile_path: str) -> gpd.GeoDataFrame:
    os.environ.setdefault("SHAPE_RESTORE_SHX", "YES")
    return gpd.read_file(shapefile_path)


def _unique_ints(series) -> List[int]:
    vals = [int(v) for v in series.dropna().tolist()]
    return sorted(list(set(vals)))


def get_unique_batches(shapefile_path: str) -> List[int]:
    gdf = _read_gdf(shapefile_path)
    if "batch" in gdf.columns:
        vals = _unique_ints(gdf["batch"])
        return vals if vals else [0]
    return [0]


def generate_mapped_paths(
        shapefile_path: str,
        n_survey: Optional[int] = None,
        batch_list: Optional[List[int]] = None,
        default_alt_m: float = 100.0,
) -> Tuple[Dict[int, List[Tuple[float, float, float]]], List[int]]:
    gdf = _read_gdf(shapefile_path)

    has_uav = "uav_id" in gdf.columns
    has_batch = "batch" in gdf.columns

    if has_uav:
        uav_ids = _unique_ints(gdf["uav_id"])
        if n_survey is None:
            n_survey = len(uav_ids) if uav_ids else 1
    else:
        uav_ids = []
        if n_survey is None:
            n_survey = 1

    n_survey = int(n_survey)
    if n_survey <= 0:
        raise ValueError("n_survey must be >= 1")

    if has_uav and uav_ids:
        if len(uav_ids) >= n_survey:
            uav_ids = uav_ids[:n_survey]
        else:
            max_id = max(uav_ids)
            uav_ids = uav_ids + list(range(max_id + 1, max_id + 1 + (n_survey - len(uav_ids))))
    else:
        uav_ids = list(range(n_survey))

    if batch_list is None:
        batch_list = get_unique_batches(shapefile_path) if has_batch else [0]

    paths: Dict[int, List[Tuple[float, float, float]]] = {i: [] for i in range(n_survey)}

    if has_uav:
        if has_batch:
            for batch in batch_list:
                selected = gdf[gdf["batch"] == batch]
                if "step" in selected.columns:
                    selected = selected.sort_values("step")

                for internal_idx, real_uav in enumerate(uav_ids):
                    drone_path = selected[selected["uav_id"] == real_uav]
                    if "step" in drone_path.columns:
                        drone_path = drone_path.sort_values("step")
                    for _, row in drone_path.iterrows():
                        pt = row["geometry"]
                        paths[internal_idx].append((float(pt.y), float(pt.x), float(default_alt_m)))
        else:
            selected = gdf
            if "step" in selected.columns:
                selected = selected.sort_values("step")
            for internal_idx, real_uav in enumerate(uav_ids):
                drone_path = selected[selected["uav_id"] == real_uav]
                if "step" in drone_path.columns:
                    drone_path = drone_path.sort_values("step")
                for _, row in drone_path.iterrows():
                    pt = row["geometry"]
                    paths[internal_idx].append((float(pt.y), float(pt.x), float(default_alt_m)))
        return paths, uav_ids

    coords: List[Tuple[float, float, float]] = []
    for _, row in gdf.iterrows():
        pt = row["geometry"]
        coords.append((float(pt.y), float(pt.x), float(default_alt_m)))
    if not coords:
        raise ValueError("Shapefile contains no point geometries.")
    for idx, p in enumerate(coords):
        paths[idx % n_survey].append(p)
    return paths, uav_ids


class RelayMoveEnvShp(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
            self,
            shapefile_path: str,
            n_relays: int = 2,
            n_survey: Optional[int] = None,
            batch_list: Optional[List[int]] = None,
            total_bandwidth_hz: float = 20e6,
            demand_bps: float = 12e6,
            max_speed_m_per_step: float = 20.0,
            # 新增：垂直最大速度 (建议设为水平速度的 1/4 到 1/5)
            max_v_speed_m_per_step: float = 4.0,
            episode_len: int = 300,
            seed: Optional[int] = None,
            init_relay_pos: Optional[np.ndarray] = None,
            ground_pos: Optional[np.ndarray] = None,
            default_survey_alt_m: float = 100.0,
            two_hop_mode: str = "df",
            # 新增高度限制参数
            min_alt_m: float = 50.0,
            max_alt_m: float = 300.0,
            air_air_csv: Optional[str] = None,
            air_ground_csv: Optional[str] = None,
            bw_ref_hz: float = 20e6,  # 实测数据对应的参考带宽
            rate_unit: str = "Mbps",  # 你的CSV第三列看起来像Mbps（0~448）
            use_meas_fit: bool = True,
            # ---【修改点 1：新增参数】---
            bw_alloc_strategy: str = "sa"  # 默认使用 SA，可选 'equal'

    ):
        super().__init__()
        self.two_hop_mode = two_hop_mode
        self.shapefile_path = str(shapefile_path)
        self.n_relays = int(n_relays)
        self.batch_list = batch_list

        self.total_bandwidth_hz = float(total_bandwidth_hz)
        self.demand_per_link_bps = float(demand_bps)
        self.max_speed = float(max_speed_m_per_step)  # 水平
        self.max_v_speed = float(max_v_speed_m_per_step)  # 垂直 (新增)
        self.episode_len = int(episode_len)
        # 新增：高度限制
        self.min_alt = float(min_alt_m)
        self.max_alt = float(max_alt_m)
        self.rng = np.random.default_rng(seed)

        self.log_data: List[Dict] = []
        self.test_mode: bool = False

        self.radio = RadioParams()

        self.use_meas_fit = bool(use_meas_fit)
        self.bw_ref_hz = float(bw_ref_hz)
        self.rate_unit = str(rate_unit)
        self.air_air_csv = air_air_csv
        self.air_ground_csv = air_ground_csv
        self.beta_aa = None
        self.beta_ag = None
        self.bounds_aa = None
        self.bounds_ag = None

        self.use_meas_fit = True
        if self.use_meas_fit:
            if (self.air_air_csv is None) or (self.air_ground_csv is None):
                raise ValueError("use_meas_fit=True requires air_air_csv and air_ground_csv paths.")

            self.beta_aa, self.bounds_aa = self._load_poly2_from_csv(self.air_air_csv)
            self.beta_ag, self.bounds_ag = self._load_poly2_from_csv(self.air_ground_csv)

        # Warm start config for SA
        self.sa_cfg = SAConfig(t_init=5.0, t_min=1e-3, alpha=0.97, n_iter=120, step_scale=0.12, bw_min=1e6)

        self.survey_paths, self.uav_ids = generate_mapped_paths(
            self.shapefile_path, n_survey=n_survey, batch_list=self.batch_list, default_alt_m=default_survey_alt_m
        )
        self.n_survey = len(self.survey_paths)

        self.max_path_len = max((len(v) for v in self.survey_paths.values()), default=0)
        if self.max_path_len == 0:
            raise ValueError("Survey paths are empty. Check shapefile.")

        all_pts = np.array([p[:2] for i in range(self.n_survey) for p in self.survey_paths[i]], dtype=np.float64)
        self.lat_min = float(np.min(all_pts[:, 0])) - 0.01
        self.lat_max = float(np.max(all_pts[:, 0])) + 0.01
        self.lon_min = float(np.min(all_pts[:, 1])) - 0.01
        self.lon_max = float(np.max(all_pts[:, 1])) + 0.01

        if init_relay_pos is None:
            self._init_relay_pos_backup = None  # 标记为无备份
            self.relay_pos = np.zeros((self.n_relays, 3), dtype=np.float64)
            self.relay_pos[:, 0] = self.rng.uniform(self.lat_min, self.lat_max, size=self.n_relays)
            self.relay_pos[:, 1] = self.rng.uniform(self.lon_min, self.lon_max, size=self.n_relays)
            # self.relay_pos[:, 2] = 150.0
            # 初始化高度在 min 和 max 之间
            self.relay_pos[:, 2] = self.rng.uniform(self.min_alt, self.max_alt, size=self.n_relays)
        else:
            arr = np.asarray(init_relay_pos, dtype=np.float64)
            if arr.shape != (self.n_relays, 3):
                raise ValueError("init_relay_pos must have shape (n_relays, 3)")
            self.relay_pos = arr.copy()
            self._init_relay_pos_backup = arr.copy()  # 备份固定位置

        if ground_pos is None:
            self.ground_pos = np.array([all_pts[0, 0], all_pts[0, 1], 0.0], dtype=np.float64)
        else:
            gp = np.asarray(ground_pos, dtype=np.float64)
            self.ground_pos = gp.copy()

        self.survey_pos = np.zeros((self.n_survey, 3), dtype=np.float64)

        # Persistent state for Warm Start
        self.link_bandwidth_hz = np.full(self.n_survey, self.total_bandwidth_hz / max(self.n_survey, 1),
                                         dtype=np.float64)
        self.sel_relay = np.full(self.n_survey, -1, dtype=np.int32)

        self.t = 0
        # self.prev_sum_rate removed (no longer needed for absolute reward)

        # 修改：动作空间变为 3 * n_relays (Lat, Lon, Alt)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_relays * 3,), dtype=np.float32)

        # 修改：观测空间维度增加 (Relay Lat/Lon/Alt)
        # Relay: 3 * N, Survey Lat/Lon: 2 * M (Survey高度通常固定，若Survey也变则需3*M), RSSI: M, Rate: M
        # 为保持稳健，这里将 Relay 的观测改为全 3D 坐标
        # obs_dim = self.n_relays * 3 + self.n_survey + self.n_survey + self.n_survey * 2
        # self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        # RelayState(3*NR) + RelativeVecs(3*NR*NS) + RSSI(NS) + Rate(NS)
        obs_dim = (self.n_relays * 3) + (self.n_relays * self.n_survey * 3) + self.n_survey + self.n_survey
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        # -----------------------------------------------------------
        # NEW: Reward Normalization Constants
        # -----------------------------------------------------------
        # 1. Rate Reward Limits
        # Estimate max possible rate: 20MHz BW * ~6 bps/Hz (high SNR) = ~120 Mbps.
        # We set a soft cap for normalization. If it exceeds, reward > 1.0, which is fine.
        # Let's set it to Sum Demand (e.g. 12Mbps * N) or a fixed physical capacity.
        # Using 50 Mbps as a scaling factor (so 50Mbps -> reward component = 1.0)
        self.rew_norm_rate_bps = 50e6

        # 2. RSSI Reward Limits (dBm)
        self.rew_rssi_min = -110.0  # Sensitivity floor
        self.rew_rssi_max = -40.0  # Saturation/Near field

        # 3. Weights
        self.w_rate = 1.0
        self.w_rssi = 0.5
        self.w_penalty = 0.1

        # ---【修改点 2：保存策略】---
        self.bw_alloc_strategy = bw_alloc_strategy.lower()
        if self.bw_alloc_strategy not in ["sa", "equal"]:
            raise ValueError("bw_alloc_strategy must be 'sa' or 'equal'")

    def _clip_relays(self):
        self.relay_pos[:, 0] = np.clip(self.relay_pos[:, 0], self.lat_min, self.lat_max)
        self.relay_pos[:, 1] = np.clip(self.relay_pos[:, 1], self.lon_min, self.lon_max)
        # 新增：高度限制
        self.relay_pos[:, 2] = np.clip(self.relay_pos[:, 2], self.min_alt, self.max_alt)
    def _enu_m_to_latlon_deg(self, lat_deg: float, dx_e: float, dy_n: float) -> Tuple[float, float]:
        m_per_deg_lat = 111320.0
        m_per_deg_lon = 111320.0 * float(np.cos(np.deg2rad(lat_deg)))
        dlat = dy_n / m_per_deg_lat
        dlon = dx_e / max(m_per_deg_lon, 1e-6)
        return float(dlat), float(dlon)

    def _distance_3d_m(self, a: np.ndarray, b: np.ndarray) -> float:
        horiz = haversine_m(a[0], a[1], b[0], b[1])
        dz = float(a[2] - b[2])
        return float(np.sqrt(horiz * horiz + dz * dz))

    def _rx_power_w(self, d_m: float) -> float:
        pl = path_loss_db(d_m, self.radio.pl0_db, self.radio.pathloss_exp)
        shadow = self.rng.normal(0.0, self.radio.shadow_std_db)
        pr_dbm = self.radio.tx_power_dbm - (pl + shadow)
        return float(dbm_to_w(pr_dbm))

    def _rssi_dbm(self, d_m: float) -> float:
        pl = path_loss_db(d_m, self.radio.pl0_db, self.radio.pathloss_exp)
        shadow = self.rng.normal(0.0, self.radio.shadow_std_db)
        pr_dbm = self.radio.tx_power_dbm - (pl + shadow)
        return float(pr_dbm)

    def _load_poly2_from_csv(self, csv_path: str):
        """
        CSV: col0=x(horizontal, m), col1=y(vertical, m), col2=rate (Mbps or bps)
        Fit: C(x,y)=b0+b1 x + b2 y + b3 x^2 + b4 xy + b5 y^2
        """
        df = pd.read_csv(csv_path, header=None)
        x = df.iloc[:, 0].to_numpy(dtype=np.float64)
        y = df.iloc[:, 1].to_numpy(dtype=np.float64)
        r = df.iloc[:, 2].to_numpy(dtype=np.float64)

        # unit convert
        if self.rate_unit.lower() == "mbps":
            r_bps = r * 1e6
        else:
            r_bps = r

        A = np.column_stack([np.ones_like(x), x, y, x ** 2, x * y, y ** 2])
        beta, *_ = np.linalg.lstsq(A, r_bps, rcond=None)

        # bounds for safe clipping (avoid wild extrapolation)
        bounds = {
            "x_min": float(np.min(x)), "x_max": float(np.max(x)),
            "y_min": float(np.min(y)), "y_max": float(np.max(y)),
        }
        return beta.astype(np.float64), bounds

    def _poly2_eval_bps(self, beta: np.ndarray, x_m: float, y_m: float, bounds: dict) -> float:
        # clip to measured range to avoid crazy extrapolation
        x = float(np.clip(x_m, bounds["x_min"], bounds["x_max"]))
        y = float(np.clip(y_m, bounds["y_min"], bounds["y_max"]))
        feat = np.array([1.0, x, y, x * x, x * y, y * y], dtype=np.float64)
        val = float(feat @ beta)
        return float(max(val, 0.0))  # avoid negative predicted rate

    def _pr_w_from_fitted_rate(self, c_ref_bps: float) -> float:
        """
        Convert fitted rate at B_ref to effective received power (W),
        so that Shannon with B_ref yields approximately c_ref_bps.
        """
        # spectral efficiency (bps/Hz)
        se = float(c_ref_bps) / max(self.bw_ref_hz, 1e-12)

        # avoid overflow in 2**se if se is huge
        se = float(np.clip(se, 0.0, 25.0))

        snr = (2.0 ** se) - 1.0
        pr = snr * self.radio.noise_density_w_per_hz * self.bw_ref_hz

        # optional: keep randomness similar to your shadowing setting
        if self.radio.shadow_std_db > 0:
            shadow_db = float(self.rng.normal(0.0, self.radio.shadow_std_db))
            pr *= 10.0 ** (shadow_db / 10.0)

        return float(max(pr, 0.0))

    def _rssi_dbm_from_pr(self, pr_w: float) -> float:
        return float(10.0 * np.log10(max(pr_w, 1e-30)) + 30.0)

    def _rate_bps(self, bw_hz: float, pr_w: float) -> float:
        bw = max(float(bw_hz), 1e-12)
        noise = self.radio.noise_density_w_per_hz * bw
        snr = pr_w / max(noise, 1e-12)
        return float(bw * np.log2(1.0 + snr))

    def _two_hop_rate_bps(self, bw_hz: float, pr_sr_w: float, pr_rg_w: float) -> float:
        c_sr = self._rate_bps(bw_hz, pr_sr_w)
        c_rg = self._rate_bps(bw_hz, pr_rg_w)
        if self.two_hop_mode == "df":
            return float((c_sr * c_rg) / (c_sr + c_rg + 1e-12))
        else:
            return float(min(c_sr, c_rg))

    def _pr_eff_for_sa(self, pr_sr: np.ndarray, pr_rg: np.ndarray) -> np.ndarray:
        """
        给 SA 用的“等效接收功率”：
        - two_hop_mode != df: 退化为 min 模型（保持你原逻辑）
        - two_hop_mode == df: 先算 DF 端到端容量，再反推等效功率
        """
        pr_sr = np.asarray(pr_sr, dtype=np.float64)
        pr_rg = np.asarray(pr_rg, dtype=np.float64)

        if self.two_hop_mode != "df":
            return np.minimum(pr_sr, pr_rg)

        # 1) 用参考带宽算两跳容量
        B = float(self.bw_ref_hz)
        c_sr = np.array([self._rate_bps(B, p) for p in pr_sr], dtype=np.float64)
        c_rg = np.array([self._rate_bps(B, p) for p in pr_rg], dtype=np.float64)

        # 2) DF 端到端（最优分时的 harmonic mean 形式）
        c_df = (c_sr * c_rg) / (c_sr + c_rg + 1e-12)

        # 3) 反推等效功率：rate_bps(B, pr_eff) = c_df
        se = c_df / max(B, 1e-12)  # bps/Hz
        se = np.clip(se, 0.0, 25.0)  # 防止 2**se 溢出
        snr = (2.0 ** se) - 1.0
        pr_eff = snr * self.radio.noise_density_w_per_hz * B
        return pr_eff
    def _set_survey_positions_by_step(self, step_idx: int):
        for s in range(self.n_survey):
            path = self.survey_paths.get(s, [])
            j = min(step_idx, len(path) - 1)
            lat, lon, h = path[j]
            self.survey_pos[s] = np.array([lat, lon, h], dtype=np.float64)

    def _select_relays(self):
        base_bw = self.total_bandwidth_hz / max(self.n_survey, 1)

        sel = np.full(self.n_survey, -1, dtype=np.int32)
        rssi_sr = np.full(self.n_survey, -140.0, dtype=np.float64)
        pr_sr = np.zeros(self.n_survey, dtype=np.float64)
        pr_rg = np.zeros(self.n_survey, dtype=np.float64)

        pr_rg_all = np.zeros(self.n_relays, dtype=np.float64)
        for r in range(self.n_relays):
            # d_rg = self._distance_3d_m(self.relay_pos[r], self.ground_pos)
            # pr_rg_all[r] = self._rx_power_w(d_rg)
            # horizontal + vertical to match CSV definition
            d_rg_h = haversine_m(self.relay_pos[r, 0], self.relay_pos[r, 1], self.ground_pos[0], self.ground_pos[1])
            dz_rg = abs(float(self.relay_pos[r, 2] - self.ground_pos[2]))

            if self.use_meas_fit:
                c_ref = self._poly2_eval_bps(self.beta_ag, d_rg_h, dz_rg, self.bounds_ag)
                pr_rg_all[r] = self._pr_w_from_fitted_rate(c_ref)
            else:
                d_rg = self._distance_3d_m(self.relay_pos[r], self.ground_pos)
                pr_rg_all[r] = self._rx_power_w(d_rg)

        for s in range(self.n_survey):
            best_val = -1.0
            best_r = -1
            best_rssi = -140.0

            for r in range(self.n_relays):
                # d_sr = self._distance_3d_m(self.survey_pos[s], self.relay_pos[r])
                # pr_sr_w = self._rx_power_w(d_sr)
                # pr_rg_w = pr_rg_all[r]
                # potential = self._two_hop_rate_bps(base_bw, pr_sr_w, pr_rg_w)
                # if potential > best_val:
                #     best_val = potential
                #     best_r = r
                #     best_rssi = self._rssi_dbm(d_sr)
                d_sr_h = haversine_m(self.survey_pos[s, 0], self.survey_pos[s, 1], self.relay_pos[r, 0],
                                     self.relay_pos[r, 1])
                dz_sr = abs(float(self.survey_pos[s, 2] - self.relay_pos[r, 2]))

                if self.use_meas_fit:
                    c_ref = self._poly2_eval_bps(self.beta_aa, d_sr_h, dz_sr, self.bounds_aa)
                    pr_sr_w = self._pr_w_from_fitted_rate(c_ref)
                    rssi_sr_dbm = self._rssi_dbm_from_pr(pr_sr_w)
                else:
                    d_sr = self._distance_3d_m(self.survey_pos[s], self.relay_pos[r])
                    pr_sr_w = self._rx_power_w(d_sr)
                    rssi_sr_dbm = self._rssi_dbm(d_sr)

                pr_rg_w = pr_rg_all[r]
                potential = self._two_hop_rate_bps(base_bw, pr_sr_w, pr_rg_w)

                if potential > best_val:
                    best_val = potential
                    best_r = r
                    best_rssi = rssi_sr_dbm

            sel[s] = best_r
            rssi_sr[s] = best_rssi

            if best_r >= 0:
                # d_sr = self._distance_3d_m(self.survey_pos[s], self.relay_pos[best_r])
                # pr_sr[s] = self._rx_power_w(d_sr)
                # pr_rg[s] = pr_rg_all[best_r]
                if self.use_meas_fit:
                    d_sr_h = haversine_m(self.survey_pos[s, 0], self.survey_pos[s, 1], self.relay_pos[best_r, 0],
                                         self.relay_pos[best_r, 1])
                    dz_sr = abs(float(self.survey_pos[s, 2] - self.relay_pos[best_r, 2]))
                    c_ref = self._poly2_eval_bps(self.beta_aa, d_sr_h, dz_sr, self.bounds_aa)
                    pr_sr[s] = self._pr_w_from_fitted_rate(c_ref)
                else:
                    d_sr = self._distance_3d_m(self.survey_pos[s], self.relay_pos[best_r])
                    pr_sr[s] = self._rx_power_w(d_sr)

                pr_rg[s] = pr_rg_all[best_r]

        return sel, rssi_sr, pr_sr, pr_rg

    def _optimize_bandwidth(self, pr_effective: np.ndarray) -> np.ndarray:
        if self.n_survey <= 1:
            return np.full(self.n_survey, self.total_bandwidth_hz, dtype=np.float64)
        # ---【修改点 3：增加分支逻辑】---
        if self.bw_alloc_strategy == "equal":
            # 策略 A：均分带宽 (Equal)
            # 无论信道如何，每个用户分得总带宽的 1/N
            return np.full(self.n_survey, self.total_bandwidth_hz / max(self.n_survey, 1), dtype=np.float64)

        else:
            # 策略 B：SA 动态优化 (原代码逻辑移动到这里)
            demand = np.full(self.n_survey, self.demand_per_link_bps, dtype=np.float64)
            # Warm Start: Init with previous step's bandwidth
            init = np.clip(self.link_bandwidth_hz, self.sa_cfg.bw_min, self.total_bandwidth_hz)

            bw = simulated_annealing(
                total_bandwidth_hz=self.total_bandwidth_hz,
                throughput_demand_bps=demand,
                rx_power_w=pr_effective,
                noise_density_w_per_hz=self.radio.noise_density_w_per_hz,
                init_solution_hz=init,
                config=self.sa_cfg,
                seed=int(self.rng.integers(0, 2 ** 31 - 1)),
            )
            return bw


    def _get_obs(self, rssi: np.ndarray, rate: np.ndarray) -> np.ndarray:
        # # 修改：获取 Relay 的 3D 坐标 (Lat, Lon, Alt)
        # relay_state = self.relay_pos.reshape(-1)
        # survey_latlon = self.survey_pos[:, :2].reshape(-1)
        # # return np.concatenate([relay_latlon, rssi, rate, survey_latlon], axis=0).astype(np.float32)
        # return np.concatenate([relay_state, rssi, rate, survey_latlon], axis=0).astype(np.float32)
        # 1. Relay 与 地面站 的相对位置 (防止 Relay 飞出地球)
        # 归一化建议：除以一个大致的范围，例如 0.01 度 (约1km)
        scale = 0.01
        rel_relay = (self.relay_pos - self.ground_pos) / scale

        # 2. Survey 与 Relay 的相对位置 (这才是最重要的导航信息！)
        # 形状: (N_relay, N_survey, 3) -> 展平
        # 这告诉 Relay：“目标在你的东北方向 500 米处”
        # 这里我们需要构建每个 Relay 对每个 Survey 的相对向量
        rel_vecs = []
        for r in range(self.n_relays):
            for s in range(self.n_survey):
                # 计算 Relay[r] 到 Survey[s] 的差值
                diff = (self.survey_pos[s] - self.relay_pos[r]) / scale
                rel_vecs.append(diff)

        obs_rel_vecs = np.array(rel_vecs).reshape(-1)

        # 拼贴所有信息
        # [Relay归一化位置, 相对向量, RSSI, Rate]
        obs = np.concatenate([
            rel_relay.reshape(-1).astype(np.float32),
            obs_rel_vecs.astype(np.float32),
            rssi.astype(np.float32),
            rate.astype(np.float32)
        ], axis=0)

        # 必须确保 observation_space 的维度与这里一致
        return obs
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.t = 0
        self.log_data = []
        # prev_sum_rate is removed

        self._set_survey_positions_by_step(0)
        if self._init_relay_pos_backup is not None:
            # 测试模式：有固定点备份，强制回到固定点
            self.relay_pos = self._init_relay_pos_backup.copy()
        else:
            # 训练模式：无备份，全随机重置
            self.relay_pos[:, 0] = self.rng.uniform(self.lat_min, self.lat_max, size=self.n_relays)
            self.relay_pos[:, 1] = self.rng.uniform(self.lon_min, self.lon_max, size=self.n_relays)
            self.relay_pos[:, 2] = self.rng.uniform(self.min_alt, self.max_alt, size=self.n_relays)
        self.link_bandwidth_hz = np.full(self.n_survey, self.total_bandwidth_hz / max(self.n_survey, 1),
                                         dtype=np.float64)

        sel, rssi, pr_sr, pr_rg = self._select_relays()
        self.sel_relay = sel.copy()

        # pr_eff = np.minimum(pr_sr, pr_rg)
        pr_eff = self._pr_eff_for_sa(pr_sr, pr_rg)
        self.link_bandwidth_hz = self._optimize_bandwidth(pr_eff)

        rate = np.zeros(self.n_survey, dtype=np.float64)
        for s in range(self.n_survey):
            if self.sel_relay[s] >= 0 and pr_sr[s] > 0 and pr_rg[s] > 0:
                rate[s] = self._two_hop_rate_bps(self.link_bandwidth_hz[s], pr_sr[s], pr_rg[s])

        return self._get_obs(rssi, rate), {}

    def step(self, action):
        self.t += 1

        # Action is in [-1, 1]. Calculate movement.
        # Save raw action for penalty calculation later.
        # 修改：Action 解析为 (N, 3) -> [dLat_scaled, dLon_scaled, dAlt_scaled]
        raw_action = np.asarray(action, dtype=np.float64).reshape(self.n_relays, 3)

        clipped_action = np.clip(raw_action, -1.0, 1.0)

        for r in range(self.n_relays):
            # 2. 分离水平和垂直控制量
            # x, y 分量使用 max_speed (例如 20m)
            dx_e = float(clipped_action[r, 0]) * self.max_speed
            dy_n = float(clipped_action[r, 1]) * self.max_speed

            # z 分量使用 max_v_speed (例如 4m)
            dz_u = float(clipped_action[r, 2]) * self.max_v_speed

            # 3. 更新位置
            # 水平更新 (将米转换为经纬度)
            dlat, dlon = self._enu_m_to_latlon_deg(self.relay_pos[r, 0], dx_e, dy_n)
            self.relay_pos[r, 0] += dlat
            self.relay_pos[r, 1] += dlon

            # 垂直更新 (直接加米数)
            self.relay_pos[r, 2] += dz_u

        self._clip_relays()
        self._set_survey_positions_by_step(self.t)

        sel, rssi, pr_sr, pr_rg = self._select_relays()
        self.sel_relay = sel.copy()

        # pr_eff = np.minimum(pr_sr, pr_rg)
        pr_eff = self._pr_eff_for_sa(pr_sr, pr_rg)
        self.link_bandwidth_hz = self._optimize_bandwidth(pr_eff)

        rate = np.zeros(self.n_survey, dtype=np.float64)
        for s in range(self.n_survey):
            if self.sel_relay[s] >= 0 and pr_sr[s] > 0 and pr_rg[s] > 0:
                rate[s] = self._two_hop_rate_bps(self.link_bandwidth_hz[s], pr_sr[s], pr_rg[s])
            else:
                rate[s] = 0.0
                rssi[s] = -140.0

        # -----------------------------------------------------------
        # NEW REWARD CALCULATION (Absolute Value)
        # -----------------------------------------------------------

        # 1. Rate Reward (Normalized)
        # Encourages finding high throughput positions (e.g. LoS)
        sum_rate = float(np.sum(rate))
        r_rate = sum_rate / self.rew_norm_rate_bps

        # 2. RSSI Reward (Normalized & Averaged)
        # Encourages coverage and link quality even if bandwidth is saturated
        # Map [-110, -40] to [0, 1]
        rssi_norm = np.clip((rssi - self.rew_rssi_min) / (self.rew_rssi_max - self.rew_rssi_min), 0.0, 1.0)
        r_rssi = float(np.mean(rssi_norm))

        # 3. Action Penalty
        # Penalizes large control inputs to reduce jitter/energy consumption
        # Using Mean Squared Error of the raw action [-1, 1]
        p_action = float(np.mean(raw_action ** 2))

        # Combined Reward
        reward = (self.w_rate * r_rate) + (self.w_rssi * r_rssi) - (self.w_penalty * p_action)

        terminated = False
        truncated = self.t >= self.episode_len

        obs = self._get_obs(rssi, rate)

        if self.test_mode:
            row: Dict[str, object] = {
                "step": int(self.t),
                "sum_rate_bps": float(sum_rate),
                "reward": float(reward),
                # New log fields for debugging reward components
                "rew_rate_comp": float(r_rate),
                "rew_rssi_comp": float(r_rssi),
                "rew_act_pen": float(p_action),
                "ground_lat": float(self.ground_pos[0]),
                "ground_lon": float(self.ground_pos[1]),
            }
            for i in range(self.n_relays):
                row[f"relay{i}_lat"] = float(self.relay_pos[i, 0])
                row[f"relay{i}_lon"] = float(self.relay_pos[i, 1])
                row[f"relay{i}_alt_m"] = float(self.relay_pos[i, 2])

            for s in range(self.n_survey):
                row[f"uav_id_{s}"] = int(self.uav_ids[s]) if s < len(self.uav_ids) else int(s)
                row[f"survey{s}_lat"] = float(self.survey_pos[s, 0])
                row[f"survey{s}_lon"] = float(self.survey_pos[s, 1])
                row[f"survey{s}_alt_m"] = float(self.survey_pos[s, 2])
                row[f"sel_relay_{s}"] = int(self.sel_relay[s])
                row[f"rssi_{s}_dbm"] = float(rssi[s])
                row[f"rate_{s}_bps"] = float(rate[s])
                row[f"bw_{s}_hz"] = float(self.link_bandwidth_hz[s])

            self.log_data.append(row)

        info = {
            "sum_rate_bps": sum_rate,
            "rate_bps": rate.copy(),
            "rssi_dbm": rssi.copy(),
            "sel_relay": self.sel_relay.copy(),
            "bw_hz": self.link_bandwidth_hz.copy(),
        }
        return obs, reward, terminated, truncated, info
