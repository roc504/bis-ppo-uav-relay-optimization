
"""
DR_SA_g.py
Resource scheduling via Simulated Annealing (SA) for bandwidth allocation.

Key fixes included (per requirements):
- Metropolis acceptance probability uses exp(-(Δcost)/T) with the REQUIRED negative sign.
- Bandwidth lower bound is strictly > 0 (default 1e6 Hz) using np.clip to avoid divide-by-zero / dead links.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


@dataclass
class SAConfig:
    # Annealing schedule
    t_init: float = 5.0
    t_min: float = 1e-3
    alpha: float = 0.97
    n_iter: int = 250
    # -------------------------
    # 4.3.1 Resource-flow neighborhood
    # -------------------------
    p_global: float = 0.25  # 选择全局搜索的概率（1-p_global 为局部搜索）
    local_frac: float = 0.05  # 局部扰动幅度：Δb ~ U(0, local_frac * B_i)
    global_frac: float = 0.40  # 全局扰动幅度：Δb ~ U(0, global_frac * B_i)

    # -------------------------
    # 4.3.3 Adaptive cooling schedule
    # -------------------------
    lambda_cool: float = 0.6  # λ：调节强度（你论文里的 λ）
    alpha_min: float = 0.90  # α 下限（防止过慢/回温过猛）
    alpha_max: float = 1.05  # α 上限（>1 允许“回温”）

    # Proposal
    step_scale: float = 0.08  # relative perturbation (fraction of current BW)
    bw_min: float = 1e6       # 1 MHz lower bound (MUST NOT be 0)

    # --- Guard band (Sigmoid) params ---
    rho0: float = 0.05  # 基础保护比下限
    rho_inc: float = 0.10  # 保护比增幅上限
    k_sigmoid: float = 3e-7  # Sigmoid斜率（单位约 1/Hz）
    bw_knee: float = 10e6  # 拐点阈值 B0（Hz）
    # 可选：净带宽下限（推荐与 bw_min 一样设 1e6），确保 Beff 不会太小
    bw_eff_min: float = 1e6

    # Objective weights
    w_mismatch: float = 3.0
    w_unmet: float = 5.0
    w_neg_throughput: float = 0.5  # encourages higher sum throughput within constraints

    # Numerical safety
    eps: float = 1e-12

    def __post_init__(self):
        rho_max = self.rho0 + self.rho_inc
        # 防止 rho_max >= 1 导致除零或负净带宽
        rho_max = float(np.clip(rho_max, 0.0, 0.95))
        min_phys = self.bw_eff_min / max(1.0 - rho_max, 1e-6)
        # bw_min 仍保持“严格>0”的要求，但提升到能保证净带宽下限
        self.bw_min = float(max(self.bw_min, min_phys))

def _guard_ratio_sigmoid(bw_hz: np.ndarray, cfg: SAConfig) -> np.ndarray:
    """
    rho(B) = rho0 + rho_inc / (1 + exp(-k*(B - B0)))
    """
    x = -cfg.k_sigmoid * (bw_hz - cfg.bw_knee)
    # 数值稳定：避免 exp 溢出
    x = np.clip(x, -60.0, 60.0)
    return cfg.rho0 + cfg.rho_inc / (1.0 + np.exp(x))


def _effective_bw(bw_hz: np.ndarray, cfg: SAConfig) -> np.ndarray:
    """
    Beff = B - G = B*(1 - rho(B))
    """
    rho = _guard_ratio_sigmoid(bw_hz, cfg)
    beff = bw_hz * (1.0 - rho)
    return np.maximum(beff, cfg.eps)

def _rates_from_bw(
    bandwidth_hz: np.ndarray,
    rx_power_w: np.ndarray,
    noise_density_w_per_hz: float,
    cfg: SAConfig,
) -> np.ndarray:
    """
    Shannon-like rate:
        rate_i = B_i * log2(1 + SNR_i)
        SNR_i = Pr_i / (N0 * B_i)
    """
    # 物理分配带宽 B（仍用于总带宽约束）
    bw = np.maximum(bandwidth_hz, cfg.eps)
    # 净带宽 Beff = B - G(B)
    beff = _effective_bw(bw, cfg)
    # Shannon rate with noise accumulated over Beff
    noise_power = noise_density_w_per_hz * bw
    snr = rx_power_w / np.maximum(noise_power, cfg.eps)
    return beff * np.log2(1.0 + snr)


def _objective(
    bandwidth_hz: np.ndarray,
    total_bandwidth_hz: float,
    demand_bps: np.ndarray,
    rx_power_w: np.ndarray,
    noise_density_w_per_hz: float,
    cfg: SAConfig,
) -> Tuple[float, np.ndarray]:
    """
    Minimize:
      - mismatch between sum(B) and total
      - unmet demand penalty
      - negative throughput (encourage larger throughput)
    """
    # Enforce bounds (lower bound must not be 0)
    bw = np.clip(bandwidth_hz, cfg.bw_min, total_bandwidth_hz)
    rates = _rates_from_bw(bw, rx_power_w, noise_density_w_per_hz, cfg)
    mismatch = abs(float(np.sum(bw)) - float(total_bandwidth_hz)) / max(total_bandwidth_hz, cfg.eps)

    # unmet demand ratio, robust to demand=0
    demand_safe = np.maximum(demand_bps, cfg.eps)
    unmet = np.maximum(0.0, demand_bps - rates) / demand_safe
    unmet_pen = float(np.mean(unmet))

    # encourage throughput; normalize by total demand to keep scaling sane
    sum_rate = float(np.sum(rates))
    sum_demand = float(np.sum(demand_safe))
    neg_throughput = -sum_rate / sum_demand

    cost = (
        cfg.w_mismatch * mismatch
        + cfg.w_unmet * unmet_pen
        + cfg.w_neg_throughput * neg_throughput
    )
    return float(cost), rates


def _project_to_total(
    bw: np.ndarray,
    total_bw: float,
    bw_min: float,
) -> np.ndarray:
    """
    Project vector to (approximately) satisfy sum(bw)=total_bw while respecting bw_min.
    This is a soft projection: it rescales and then re-adjusts residual.
    """
    bw = bw.astype(np.float64, copy=True)

    # First clip to lower bound
    bw = np.maximum(bw, bw_min)

    s = float(np.sum(bw))
    if s <= 0:
        # fallback: uniform
        bw[:] = total_bw / bw.size
        bw = np.maximum(bw, bw_min)
        return bw

    # Rescale to target
    bw *= (total_bw / s)

    # Enforce lower bound again (may break sum)
    bw = np.maximum(bw, bw_min)

    # Distribute residual to keep sum close to total
    residual = total_bw - float(np.sum(bw))
    if abs(residual) < 1e-6:
        return bw

    # Allocate residual proportionally to "free" bandwidth above bw_min
    free = bw - bw_min
    free_sum = float(np.sum(free))
    if free_sum <= 0:
        # cannot redistribute; add evenly
        bw += residual / bw.size
        bw = np.maximum(bw, bw_min)
        return bw

    bw += residual * (free / free_sum)
    bw = np.maximum(bw, bw_min)
    return bw


def simulated_annealing(
    total_bandwidth_hz: float,
    throughput_demand_bps: Sequence[float],
    rx_power_w: Sequence[float],
    noise_density_w_per_hz: float = 1e-20,
    init_solution_hz: Optional[Sequence[float]] = None,
    config: Optional[SAConfig] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    SA optimizer for bandwidth allocation.

    Inputs:
      - total_bandwidth_hz: scalar total bandwidth
      - throughput_demand_bps: array of per-link demand
      - rx_power_w: per-link received power (linear, watts)
      - noise_density_w_per_hz: noise PSD (watts/Hz)

    Output:
      - best bandwidth allocation (Hz), shape (n_links,)
    """
    cfg = config or SAConfig()
    rng = np.random.default_rng(seed)

    demand = np.asarray(throughput_demand_bps, dtype=np.float64)
    pr = np.asarray(rx_power_w, dtype=np.float64)

    if demand.ndim != 1 or pr.ndim != 1 or demand.size != pr.size:
        raise ValueError("throughput_demand_bps and rx_power_w must be 1D arrays of the same length.")

    n = demand.size
    if n == 0:
        return np.array([], dtype=np.float64)
    if n == 1:
        return np.array([float(total_bandwidth_hz)], dtype=np.float64)

    # Initialize solution
    if init_solution_hz is None:
        cur = np.full(n, total_bandwidth_hz / n, dtype=np.float64)
    else:
        cur = np.asarray(init_solution_hz, dtype=np.float64).copy()
        if cur.shape != (n,):
            raise ValueError("init_solution_hz must have shape (n_links,).")

    # Enforce bounds + approximate sum constraint
    cur = np.clip(cur, cfg.bw_min, total_bandwidth_hz)
    cur = _project_to_total(cur, total_bandwidth_hz, cfg.bw_min)

    cur_cost, _ = _objective(cur, total_bandwidth_hz, demand, pr, noise_density_w_per_hz, cfg)
    best = cur.copy()
    best_cost = float(cur_cost)

    T = float(cfg.t_init)
    alpha = float(cfg.alpha)  # 初始冷却速率（后续自适应更新）
    for _ in range(cfg.n_iter):
        '''
        # Propose: multiplicative jitter + small gaussian noise
        jitter = 1.0 + rng.normal(0.0, cfg.step_scale, size=n)
        prop = cur * jitter + rng.normal(0.0, cfg.step_scale * (total_bandwidth_hz / max(n, 1)), size=n)

        # Enforce strict lower bound > 0, as required
        prop = np.clip(prop, cfg.bw_min, total_bandwidth_hz)
        prop = _project_to_total(prop, total_bandwidth_hz, cfg.bw_min)
        '''
        prop = cur.copy()

        # 随机选两条不同链路 i, j
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n - 1))
        if j >= i:
            j += 1

        # 局部/全局两模式：决定 Δb 的动态范围
        if rng.random() < cfg.p_global:
            frac = cfg.global_frac  # 全局：大幅转移
        else:
            frac = cfg.local_frac  # 局部：微小转移

        # Δb ~ U(0, frac * b_i)（动态范围，随当前 b_i 变化）
        delta_b = float(rng.uniform(0.0, frac * prop[i]))

        # 做资源转移
        prop[i] -= delta_b
        prop[j] += delta_b

        # 合法性修正：确保不低于 bw_min，并保持 sum 恒定
        # 如果 i 触底，则把超出的量退回给 j
        if prop[i] < cfg.bw_min:
            deficit = cfg.bw_min - prop[i]
            prop[i] = cfg.bw_min
            prop[j] -= deficit

        # 如果 j 触底（可能因为上一步回退），则反向回退给 i
        if prop[j] < cfg.bw_min:
            deficit = cfg.bw_min - prop[j]
            prop[j] = cfg.bw_min
            prop[i] -= deficit

        # 最后再做一次安全兜底：如果仍然不合法，则放弃该提议，改用均分（或直接 continue）
        if prop[i] < cfg.bw_min or prop[j] < cfg.bw_min:
            prop = cur.copy()

        new_cost, _ = _objective(prop, total_bandwidth_hz, demand, pr, noise_density_w_per_hz, cfg)
        old_cost = float(cur_cost)
        delta = float(new_cost - cur_cost)

        # Metropolis criterion with REQUIRED negative sign: exp(-(Δ)/T)
        if delta <= 0:
            accept = True
        else:
            # Numerical safety: avoid overflow in exp
            prob = float(np.exp(-delta / max(T, cfg.eps)))
            prob = min(1.0, max(0.0, prob))
            accept = rng.random() < prob

        if accept:
            cur = prop
            cur_cost = float(new_cost)
            if cur_cost < best_cost:
                best = cur.copy()
                best_cost = float(cur_cost)

        '''# Cool down
        T = max(cfg.t_min, T * cfg.alpha)'''
        # -------------------------
        # 4.3.3 Adaptive cooling based on relative cost change
        # ΔC = (C_new - C_old) / (C_old + eps)
        # alpha_new = alpha_old * (1 + lambda * sgn(ΔC) * |ΔC|)
        # -------------------------
        deltaC = (float(new_cost) - old_cost) / (abs(old_cost) + cfg.eps)
        alpha = alpha * (1.0 + cfg.lambda_cool * np.sign(deltaC) * abs(deltaC))
        alpha = float(np.clip(alpha, cfg.alpha_min, cfg.alpha_max))
        T = max(cfg.t_min, T * alpha)

    # Final enforce
    best = np.clip(best, cfg.bw_min, total_bandwidth_hz)
    best = _project_to_total(best, total_bandwidth_hz, cfg.bw_min)
    return best
