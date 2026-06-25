# -*- coding: utf-8 -*-
"""
chibi_PPO_matplot_shp_v4_B_I.py

Train/test PPO with the Shapefile-driven multi-survey-UAV environment.

Changes:
- Does NOT hardcode n_survey. It infers number of survey UAVs from the shapefile by passing n_survey=None.
- Plot includes the ground station point.
- Implements Beta Distribution for action space (BetaActorCriticPolicy).
- Implements Intrinsic Curiosity Module (ICM) for exploration.
- Adds a horizontal mean-fit line for total throughput.

Important:
- If you trained a model with n_survey=2, you MUST retrain when n_survey changes (obs dimension changes).
"""
'''
赤壁及周边
地面端在研究区内
'''
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import re
from typing import Callable

import gym
from gym import spaces  # <--- 修复 NameError 的关键导入
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecEnvWrapper
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import explained_variance  # <--- 修复 train 方法需要的导入

# from Env_matplot_relaymove_shp_v4_copy import RelayMoveEnvShp
from Env_matplot_relaymove_shp_v4_copy import RelayMoveEnvShp

'''
使用df，双半工路径
BETA
ICM
'''

is_training = False  # True: train; False: test

SHAPEFILE_PATH = r"E:\毕业论文\1planning\数据\ChibiUAVbase\chibi_UAVbase_boustrophedon_paths_energy_with_batch_and_uav_point.shp"

INIT_RELAY_POS = [
    # [30.52131148155365, 114.370137061499, 80.0],  # 研究区右下角
    # [30.527200552833573, 114.3556190247159, 100.0]#武大
    [29.788655458031535 , 113.9091413860207, 80.0],
    [29.794116735397676 , 113.91302969947397, 100.0]#赤壁
]

GROUND_POS = [29.789405942654724,  113.91183988045603, 10.0]#赤壁基地
# GROUND_POS = [30.516004194419658, 114.38530939375414, 0.0]

MODEL_DIR = "runs_ppo_relay_shp_relative_obs_BIC_chibi"
MODEL_PATH = os.path.join(MODEL_DIR, "ppo_relay_model.zip")
VECNORM_PATH = os.path.join(MODEL_DIR, "vecnormalize.pkl")
CSV_PATH = os.path.join(MODEL_DIR, "test_log.csv")


# ==========================================
# 1. 定义 Beta 分布 (用于替代默认的高斯分布)
# ==========================================
class BetaDistribution(Distribution):
    def __init__(self, action_dim: int):
        super().__init__()
        self.action_dim = action_dim

    def proba_distribution_net(self, latent_dim: int, log_std_init: float = 0.0) -> tuple:
        # self.mean_actions = nn.Linear(latent_dim, self.action_dim)
        # self.log_std = nn.Linear(latent_dim, self.action_dim)
        # return self.mean_actions, self.log_std
        self.mu_net = nn.Linear(latent_dim, self.action_dim)
        self.kappa_net = nn.Linear(latent_dim, self.action_dim)
        return self.mu_net, self.kappa_net

    # def proba_distribution(self, mean_actions: torch.Tensor, log_std: torch.Tensor) -> "BetaDistribution":
        # self.alpha = F.softplus(mean_actions) + 1.0
        # self.beta = F.softplus(log_std) + 1.0
        # self.distribution = Beta(self.alpha, self.beta)
        # return self
    def proba_distribution(self, mu_raw: torch.Tensor, kappa_raw: torch.Tensor) -> "BetaDistribution":
        mu = torch.tanh(mu_raw)  # [-1, 1]
        mu01 = torch.clamp((mu + 1.0) / 2.0, 1e-6, 1 - 1e-6)
        kappa = F.softplus(kappa_raw) + 2.0  # > 2 保证 alpha,beta>1 更稳定

        self.alpha = mu01 * kappa + 1.0
        self.beta = (1.0 - mu01) * kappa + 1.0
        self.distribution = Beta(self.alpha, self.beta)
        return self

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        beta_actions = (actions + 1) / 2
        beta_actions = torch.clamp(beta_actions, 1e-6, 1 - 1e-6)  # <-- 必加
        return self.distribution.log_prob(beta_actions).sum(dim=1) - self.action_dim * np.log(2.0)

    def sample(self) -> torch.Tensor:
        sample = self.distribution.rsample()
        return sample * 2.0 - 1.0

    def mode(self) -> torch.Tensor:
        a, b = self.alpha, self.beta
        # Beta 分布 mode： (a-1)/(a+b-2)  (仅当 a>1 且 b>1)
        m = (a - 1.0) / (a + b - 2.0 + 1e-8)

        # 边界情况（a<=1 或 b<=1）时，mode 在 0 或 1 附近
        m = torch.where((a <= 1) & (b > 1), torch.zeros_like(m), m)
        m = torch.where((a > 1) & (b <= 1), torch.ones_like(m), m)
        m = torch.clamp(m, 1e-6, 1 - 1e-6)

        return m * 2.0 - 1.0

    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=1)

    def actions_from_params(self, mean_actions: torch.Tensor, log_std: torch.Tensor,
                            deterministic: bool = False) -> torch.Tensor:
        self.proba_distribution(mean_actions, log_std)
        return self.mode() if deterministic else self.sample()

    def log_prob_from_params(self, mean_actions: torch.Tensor, log_std: torch.Tensor) -> tuple:
        actions = self.actions_from_params(mean_actions, log_std)
        log_prob = self.log_prob(actions)
        return actions, log_prob


# 定义使用 Beta 分布的策略网络
class BetaActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 覆盖 action_dist
        self.action_dist = BetaDistribution(self.action_space.shape[0])

        # === 关键修正: 删除父类创建的 log_std Parameter ===
        if hasattr(self, "log_std"):
            del self.log_std

        # 重新初始化 action_net 以匹配 Beta 分布所需的输出
        # self.action_net, self.log_std = self.action_dist.proba_distribution_net(
        #     latent_dim=self.mlp_extractor.latent_dim_pi)
        self.action_net, self.kappa_net = self.action_dist.proba_distribution_net(
        latent_dim=self.mlp_extractor.latent_dim_pi
        )
    # def _get_action_dist_from_latent(self, latent_pi: torch.Tensor) -> Distribution:
    #     mean_actions = self.action_net(latent_pi)
    #     log_std = self.log_std(latent_pi)
    #     return self.action_dist.proba_distribution(mean_actions, log_std)
    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor) -> Distribution:
        mu_raw = self.action_net(latent_pi)
        kappa_raw = self.kappa_net(latent_pi)
        return self.action_dist.proba_distribution(mu_raw, kappa_raw)


# ==========================================
# 2. 定义 ICM (内在好奇心模块)
# ==========================================
class ICMModule(nn.Module):
    def __init__(self, input_dim, action_dim, feature_dim=64):
        super(ICMModule, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim)
        )

        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim)
        )

        self.inverse_model = nn.Sequential(
            nn.Linear(feature_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, state, next_state, action):
        phi_state = self.encoder(state)
        phi_next_state = self.encoder(next_state)
        pred_phi_next_state = self.forward_model(torch.cat([phi_state, action], dim=1))
        pred_action = self.inverse_model(torch.cat([phi_state, phi_next_state], dim=1))
        return phi_next_state, pred_phi_next_state, pred_action


# ==========================================
# 3. 自定义 BetaICMPPO 算法
# ==========================================
class BetaICMPPO(PPO):
    def __init__(self, policy, env, icm_lr=1e-3, icm_beta=0.2, icm_eta=0.1, *args, **kwargs):
        # 强制使用自定义的 Beta 策略
        super().__init__(BetaActorCriticPolicy, env, *args, **kwargs)

        self.icm_lr = icm_lr
        self.icm_beta = icm_beta
        self.icm_eta = icm_eta

        # 初始化 ICM 模块
        obs_dim = self.observation_space.shape[0]
        action_dim = self.action_space.shape[0]

        self.icm = ICMModule(obs_dim, action_dim).to(self.device)
        self.icm_optimizer = torch.optim.Adam(self.icm.parameters(), lr=self.icm_lr)

    def train(self):
        """
        重写 train 方法以避免 SB3 默认日志记录 log_std 时报错。
        """
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        self._update_learning_rate(self.icm_optimizer)

        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses = []
        value_losses = []
        clip_fractions = []

        continue_training = True

        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):  # <--- 之前报错的地方
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()

                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = torch.exp(log_prob - rollout_data.old_log_prob)

                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                pg_losses.append(policy_loss.item())
                clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)

        # <--- 修复日志报错: 只在 log_std 是 Tensor 时记录
        if hasattr(self.policy, "log_std"):
            if isinstance(self.policy.log_std, torch.Tensor):
                self.logger.record("train/std", torch.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        """
        重写 collect_rollouts 以注入内在奖励 (Intrinsic Reward) 并在线训练 ICM
        """
        assert self._last_obs is not None, "No previous observation was provided"
        self.policy.set_training_mode(False)
        self.icm.eval()

        n_steps = 0
        rollout_buffer.reset()
        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            with torch.no_grad():
                # 转 Tensor
                obs_tensor, _ = self.policy.obs_to_tensor(self._last_obs)
                # 采样动作
                actions, values, log_probs = self.policy(obs_tensor)

            actions_np = actions.cpu().numpy()

            # 环境步进
            new_obs, rewards, dones, infos = env.step(actions_np)
            self.num_timesteps += env.num_envs

            # === 计算内在奖励 ===
            with torch.no_grad():
                next_obs_tensor, _ = self.policy.obs_to_tensor(new_obs)
                # ICM Forward
                phi_next, pred_phi_next, _ = self.icm(obs_tensor, next_obs_tensor, actions)
                intrinsic_reward = self.icm_eta * F.mse_loss(pred_phi_next, phi_next, reduction='none').mean(dim=1)
                intrinsic_reward = intrinsic_reward.cpu().numpy()

            # 叠加奖励
            total_rewards = rewards + intrinsic_reward

            # === ICM 在线更新 ===
            self.icm.train()
            phi_next_train, pred_phi_next_train, pred_action_train = self.icm(obs_tensor, next_obs_tensor, actions)
            forward_loss = F.mse_loss(pred_phi_next_train, phi_next_train)
            inverse_loss = F.mse_loss(pred_action_train, actions)
            icm_loss = (1 - self.icm_beta) * forward_loss + self.icm_beta * inverse_loss

            self.icm_optimizer.zero_grad()
            icm_loss.backward()
            self.icm_optimizer.step()
            self.icm.eval()

            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, list):
                actions_buffer = actions_np.reshape(-1, 1)
            else:
                actions_buffer = actions_np

            rollout_buffer.add(self._last_obs, actions_buffer, total_rewards, self._last_episode_starts, values,
                               log_probs)
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with torch.no_grad():
            obs_tensor, _ = self.policy.obs_to_tensor(self._last_obs)
            values = self.policy.predict_values(obs_tensor)

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        callback.on_rollout_end()
        return True


def make_env(episode_len: int, seed: int = 42, fixed_start: bool = False, test_mode: bool = False) -> Callable[
    [], RelayMoveEnvShp]:
    def _init() -> RelayMoveEnvShp:
        start_pos = INIT_RELAY_POS if fixed_start else None

        env = RelayMoveEnvShp(
            shapefile_path=SHAPEFILE_PATH,
            n_relays=2,
            n_survey=None,
            batch_list=None,
            total_bandwidth_hz=20e6,
            demand_bps=12e6,
            max_speed_m_per_step=20.0,
            max_v_speed_m_per_step=4.0,
            min_alt_m=50.0,
            max_alt_m=300.0,
            episode_len=episode_len,
            seed=seed,
            init_relay_pos=start_pos,
            ground_pos=GROUND_POS,
            use_meas_fit=True,
            air_air_csv=r"E:\毕业论文\2deployment\data\air_air_data.csv",
            air_ground_csv=r"E:\毕业论文\2deployment\data\air_ground_data.csv",
        )
        env.test_mode = test_mode
        return env

    return _init


def _infer_indices(cols, prefix: str, suffix: str):
    pat = re.compile(rf"^{re.escape(prefix)}(\d+){re.escape(suffix)}$")
    idx = set()
    for c in cols:
        m = pat.match(c)
        if m:
            idx.add(int(m.group(1)))
    return sorted(idx)


def train():
    os.makedirs(MODEL_DIR, exist_ok=True)
    env = DummyVecEnv([make_env(episode_len=300, seed=42, fixed_start=False)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))

    model = BetaICMPPO(
        policy=BetaActorCriticPolicy,
        env=env,
        verbose=1,
        learning_rate=3e-4,
        policy_kwargs=policy_kwargs,
        icm_lr=1e-3,
        icm_beta=0.2,
        icm_eta=0.5,
        ent_coef=0.01
    )

    print("Start training with Beta-ICM-PPO...")
    model.learn(total_timesteps=1000000)  # 根据需要改为 1000000

    model.save(MODEL_PATH)
    env.save(VECNORM_PATH)
    env.close()

    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved VecNormalize stats to: {VECNORM_PATH}")


def test():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(VECNORM_PATH)):
        raise FileNotFoundError("Model or VecNormalize stats not found. Train first or set correct paths.")

    os.makedirs(MODEL_DIR, exist_ok=True)

    eval_env = DummyVecEnv([make_env(episode_len=1300, seed=123, fixed_start=True, test_mode=True)])
    eval_env = VecNormalize.load(VECNORM_PATH, eval_env)
    eval_env.training = False
    eval_env.norm_reward = False

    custom_objects = {
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
    }
    # 使用自定义类加载
    model = BetaICMPPO.load(MODEL_PATH, env=eval_env, custom_objects=custom_objects)

    eval_env.set_attr("test_mode", True)

    obs = eval_env.reset()
    for _ in range(1210):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)

    log_data = eval_env.get_attr("log_data")[0]
    if not log_data:
        raise RuntimeError("log_data is empty. Check test_mode and env logging.")

    df = pd.DataFrame(log_data)
    df.to_csv(CSV_PATH, index=False)
    print(f"Saved test log CSV to: {CSV_PATH}")

    relay_ids = _infer_indices(df.columns, "relay", "_lat")
    survey_ids = _infer_indices(df.columns, "survey", "_lat")

    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    for i in relay_ids:
        plt.plot(df[f"relay{i}_lon"], df[f"relay{i}_lat"], label=f"Relay {i}")
        plt.scatter(df[f"relay{i}_lon"].iloc[0], df[f"relay{i}_lat"].iloc[0], marker='o', s=50, c='g')
        plt.scatter(df[f"relay{i}_lon"].iloc[-1], df[f"relay{i}_lat"].iloc[-1], marker='x', s=50, c='r')

    for s in survey_ids:
        label = f"Survey {s}"
        uav_col = f"uav_id_{s}"
        if uav_col in df.columns:
            label = f"Survey {s} (uav={int(df[uav_col].iloc[0])})"
        plt.plot(df[f"survey{s}_lon"], df[f"survey{s}_lat"], linestyle="--", alpha=0.5, label=label)

    plt.scatter([GROUND_POS[1]], [GROUND_POS[0]], marker="*", s=200, c='gold', edgecolors='k', label="Ground Station")

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("2D Trajectories (Top-Down View)")
    plt.legend(loc='upper right')

    plt.grid(True)


    plt.figure(figsize=(10, 4))
    for i in relay_ids:
        col_alt = f"relay{i}_alt_m"
        if col_alt in df.columns:
            plt.plot(df["step"], df[col_alt], label=f"Relay {i} Altitude")
    plt.xlabel("Step")
    plt.ylabel("Altitude (m)")
    plt.title("Relay UAV Altitude Change")
    plt.legend()
    plt.grid(True)

    plt.figure(figsize=(10, 4))
    for s in survey_ids:
        col = f"rate_{s}_bps"
        if col in df.columns:
            plt.plot(df["step"], df[col] / 1e6, label=f"Rate Survey {s}")
    if "sum_rate_bps" in df.columns:
        total_rate_mbps = df["sum_rate_bps"].to_numpy(dtype=float) / 1e6
        plt.plot(df["step"], total_rate_mbps, 'r-', linewidth=2, label="Sum Rate")
        finite_rates = total_rate_mbps[np.isfinite(total_rate_mbps)]
        if finite_rates.size > 0:
            mean_rate_mbps = float(np.mean(finite_rates))
            plt.axhline(
                y=mean_rate_mbps,
                color='r',
                linestyle='--',
                linewidth=2.0,
                alpha=0.85,
                label=f"Sum Rate Mean Fit ({mean_rate_mbps:.2f} Mbps)",
            )
    plt.xlabel("Step")
    plt.ylabel("Rate (Mbps)")
    plt.title("Network Throughput")
    plt.legend()
    plt.grid(True)

    plt.figure(figsize=(10, 4))
    for s in survey_ids:
        col = f"rssi_{s}_dbm"
        if col in df.columns:
            plt.plot(df["step"], df[col], label=f"RSSI Survey {s}")
    plt.xlabel("Step")
    plt.ylabel("RSSI (dBm)")
    plt.title("Link RSSI")
    plt.legend()
    plt.grid(True)

    plt.show(block=True)
    eval_env.close()


if __name__ == "__main__":
    if is_training:
        train()
    else:
        test()
