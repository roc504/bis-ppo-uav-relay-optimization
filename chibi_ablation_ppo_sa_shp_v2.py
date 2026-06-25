# -*- coding: utf-8 -*-
"""
Ablation_Experiment.py
消融实验对比：PSO vs PPO，Equal vs SA
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 必须放在最前

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from Env_matplot_relaymove_shp_v4 import RelayMoveEnvShp
from PSO_Agent import PSOAgent

# === 配置参数 ===
SHAPEFILE_PATH = r"E:\毕业论文\1planning\数据\boustrophedon_paths_with_batch_uav.shp"
MODEL_DIR = "runs_ppo_relay_shp_relative_obs"
MODEL_PATH = os.path.join(MODEL_DIR, "ppo_relay_model.zip")
VECNORM_PATH = os.path.join(MODEL_DIR, "vecnormalize.pkl")
OUTPUT_DIR = "ablation_results"
EPISODE_LEN = 300

INIT_RELAY_POS = [
    [30.52131148155365, 114.370137061499, 80.0],
    [30.527200552833573, 114.3556190247159, 100.0]
]
GROUND_POS = [30.516004194419658, 114.38530939375414, 0.0]


def make_env_ablation(bw_strategy="sa"):
    # 技巧：把 episode_len 设为 EPISODE_LEN + 50
    # 这样在运行 EPISODE_LEN 步时，环境不会判定为结束，
    # 从而避免 DummyVecEnv 触发自动 reset 清空 log_data
    safe_len = EPISODE_LEN + 50
    env = RelayMoveEnvShp(
        shapefile_path=SHAPEFILE_PATH,
        n_relays=2,
        n_survey=None,
        init_relay_pos=INIT_RELAY_POS,
        ground_pos=GROUND_POS,
        episode_len=safe_len,
        bw_alloc_strategy=bw_strategy,  # 'sa' or 'equal'
        use_meas_fit=True,
        air_air_csv=r"E:\毕业论文\2deployment\data\air_air_data.csv",
        air_ground_csv=r"E:\毕业论文\2deployment\data\air_ground_data.csv",
    )
    return env


def run_experiment(variant_name, use_ppo, bw_strategy, use_pso=False):
    print(f"\n=== Running Variant: {variant_name} ===")

    # 1. 基础环境 (PSO直接用这个)
    raw_env = make_env_ablation(bw_strategy=bw_strategy)
    raw_env.test_mode = True

    # 定义一个变量来持有底层环境的引用 (用于 PPO)
    ppo_base_env = None

    # 2. 初始化 Agent
    if use_ppo:
        dummy_env = DummyVecEnv([lambda: make_env_ablation(bw_strategy=bw_strategy)])

        # 获取底层环境引用并开启 test_mode
        ppo_base_env = dummy_env.envs[0]
        ppo_base_env.test_mode = True

        norm_env = VecNormalize.load(VECNORM_PATH, dummy_env)
        norm_env.training = False
        norm_env.norm_reward = False

        agent = PPO.load(MODEL_PATH)
        obs_norm = norm_env.reset()

        # Reset 会清空 log_data，所以为了保险，再次确认 test_mode
        ppo_base_env.test_mode = True

    elif use_pso:
        print("Initializing PSO Agent...")
        agent = PSOAgent(raw_env, n_particles=30, n_iterations=10)
        raw_env.reset()
    else:
        raise ValueError("Must use either PPO or PSO")

    # 3. 运行循环
    for t in range(EPISODE_LEN):
        if use_ppo:
            action, _ = agent.predict(obs_norm, deterministic=True)
            # Step 返回的是 normalized obs，但底层环境已经记录了 raw log
            obs_norm, _, _, _ = norm_env.step(action)
        elif use_pso:
            action, _ = agent.predict(None)
            raw_env.step(action)
            if t % 50 == 0: print(f"PSO Step {t}")

    # 4. 提取日志
    if use_ppo:
        # --- 【修复核心：直接从底层实例读取数据】 ---
        log_data = ppo_base_env.log_data
        norm_env.close()
    else:
        log_data = raw_env.log_data
        raw_env.close()

    if not log_data:
        print(f"Warning: Log data is empty for {variant_name}!")
        return pd.DataFrame()

    df = pd.DataFrame(log_data)
    df.to_csv(os.path.join(OUTPUT_DIR, f"{variant_name}.csv"), index=False)
    print(f"Saved: {variant_name}.csv with {len(df)} rows.")
    return df


# --- 【修改绘图函数：添加坐标轴标签】 ---
def plot_results(all_dfs):
    plt.figure(figsize=(10, 6), dpi=100)  # 增加dpi让图更清晰

    colors = {
        "PSO-Equal": "green", "PSO-SA": "blue",
        "PPO-Equal": "orange", "PPO-SA": "red"
    }
    styles = {
        "PSO-Equal": "--", "PSO-SA": "--",
        "PPO-Equal": "-", "PPO-SA": "-"
    }

    for name, df in all_dfs.items():
        if name in colors and not df.empty:
            # 简单平滑处理，让曲线更好看 (Window=5)
            # 如果想要原始数据，把 .rolling(...).mean() 去掉即可
            steps = df["step"]
            rate_mbps = df["sum_rate_bps"] / 1e6

            plt.plot(steps, rate_mbps,
                     label=name,
                     color=colors[name],
                     linestyle=styles[name],
                     linewidth=2)  # 加粗线条

    # --- 添加坐标轴标签 ---
    plt.xlabel("Time Step", fontsize=12)  # 横坐标
    plt.ylabel("Sum Rate (Mbps)", fontsize=12)  # 纵坐标
    plt.title("Ablation Study Comparison", fontsize=14)  # 标题

    plt.legend(fontsize=10, loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)  # 网格线设为虚线
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 实验配置字典 ---
    experiments = {
        # 1. 传统方法基准：PSO轨迹 + 均分带宽
        "PSO-Equal": {"use_ppo": False, "use_pso": True, "bw": "equal"},

        # 2. 混合基准：PSO轨迹 + SA资源调度
        "PSO-SA": {"use_ppo": False, "use_pso": True, "bw": "sa"},

        # 3. 消融：PPO轨迹 + 均分带宽
        "PPO-Equal": {"use_ppo": True, "use_pso": False, "bw": "equal"},

        # 4. 我们的方法：PPO轨迹 + SA资源调度
        "PPO-SA": {"use_ppo": True, "use_pso": False, "bw": "sa"},
    }

    results = {}
    for name, cfg in experiments.items():
        try:
            df = run_experiment(name, cfg["use_ppo"], cfg["bw"], use_pso=cfg.get("use_pso", False))
            results[name] = df
        except Exception as e:
            print(f"Error {name}: {e}")
            import traceback

            traceback.print_exc()

    plot_results(results)