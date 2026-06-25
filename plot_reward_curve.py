# plot_reward_curve.py
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_reward_curve(csv_path):
    df = pd.read_csv(csv_path)

    x = df["episode"].values.astype(float)
    y = df["episode_reward"].values.astype(float)

    plt.figure(figsize=(10, 4))

    # 原始 reward 曲线：浅蓝色，透明度较低，作为背景波动
    plt.plot(
        x,
        y,
        linewidth=1.0,
        alpha=0.55,
        color="#8EC7E8",
        label="Episode Reward"
    )

    # 平滑趋势线：深蓝色，突出训练收敛趋势
    window = 150
    smooth = (
        pd.Series(y)
        .rolling(window=window, center=True, min_periods=1)
        .median()
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .values
    )

    plt.plot(
        x,
        smooth,
        color="#1F5A85",
        linewidth=2.6,
        label="Reward Trend"
    )

    plt.xlabel("Training iteration")
    plt.ylabel("Reward")
    plt.title("Training Reward Curve")
    plt.grid(True, alpha=0.28)
    plt.legend(frameon=True)
    plt.tight_layout()

    out_path = os.path.splitext(csv_path)[0] + "_plot.jpg"
    plt.savefig(out_path, dpi=500)
    plt.show()

    print("Saved figure to:", out_path)


if __name__ == "__main__":
    csv_path = r"E:\毕业论文\2deployment\code\run_ppo_curvey\v4\v4_training_curve.csv"
    plot_reward_curve(csv_path)