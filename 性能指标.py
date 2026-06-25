import pandas as pd
import numpy as np
import os


def calculate_comprehensive_metrics(file_path, algorithm_name):
    """
    计算单个算法日志文件的综合性能指标
    """
    if not os.path.exists(file_path):
        print(f"警告: 找不到文件 {file_path}")
        return None

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"错误: 无法读取 {file_path}. {e}")
        return None

    # --- 1. 平均系统吞吐量 (Average System Throughput) ---
    # 定义: 任务周期内 sum_rate_bps 的均值
    # 单位转换: bps -> Mbps
    if 'sum_rate_bps' in df.columns:
        avg_throughput_mbps = df['sum_rate_bps'].mean() / 1e6
    else:
        avg_throughput_mbps = 0.0

    # --- 2. 覆盖中断概率 (Coverage Outage Probability) ---
    # 定义: RSSI < -110 dBm 或 Rate < 12 Mbps 的时间比例
    # 阈值
    RSSI_THRESHOLD = -110.0
    RATE_THRESHOLD = 12e6  # 12 Mbps

    total_samples = 0
    outage_samples = 0

    # 自动检测有多少个测绘用户 (survey0, survey1, ...)
    # 假设列名格式为 rssi_{id}_dbm 和 rate_{id}_bps
    user_ids = []
    for col in df.columns:
        if col.startswith('rssi_') and col.endswith('_dbm'):
            try:
                uid = int(col.split('_')[1])
                user_ids.append(uid)
            except:
                pass
    user_ids = sorted(list(set(user_ids)))

    if len(user_ids) > 0:
        for uid in user_ids:
            rssi_col = f'rssi_{uid}_dbm'
            rate_col = f'rate_{uid}_bps'

            if rssi_col in df.columns and rate_col in df.columns:
                # 当前用户的样本数
                total_samples += len(df)

                # 统计中断次数 (逻辑或: 信号弱 OR 速率低)
                # 使用向量化操作加速计算
                current_outages = ((df[rssi_col] < RSSI_THRESHOLD) |
                                   (df[rate_col] < RATE_THRESHOLD)).sum()
                outage_samples += current_outages

    if total_samples > 0:
        outage_prob_percent = (outage_samples / total_samples) * 100.0
    else:
        outage_prob_percent = 0.0

    # --- 3. 轨迹平滑度代价 (Trajectory Smoothness Cost) ---
    # 定义: 累积的动作惩罚 (rew_act_pen)，反映了加速度/控制输入的 L2 范数总和
    if 'rew_act_pen' in df.columns:
        smoothness_cost = df['rew_act_pen'].sum()
    else:
        # 如果没有该列 (如静态算法)，代价为 0
        smoothness_cost = 0.0

    return {
        'Algorithm': algorithm_name,
        'Avg Throughput (Mbps)': round(avg_throughput_mbps, 2),
        'Outage Prob (%)': round(outage_prob_percent, 2),
        'Smoothness Cost': round(smoothness_cost, 2)
    }


# ==========================================
# 主程序：处理所有文件并生成表格
# ==========================================

# 文件名映射到算法名称
files_map = [
    ('runs_ppo_relay_shp_relative_obs/test_log.csv', 'PPO-SA-Hybrid (Ours)'),
    ('runs_baselines_aligned/SGC_test_log.csv', 'SGC (Static)'),
    ('runs_baselines_aligned/GST_test_log.csv', 'GST (Greedy)'),
    ('runs_baselines_aligned/PSO_test_log.csv', 'PSO (Swarm)')
]

results = []

print("正在计算指标...\n")

for csv_file, algo_name in files_map:
    metrics = calculate_comprehensive_metrics(csv_file, algo_name)
    if metrics:
        results.append(metrics)

# 创建 DataFrame 展示结果
if results:
    results_df = pd.DataFrame(results)
    # 调整列顺序
    cols = ['Algorithm', 'Avg Throughput (Mbps)', 'Outage Prob (%)', 'Smoothness Cost']
    results_df = results_df[cols]

    print("=== 表 5-2: 核心性能指标对比统计 ===")
    print(results_df.to_string(index=False))

    # 同时也打印出 Markdown 格式，方便您复制到论文或文档中
    print("\n[Markdown 格式]:")
    print(results_df.to_markdown(index=False))
else:
    print("未生成任何结果，请检查文件路径。")