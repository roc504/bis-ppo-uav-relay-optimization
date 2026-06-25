# UAV Remote Sensing Backhaul Optimization

This repository contains the core code for a UAV relay deployment and bandwidth allocation method for remote sensing data backhaul in infrastructure-limited emergency mapping scenarios.

## Files

| File | Description |
|---|---|
| `Env_matplot_relaymove_shp_v4.py` | Main simulation environment. It models survey UAVs, relay UAVs, the ground station, two-hop communication links, RSSI, throughput, relay movement, reward calculation, and logging. |
| `DR_SA_g.py` | Simulated annealing based bandwidth allocation module. It optimizes bandwidth allocation under total bandwidth and protection-bandwidth constraints. |
| `chibi_PPO_matplot_shp_v4_B_I.py` | Main BI-PPO training and testing script. It implements the Beta action distribution, intrinsic curiosity module, and PPO-based relay UAV deployment policy. |
| `chibi_baselines_relay_shp_V3.py` | Baseline comparison script. It includes SGC, GST, and PSO relay deployment methods and generates comparison results. |
| `chibi_ablation_ppo_sa_shp_v2.py` | Ablation experiment script. It compares PPO/PSO deployment with equal bandwidth allocation and SA-based bandwidth allocation. |
| `multi_seed_chibi_eval.py` | Multi-seed evaluation script. It runs repeated experiments under different random seeds and summarizes average performance and stability. |
| `measure_inference_time.py` | Online decision time evaluation script. It measures per-step inference or decision time for SGC, GST, PSO, and BI-PPO. |
| `性能指标.py` | Performance metric calculation script. It computes metrics such as average throughput, weak coverage risk, and trajectory smoothness cost. |
| `G_generate_air_user_comm_field_snapshots_with_basemap.py` | Visualization script for communication fields. It generates RSSI, throughput, and best-relay selection maps over a basemap. |
| `plot_reward_curve.py` | Training curve plotting script. It visualizes the reward curve during model training. |

## Notes

The code is intended to support experiments on joint relay UAV deployment and bandwidth allocation for UAV remote sensing data backhaul. Some scripts depend on experiment logs or trained model files generated during training and evaluation.
