# -*- coding: utf-8 -*-
"""
G_generate_air_user_comm_field_snapshots_with_survey.py

生成“空中测绘用户”通信场时序快照热力图，并叠加当前测绘无人机位置与接入中继关系。

每个热力图像素含义：
    假设一架虚拟测绘无人机位于该网格点、飞行高度为 USER_ALT_M，
    它连接当前 step 下的中继网络时，可获得的 RSSI / 双跳吞吐量。

核心逻辑：
    1) 虚拟测绘无人机 -> 中继：使用 air_air_data.csv 拟合模型；
    2) 中继 -> 地面站：使用 air_ground_data.csv 拟合模型；
    3) 双跳 DF 吞吐量；
    4) 每个网格点选择吞吐量最高的中继；
    5) 图上叠加真实测绘无人机当前位置，并标注 Sx->Ry。

输出：
    step_xxxx_air_user_rssi_field.png
    step_xxxx_air_user_rate_field.png
    step_xxxx_best_relay_field.png
"""
'''
带地图作为底图
'''
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 底图读取：优先使用带地理参考的 GeoTIFF
try:
    import rasterio
    from rasterio.warp import transform_bounds
except ImportError:
    rasterio = None
    transform_bounds = None


# ============================================================
# 1. 用户参数区：主要改这里
# ============================================================
CSV_PATH = r"runs_ppo_relay_shp_relative_obs_BIC_chibi/test_log.csv"

AIR_AIR_CSV = r"E:\毕业论文\2deployment\data\air_air_data.csv"
AIR_GROUND_CSV = r"E:\毕业论文\2deployment\data\air_ground_data.csv"

OUTPUT_DIR = r"runs_ppo_relay_shp_relative_obs_BIC_chibi/heatmaps/air_user_field_snapshots_with_basemap"

# 研究区底图：推荐使用带地理坐标的 GeoTIFF
# 你上传的 chibi_area.tif 带 EPSG:3857 坐标，脚本会自动转换为经纬度范围。
DRAW_BASEMAP = True
BASEMAP_PATH = r"E:\毕业论文\2deployment\插图\chibi_area_extract.tif"
# 如果使用普通 PNG/JPG，需要手动填写经纬度范围；GeoTIFF 时保持 None 即可。
# 格式：(lon_min, lon_max, lat_min, lat_max)
BASEMAP_EXTENT_LONLAT = None
BASEMAP_ALPHA = 1.0
HEATMAP_ALPHA = 0.5

# 需要输出的运行步
SNAPSHOT_STEPS = [1, 100, 200, 500, 1000]

# 热力图中“区域点”代表的虚拟测绘无人机高度
USER_ALT_M = 200.0

# 地面站高度，与你环境中的 GROUND_POS 第三个值一致
GROUND_ALT_M = 10.0

# 通信参数，与环境保持一致
TOTAL_BANDWIDTH_HZ = 20e6
BW_REF_HZ = 20e6
RATE_UNIT = "Mbps"
NOISE_DENSITY_W_PER_HZ = 1e-20
TWO_HOP_MODE = "df"  # "df" 或 "min"

# "per_user_equal": 每个虚拟用户使用 TOTAL_BANDWIDTH_HZ / survey数量
# "full": 每个虚拟用户使用 TOTAL_BANDWIDTH_HZ
BANDWIDTH_MODE = "per_user_equal"

GRID_SIZE = 180
AREA_PADDING_RATIO = 0.08

# 图形开关
DRAW_RELAYS = True
DRAW_GROUND = True
DRAW_SURVEY_POINTS = True
DRAW_SURVEY_LABELS = True
DRAW_RELAY_LABELS = True
DRAW_SURVEY_TO_RELAY_LINES = True

DPI = 300

# 色条范围
# RSSI 推荐固定范围，方便不同 step 对比
RSSI_VMIN = -105.0
RSSI_VMAX = -60.0

# 吞吐量色条范围；如果觉得太亮，把 RATE_VMAX_MBPS 改小，如 12 或 10
RATE_VMIN_MBPS = 0.0
RATE_VMAX_MBPS = 25.0


# ============================================================
# 2. 工具函数
# ============================================================
def infer_indices(cols: Iterable[str], prefix: str, suffix: str) -> List[int]:
    pat = re.compile(rf"^{re.escape(prefix)}(\d+){re.escape(suffix)}$")
    idx = set()
    for c in cols:
        m = pat.match(c)
        if m:
            idx.add(int(m.group(1)))
    return sorted(idx)


def haversine_m(lat1, lon1, lat2, lon2):
    """向量化 haversine 距离，单位 m。"""
    R = 6371000.0

    lat1 = np.asarray(lat1, dtype=np.float64)
    lon1 = np.asarray(lon1, dtype=np.float64)
    lat2 = np.asarray(lat2, dtype=np.float64)
    lon2 = np.asarray(lon2, dtype=np.float64)

    phi1 = np.deg2rad(lat1)
    phi2 = np.deg2rad(lat2)
    dphi = np.deg2rad(lat2 - lat1)
    dlmb = np.deg2rad(lon2 - lon1)

    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlmb / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(1.0 - a, 0.0)))
    return R * c


def load_poly2_from_csv(csv_path: str, rate_unit: str = "Mbps") -> Tuple[np.ndarray, Dict[str, float]]:
    """
    CSV:
        col0 = horizontal distance x, m
        col1 = vertical distance y, m
        col2 = rate, Mbps 或 bps

    拟合：
        C(x,y)=b0+b1*x+b2*y+b3*x^2+b4*x*y+b5*y^2
    """
    df = pd.read_csv(csv_path, header=None)

    x = df.iloc[:, 0].to_numpy(dtype=np.float64)
    y = df.iloc[:, 1].to_numpy(dtype=np.float64)
    r = df.iloc[:, 2].to_numpy(dtype=np.float64)

    if rate_unit.lower() == "mbps":
        r_bps = r * 1e6
    else:
        r_bps = r

    A = np.column_stack([np.ones_like(x), x, y, x ** 2, x * y, y ** 2])
    beta, *_ = np.linalg.lstsq(A, r_bps, rcond=None)

    bounds = {
        "x_min": float(np.min(x)), "x_max": float(np.max(x)),
        "y_min": float(np.min(y)), "y_max": float(np.max(y)),
    }

    return beta.astype(np.float64), bounds


def poly2_eval_bps(beta: np.ndarray, x_m, y_m, bounds: Dict[str, float]):
    """向量化二次多项式评估，超出实测范围时裁剪，避免外推发散。"""
    x = np.clip(np.asarray(x_m, dtype=np.float64), bounds["x_min"], bounds["x_max"])
    y = np.clip(np.asarray(y_m, dtype=np.float64), bounds["y_min"], bounds["y_max"])

    val = (
        beta[0]
        + beta[1] * x
        + beta[2] * y
        + beta[3] * x * x
        + beta[4] * x * y
        + beta[5] * y * y
    )
    return np.maximum(val, 0.0)


def pr_w_from_fitted_rate(c_ref_bps, bw_ref_hz: float = BW_REF_HZ):
    """
    由参考带宽下的拟合速率反推等效接收功率。
    这里不加入随机阴影衰落，保证热力图稳定可复现。
    """
    c_ref_bps = np.asarray(c_ref_bps, dtype=np.float64)
    se = c_ref_bps / max(float(bw_ref_hz), 1e-12)
    se = np.clip(se, 0.0, 25.0)

    snr = np.power(2.0, se) - 1.0
    pr = snr * NOISE_DENSITY_W_PER_HZ * float(bw_ref_hz)

    return np.maximum(pr, 0.0)


def rssi_dbm_from_pr(pr_w):
    pr_w = np.asarray(pr_w, dtype=np.float64)
    return 10.0 * np.log10(np.maximum(pr_w, 1e-30)) + 30.0


def rate_bps(bw_hz: float, pr_w):
    bw = max(float(bw_hz), 1e-12)
    pr_w = np.asarray(pr_w, dtype=np.float64)
    noise = NOISE_DENSITY_W_PER_HZ * bw
    snr = pr_w / max(noise, 1e-12)
    return bw * np.log2(1.0 + snr)


def two_hop_rate_bps(bw_hz: float, pr_user_relay_w, pr_relay_ground_w):
    c_ur = rate_bps(bw_hz, pr_user_relay_w)
    c_rg = rate_bps(bw_hz, pr_relay_ground_w)

    if TWO_HOP_MODE.lower() == "df":
        return (c_ur * c_rg) / (c_ur + c_rg + 1e-12)

    return np.minimum(c_ur, c_rg)


def get_study_extent(df: pd.DataFrame, relay_ids: List[int], survey_ids: List[int]):
    """用测绘无人机轨迹 + 中继轨迹共同确定固定研究区范围。"""
    lon_series = []
    lat_series = []

    for s in survey_ids:
        lon_col, lat_col = f"survey{s}_lon", f"survey{s}_lat"
        if lon_col in df.columns and lat_col in df.columns:
            lon_series.append(df[lon_col])
            lat_series.append(df[lat_col])

    for r in relay_ids:
        lon_col, lat_col = f"relay{r}_lon", f"relay{r}_lat"
        if lon_col in df.columns and lat_col in df.columns:
            lon_series.append(df[lon_col])
            lat_series.append(df[lat_col])

    if not lon_series or not lat_series:
        raise ValueError("Cannot infer study extent: no relay/survey lon-lat columns found.")

    lon_all = pd.concat(lon_series).dropna()
    lat_all = pd.concat(lat_series).dropna()

    lon_min, lon_max = float(lon_all.min()), float(lon_all.max())
    lat_min, lat_max = float(lat_all.min()), float(lat_all.max())

    lon_span = max(lon_max - lon_min, 1e-9)
    lat_span = max(lat_max - lat_min, 1e-9)

    lon_pad = lon_span * AREA_PADDING_RATIO
    lat_pad = lat_span * AREA_PADDING_RATIO

    return lon_min - lon_pad, lon_max + lon_pad, lat_min - lat_pad, lat_max + lat_pad


def nearest_row_by_step(df: pd.DataFrame, target_step: int) -> pd.Series:
    if "step" not in df.columns:
        raise ValueError("CSV must contain a 'step' column.")

    idx = (df["step"] - target_step).abs().idxmin()
    return df.loc[idx]



# def load_basemap_for_plot(basemap_path: str):
#     """
#     读取底图，并返回:
#         image, extent_lonlat
#
#     extent_lonlat 格式:
#         (lon_min, lon_max, lat_min, lat_max)
#
#     规则:
#         1) GeoTIFF：自动读取 CRS 和 bounds，并转换到 EPSG:4326；
#         2) PNG/JPG：必须手动设置 BASEMAP_EXTENT_LONLAT。
#     """
#     if not DRAW_BASEMAP:
#         return None, None
#
#     if not basemap_path or not os.path.exists(basemap_path):
#         print(f"[Warning] Basemap not found: {basemap_path}")
#         return None, None
#
#     suffix = Path(basemap_path).suffix.lower()
#
#     if suffix in [".tif", ".tiff"]:
#         if rasterio is None:
#             raise ImportError("读取 GeoTIFF 底图需要安装 rasterio。")
#
#         with rasterio.open(basemap_path) as src:
#             img = src.read()
#
#             # rasterio: (bands, height, width) -> matplotlib: (height, width, bands)
#             if img.shape[0] >= 3:
#                 img = np.transpose(img[:3], (1, 2, 0))
#             else:
#                 img = img[0]
#
#             # 归一化到 0~1，避免显示过曝
#             img = img.astype(np.float32)
#             if img.ndim == 3:
#                 max_val = np.nanmax(img)
#                 if max_val > 1.0:
#                     img = img / 255.0 if max_val <= 255 else img / max_val
#
#             if src.crs is None:
#                 if BASEMAP_EXTENT_LONLAT is None:
#                     raise ValueError("GeoTIFF 没有 CRS，请手动设置 BASEMAP_EXTENT_LONLAT。")
#                 extent = BASEMAP_EXTENT_LONLAT
#             else:
#                 left, bottom, right, top = transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21)
#                 extent = (left, right, bottom, top)
#
#             return img, extent
#
#     # 普通图片：必须手动提供经纬度范围
#     img = plt.imread(basemap_path)
#     if BASEMAP_EXTENT_LONLAT is None:
#         raise ValueError(
#             "PNG/JPG 没有地理参考，必须手动设置 BASEMAP_EXTENT_LONLAT=(lon_min, lon_max, lat_min, lat_max)。"
#         )
#
#     return img, BASEMAP_EXTENT_LONLAT

def load_basemap_for_plot(basemap_path: str):
    """
    读取底图，并返回:
        image, extent_lonlat

    extent_lonlat 格式:
        (lon_min, lon_max, lat_min, lat_max)
    """
    if not DRAW_BASEMAP:
        return None, None

    if not basemap_path or not os.path.exists(basemap_path):
        print(f"[Warning] Basemap not found: {basemap_path}")
        return None, None

    suffix = Path(basemap_path).suffix.lower()

    if suffix in [".tif", ".tiff"]:
        if rasterio is None:
            raise ImportError("读取 GeoTIFF 底图需要安装 rasterio。")

        with rasterio.open(basemap_path) as src:
            img = src.read().astype(np.float32)

            # 关键1：处理 NoData，尤其是 Extract by Mask 后常见的 65535
            nodata = src.nodata
            if nodata is not None:
                img[img == nodata] = np.nan

            # 关键2：bands, height, width -> height, width, bands
            if img.shape[0] >= 3:
                img = np.transpose(img[:3], (1, 2, 0))
            else:
                img = img[0]

            # 关键3：按有效像元做 2%-98% 拉伸，避免 NoData 把影像压黑
            valid = np.isfinite(img)
            if np.any(valid):
                p2, p98 = np.nanpercentile(img[valid], [2, 98])
                img = (img - p2) / max(p98 - p2, 1e-6)
                img = np.clip(img, 0, 1)
            else:
                img = np.zeros_like(img)

            # 关键4：NaN 区域设为透明/黑色背景，这里设为 0
            img = np.nan_to_num(img, nan=0.0)

            if src.crs is None:
                if BASEMAP_EXTENT_LONLAT is None:
                    raise ValueError("GeoTIFF 没有 CRS，请手动设置 BASEMAP_EXTENT_LONLAT。")
                extent = BASEMAP_EXTENT_LONLAT
            else:
                left, bottom, right, top = transform_bounds(
                    src.crs, "EPSG:4326", *src.bounds, densify_pts=21
                )
                extent = (left, right, bottom, top)

            return img, extent

    img = plt.imread(basemap_path)
    if BASEMAP_EXTENT_LONLAT is None:
        raise ValueError(
            "PNG/JPG 没有地理参考，必须手动设置 BASEMAP_EXTENT_LONLAT=(lon_min, lon_max, lat_min, lat_max)。"
        )

    return img, BASEMAP_EXTENT_LONLAT



# ============================================================
# 3. 通信场计算
# ============================================================
def compute_air_user_field(
    row: pd.Series,
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    relay_ids: List[int],
    beta_aa: np.ndarray,
    bounds_aa: Dict[str, float],
    beta_ag: np.ndarray,
    bounds_ag: Dict[str, float],
    link_bw_hz: float,
):
    """
    计算虚拟空中测绘用户在整个区域的通信场。

    返回：
        best_rssi: 每个网格点选择最优中继后的 user-relay RSSI
        best_rate: 每个网格点选择最优中继后的双跳吞吐量
        best_relay: 每个网格点选择的中继编号
    """
    if "ground_lat" not in row.index or "ground_lon" not in row.index:
        raise ValueError("CSV must contain ground_lat and ground_lon columns.")

    ground_lat = float(row["ground_lat"])
    ground_lon = float(row["ground_lon"])

    best_rate = np.full(lon_grid.shape, -np.inf, dtype=np.float64)
    best_rssi = np.full(lon_grid.shape, -300.0, dtype=np.float64)
    best_relay = np.full(lon_grid.shape, -1, dtype=np.int32)

    for r in relay_ids:
        relay_lat = float(row[f"relay{r}_lat"])
        relay_lon = float(row[f"relay{r}_lon"])
        relay_alt = float(row[f"relay{r}_alt_m"])

        # 虚拟测绘无人机 -> 中继，空空链路
        d_ur_h = haversine_m(lat_grid, lon_grid, relay_lat, relay_lon)
        dz_ur = abs(USER_ALT_M - relay_alt)

        c_ur_ref = poly2_eval_bps(beta_aa, d_ur_h, dz_ur, bounds_aa)
        pr_ur = pr_w_from_fitted_rate(c_ur_ref)
        rssi_ur = rssi_dbm_from_pr(pr_ur)

        # 中继 -> 地面站，空地链路；对同一个中继，该值在网格上为常数
        d_rg_h = haversine_m(relay_lat, relay_lon, ground_lat, ground_lon)
        dz_rg = abs(relay_alt - GROUND_ALT_M)

        c_rg_ref = poly2_eval_bps(beta_ag, d_rg_h, dz_rg, bounds_ag)
        pr_rg = pr_w_from_fitted_rate(c_rg_ref)

        field_rate = two_hop_rate_bps(link_bw_hz, pr_ur, pr_rg)

        mask = field_rate > best_rate
        best_rate[mask] = field_rate[mask]
        best_rssi[mask] = rssi_ur[mask]
        best_relay[mask] = int(r)

    best_rate = np.where(np.isfinite(best_rate), best_rate, 0.0)

    return best_rssi, best_rate, best_relay


# ============================================================
# 4. 绘图
# ============================================================
def draw_network_objects(
        ax,
        row: pd.Series,
        df: pd.DataFrame,
        relay_ids: List[int],
        survey_ids: List[int],
):
    """在热力图上叠加中继、地面站、真实测绘无人机及连接关系。"""
    # 中继
    if DRAW_RELAYS:
        for r in relay_ids:
            x = float(row[f"relay{r}_lon"])
            y = float(row[f"relay{r}_lat"])
            ax.scatter(
                x, y,
                marker="^",
                s=95,
                edgecolors="k",
                linewidths=0.8,
                label=f"Relay {r}",
                zorder=6,
            )
            if DRAW_RELAY_LABELS:
                ax.text(x, y, f" R{r}", fontsize=9, weight="bold", zorder=7)

    # 地面站
    if DRAW_GROUND:
        ax.scatter(
            float(row["ground_lon"]),
            float(row["ground_lat"]),
            marker="*",
            s=190,
            c="limegreen",
            edgecolors="k",
            linewidths=0.8,
            label="Ground Station",
            zorder=7,
        )

    current_step = int(row["step"])

    for s in survey_ids:
        lon_col = f"survey{s}_lon"
        lat_col = f"survey{s}_lat"

        if lon_col not in df.columns or lat_col not in df.columns:
            continue

        # 完整测绘路线：浅色虚线
        ax.plot(
            df[lon_col],
            df[lat_col],
            linestyle="--",
            linewidth=0.9,
            alpha=0.35,
            label="Survey planned route" if s == survey_ids[0] else None,
            zorder=3,
        )

        # 已飞轨迹：深色实线
        hist = df[df["step"] <= current_step]
        ax.plot(
            hist[lon_col],
            hist[lat_col],
            linestyle="-",
            linewidth=0.6,
            alpha=0.25,
            color="black",
            label="Survey flown route" if s == survey_ids[0] else None,
            zorder=4,
        )
    # 测绘无人机及其接入中继
    if DRAW_SURVEY_POINTS:
        for s in survey_ids:
            lat_col = f"survey{s}_lat"
            lon_col = f"survey{s}_lon"
            relay_col = f"sel_relay_{s}"

            if lat_col not in row.index or lon_col not in row.index:
                continue

            sx = float(row[lon_col])
            sy = float(row[lat_col])
            relay_id = int(row[relay_col]) if relay_col in row.index and not pd.isna(row[relay_col]) else -1

            ax.scatter(
                sx, sy,
                marker="o",
                s=42,
                c="white",
                edgecolors="k",
                linewidths=0.8,
                label="Survey UAV" if s == survey_ids[0] else None,
                zorder=8,
            )

            if DRAW_SURVEY_LABELS:
                if relay_id >= 0:
                    label = f" S{s}->R{relay_id}"
                else:
                    label = f" S{s}"
                ax.text(sx, sy, label, fontsize=8, weight="bold", zorder=9)

            # 画 survey -> selected relay 的连线
            if DRAW_SURVEY_TO_RELAY_LINES and relay_id >= 0:
                rx_col = f"relay{relay_id}_lon"
                ry_col = f"relay{relay_id}_lat"

                if rx_col in row.index and ry_col in row.index:
                    rx = float(row[rx_col])
                    ry = float(row[ry_col])
                    ax.annotate(
                        "",
                        xy=(rx, ry),
                        xytext=(sx, sy),
                        arrowprops=dict(
                            arrowstyle="->",
                            lw=1.2,
                            color="black",
                            alpha=0.65,
                            linestyle="--",
                            shrinkA=8,
                            shrinkB=10,
                            mutation_scale=13,
                        ),
                        zorder=6,
                    )

def plot_field(
    field: np.ndarray,
    extent: Tuple[float, float, float, float],
    row: pd.Series,
    df: pd.DataFrame,
    relay_ids: List[int],
    survey_ids: List[int],
    title: str,
    colorbar_label: str,
    save_path: Path,
    vmin=None,
    vmax=None,
    cmap: str = "viridis",
):
    fig, ax = plt.subplots(figsize=(10, 6))

    # 1) 先画研究区底图
    basemap_img, basemap_extent = load_basemap_for_plot(BASEMAP_PATH)
    if basemap_img is not None and basemap_extent is not None:
        ax.imshow(
            basemap_img,
            origin="upper",
            extent=basemap_extent,
            aspect="auto",
            alpha=BASEMAP_ALPHA,
            zorder=0,
        )

    # 2) 再叠加半透明热力图
    im = ax.imshow(
        field,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=HEATMAP_ALPHA if DRAW_BASEMAP else 1.0,
        zorder=1,
    )

    # 保持最终显示范围仍然是研究区范围
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])

    draw_network_objects(ax, row, df, relay_ids, survey_ids)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.grid(True, alpha=0.25, zorder=2)

    # 去重图例
    handles, labels = ax.get_legend_handles_labels()
    unique = {}
    for h, lab in zip(handles, labels):
        if lab and lab not in unique:
            unique[lab] = h

    ax.legend(unique.values(), unique.keys(), loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI)
    plt.close(fig)


# ============================================================
# 5. 主程序
# ============================================================
def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV_PATH not found: {CSV_PATH}")
    if not os.path.exists(AIR_AIR_CSV):
        raise FileNotFoundError(f"AIR_AIR_CSV not found: {AIR_AIR_CSV}")
    if not os.path.exists(AIR_GROUND_CSV):
        raise FileNotFoundError(f"AIR_GROUND_CSV not found: {AIR_GROUND_CSV}")

    df = pd.read_csv(CSV_PATH)

    relay_ids = infer_indices(df.columns, "relay", "_lat")
    survey_ids = infer_indices(df.columns, "survey", "_lat")

    if not relay_ids:
        raise ValueError("No relay*_lat columns found in CSV.")
    if not survey_ids:
        raise ValueError("No survey*_lat columns found in CSV.")

    beta_aa, bounds_aa = load_poly2_from_csv(AIR_AIR_CSV, RATE_UNIT)
    beta_ag, bounds_ag = load_poly2_from_csv(AIR_GROUND_CSV, RATE_UNIT)

    extent = get_study_extent(df, relay_ids, survey_ids)
    lon_min, lon_max, lat_min, lat_max = extent

    lon_vec = np.linspace(lon_min, lon_max, GRID_SIZE)
    lat_vec = np.linspace(lat_min, lat_max, GRID_SIZE)
    lon_grid, lat_grid = np.meshgrid(lon_vec, lat_vec)

    if BANDWIDTH_MODE == "full":
        link_bw_hz = TOTAL_BANDWIDTH_HZ
    elif BANDWIDTH_MODE == "per_user_equal":
        link_bw_hz = TOTAL_BANDWIDTH_HZ / max(len(survey_ids), 1)
    else:
        raise ValueError("BANDWIDTH_MODE must be 'full' or 'per_user_equal'.")

    print(f"Relay IDs: {relay_ids}")
    print(f"Survey IDs: {survey_ids}")
    print(f"Virtual aerial user altitude: {USER_ALT_M} m")
    print(f"Bandwidth mode: {BANDWIDTH_MODE}, link_bw_hz={link_bw_hz:.3f}")
    print(f"Study extent: lon[{lon_min:.8f}, {lon_max:.8f}], lat[{lat_min:.8f}, {lat_max:.8f}]")
    print(f"Output directory: {out_dir}")

    for step in SNAPSHOT_STEPS:
        row = nearest_row_by_step(df, step)
        actual_step = int(row["step"])

        if actual_step != step:
            print(f"Requested step {step}, using nearest available step {actual_step}.")

        rssi_field, rate_field_bps, best_relay = compute_air_user_field(
            row=row,
            lon_grid=lon_grid,
            lat_grid=lat_grid,
            relay_ids=relay_ids,
            beta_aa=beta_aa,
            bounds_aa=bounds_aa,
            beta_ag=beta_ag,
            bounds_ag=bounds_ag,
            link_bw_hz=link_bw_hz,
        )

        rate_field_mbps = rate_field_bps / 1e6

        plot_field(
            field=rssi_field,
            extent=extent,
            row=row,
            df=df,
            relay_ids=relay_ids,
            survey_ids=survey_ids,
            title=f"Aerial User RSSI Field at Step {actual_step} (alt={USER_ALT_M:.0f} m)",
            colorbar_label="RSSI of selected user-relay link (dBm)",
            save_path=out_dir / f"step_{actual_step:04d}_air_user_rssi_field.png",
            vmin=RSSI_VMIN,
            vmax=RSSI_VMAX,
            cmap="viridis",
        )

        plot_field(
            field=rate_field_mbps,
            extent=extent,
            row=row,
            df=df,
            relay_ids=relay_ids,
            survey_ids=survey_ids,
            title=f"Aerial User Two-hop Throughput Field at Step {actual_step} (alt={USER_ALT_M:.0f} m)",
            colorbar_label="Two-hop throughput (Mbps)",
            save_path=out_dir / f"step_{actual_step:04d}_air_user_rate_field.png",
            vmin=RATE_VMIN_MBPS,
            vmax=RATE_VMAX_MBPS,
            cmap="plasma",
        )

        plot_field(
            field=best_relay.astype(float),
            extent=extent,
            row=row,
            df=df,
            relay_ids=relay_ids,
            survey_ids=survey_ids,
            title=f"Best Relay Selection Field at Step {actual_step} (alt={USER_ALT_M:.0f} m)",
            colorbar_label="Selected relay ID",
            save_path=out_dir / f"step_{actual_step:04d}_best_relay_field.png",
            vmin=min(relay_ids),
            vmax=max(relay_ids),
            cmap="tab10",
        )

        print(f"Saved fields for step {actual_step}.")

    print("Done.")


if __name__ == "__main__":
    main()
