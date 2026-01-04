import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from matplotlib.collections import LineCollection

from src.dataloader import load_match_groups
from src.trend_segmentation import contour_to_variable_trends


# 绘制一个序列的图形
def plot_series_bar(series, title=None):
    series = np.asarray(series)
    x = np.arange(len(series))

    colors = np.where(series >= 0, "green", "orange")

    plt.bar(
        x,
        series,
        color=colors,
        width=0.9,
        alpha=0.75,
        zorder=1
    )

    plt.axhline(0, color="black", linewidth=1)
    plt.ylim(-100, 100)
    plt.xlabel("Index")
    plt.ylabel("Value")

    if title:
        plt.title(title)

# 生成轮廓线
def extract_signed_area_contour(
    series,
    window=15,   # 人眼感知宽度（10~20 推荐）
    smooth=5,    # 视觉平滑（必须奇数）
    poly=2
):
    """
    返回一条：代表局部柱状“整体面积感”的轮廓线
    """
    series = np.asarray(series)
    n = len(series)

    half = window // 2
    contour = np.zeros(n)

    for i in range(n):
        l = max(0, i - half)
        r = min(n, i + half + 1)
        seg = series[l:r]

        # 🔥 带符号面积（人眼判断核心）
        contour[i] = np.sum(seg) / len(seg)

    # 仅用于视觉连续，不改变语义
    if smooth >= 5 and smooth < n:
        contour = savgol_filter(contour, smooth, poly)

    return contour

def plot_signed_contour(contour):
    x = np.arange(len(contour))

    # 构造连续线段
    points = np.array([x, contour]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # 按线段中点的正负决定颜色
    colors = [
        "red" if (contour[i] + contour[i + 1]) / 2 >= 0 else "blue"
        for i in range(len(contour) - 1)
    ]

    lc = LineCollection(
        segments,
        colors=colors,
        linewidths=3,
        alpha=0.95,
        zorder=3
    )

    plt.gca().add_collection(lc)

def plot_series_with_contour(
    series,
    window=3,
    title=None
):
    series = np.asarray(series)
    x = np.arange(len(series))

    colors = np.where(series >= 0, "green", "orange")
    plt.bar(x, series, color=colors, alpha=0.6)
    plt.axhline(0, color="black", linewidth=1)

    contour = extract_signed_area_contour(series, window=window)
    cx = np.linspace(0, len(series) - 1, len(contour))
    plt.plot(cx, contour, color="blue", linewidth=2)

    if title:
        plt.title(title)

    return contour
# 把趋势列表画到图上
def annotate_trend_sequence(
    trends,
    prefix="trend"
):
    """
    在当前 subplot 底部添加趋势文本，如:
    trend: [+, 0, -, +]
    """
    text = f"{prefix}: [{', '.join(trends)}]"

    plt.gca().text(
        0.5, -0.25,          # ⬅️ 关键：轴坐标（居中、在下方）
        text,
        ha="center",
        va="top",
        fontsize=12,
        transform=plt.gca().transAxes
    )

def visualize_series_with_signed_contour(
    series,
    title=None,
    window=15
):
    plt.figure(figsize=(14, 4))

    # 原始柱状
    plot_series_bar(series, title=title)

    # 面积轮廓线
    contour = extract_signed_area_contour(
        series,
        window=window
    )
    plot_signed_contour(contour)

    plt.tight_layout()
    plt.show()

def visualize_match_with_signed_contour(
    match_data,
    window=3,
    trend_window=8
):
    main = match_data["main"]
    subs = match_data["subs"]

    total = 1 + len(subs)
    plt.figure(figsize=(14, 3 * total))

    # ===== 主图 =====
    plt.subplot(total, 1, 1)
    main_contour = plot_series_with_contour(
        main["series"],
        window=window,
        title=f"MAIN: {main['id']}"
    )

    main_trend = contour_to_variable_trends(
        main_contour,
        window_size=trend_window,
    )
    main["trend_seq"] = main_trend

    annotate_trend_sequence(main_trend)

    # ===== 子图 =====
    for i, sub in enumerate(subs, start=2):
        plt.subplot(total, 1, i)
        sub_contour = plot_series_with_contour(
            sub["series"],
            window=window,
            title=f"SUB: {sub['id']}"
        )

        sub_trend = contour_to_variable_trends(
            sub_contour,
            window_size=trend_window,
        )
        sub["trend_seq"] = sub_trend

        annotate_trend_sequence(sub_trend)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    excel_path = "../data/2.xlsx"
    data = load_match_groups(excel_path)

    match_id = "2025/05/05-2783VS51-60"
    visualize_match_with_signed_contour(
        data[match_id],
        window=3
    )

