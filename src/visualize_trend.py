# src/visualize_trend_compare.py

import matplotlib.pyplot as plt

from src.dataloader import load_match_groups
import matplotlib.pyplot as plt
import numpy as np

from src.show import extract_signed_area_contour
from src.trend_segmentation import contour_to_trend_segments


def plot_trend_segments(ax, segments, title):
    """
    在指定 ax 上绘制趋势段
    segments: [
        {"trend": "+", "start": 0, "end": 18, "value": 12.3},
        ...
    ]
    """
    for seg in segments:
        start = seg["start"]
        end = seg["end"]
        width = end - start
        value = seg["value"]
        trend = seg["trend"]

        color = "green" if trend == "+" else "red"

        ax.bar(
            start,
            value,
            width=width,
            align="edge",
            color=color,
            alpha=0.6,
            edgecolor="black"
        )

        # 趋势标注
        ax.text(
            start + width / 2,
            value * 0.6,
            trend,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="white"
        )

    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylim(-100, 100)
    ax.set_ylabel("Trend Value")
    ax.set_title(title)


def visualize_match_trend_segments(
    match_data,
    window=12
):
    """
    输入:
    match_data = {
        "main": {"id": ..., "series": np.ndarray},
        "subs": [ {"id": ..., "series": np.ndarray}, ... ]
    }
    """

    main = match_data["main"]
    subs = match_data["subs"]

    total = 1 + len(subs)
    fig, axes = plt.subplots(
        total,
        1,
        figsize=(14, 3 * total),
        sharex=True
    )

    if total == 1:
        axes = [axes]

    # ===== MAIN =====
    main_contour = extract_signed_area_contour(
        main["series"],
        window=window
    )
    main_segments = contour_to_trend_segments(main_contour)

    plot_trend_segments(
        axes[0],
        main_segments,
        title=f"MAIN: {main['id']}"
    )

    # ===== SUBS =====
    for i, sub in enumerate(subs, start=1):
        sub_contour = extract_signed_area_contour(
            sub["series"],
            window=window
        )
        sub_segments = contour_to_trend_segments(sub_contour)

        plot_trend_segments(
            axes[i],
            sub_segments,
            title=f"SUB: {sub['id']}"
        )

    axes[-1].set_xlabel("Index / Time")

    plt.tight_layout()
    plt.show()


#  main_contour = extract_signed_area_contour(main["series"])
#  main_seg = contour_to_trend_segments(main_contour)
def visualize_main_sub_trend_compare(
    main_segments,
    sub_segments,
    main_id="MAIN",
    sub_id="SUB"
):
    """
    主 / 子 match 趋势段对比图
    """

    fig, axes = plt.subplots(
        2, 1,
        figsize=(14, 6),
        sharex=True
    )

    # ===== 主趋势 =====
    plot_trend_segments(
        axes[0],
        main_segments,
        title=f"MAIN: {main_id}"
    )

    # ===== 子趋势 =====
    plot_trend_segments(
        axes[1],
        sub_segments,
        title=f"SUB: {sub_id}"
    )

    axes[1].set_xlabel("Time / Index")

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    excel_path = "../data/2.xlsx"
    data = load_match_groups(excel_path)


    match_id = "2025/05/18-55VS53-60"
    print(data[match_id])
    # contour = contour_to_variable_trends
    visualize_match_trend_segments(
        data[match_id],
        window=3
    )
