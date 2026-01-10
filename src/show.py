import numpy as np
from scipy.signal import savgol_filter
from matplotlib.collections import LineCollection

from src.utils.dataloader import load_match_groups


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
    window=3,   # 人眼感知宽度（10~20 推荐）
    smooth=7,    # 视觉平滑（必须奇数）
    poly=3  # 决定轮廓“弯不弯”
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

        area = np.sum(seg) / len(seg)
        peak = np.max(np.abs(seg))
        mean_amp = np.mean(np.abs(seg)) + 1e-6

        # contour[i] = area * (peak / mean_amp)
        contour[i] = round(area * (peak / mean_amp), 2)
        # 🔥 带符号面积（人眼判断核心）
        # contour[i] = np.sum(seg) / len(seg)

    # 仅用于视觉连续，不改变语义
    if smooth >= 5 and smooth < n:
        contour = savgol_filter(contour, smooth, poly)
    # print(contour)
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

# 绘制轮廓曲线
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
    # text = f"{prefix}: [{', '.join(trends)}]"
    text = f"{prefix}: [{', '.join(str(t) for t in trends)}]"

    plt.gca().text(
        0.5, -0.25,          # ⬅️ 关键：轴坐标（居中、在下方）
        text,
        ha="center",
        va="top",
        fontsize=12,
        transform=plt.gca().transAxes
    )


def draw_mean_lines(ax, contour, x_offset=0.99):
    """
    在 ax 上绘制：
    - 全局均值
    - 正值均值
    - 负值均值
    并在右侧显示数值
    """
    contour = np.asarray(contour)

    mean_all = np.mean(contour)

    pos_vals = contour[contour > 0]
    neg_vals = contour[contour < 0]

    mean_pos = np.mean(pos_vals) if len(pos_vals) > 0 else None
    mean_neg = np.mean(neg_vals) if len(neg_vals) > 0 else None

    amp = mean_pos - mean_neg
    # x 位置（按坐标轴比例）
    x = ax.get_xlim()[0] + x_offset * (ax.get_xlim()[1] - ax.get_xlim()[0])

    # ===== 全局均值 =====
    ax.axhline(mean_all, color="gray", linestyle="--", linewidth=1.2)
    ax.text(
        x, mean_all,
        f"{mean_all:.1f}",
        color="gray",
        fontsize=9,
        va="center",
        ha="right",
        backgroundcolor="white"
    )

    # ===== 正势均值 =====
    if mean_pos is not None:
        ax.axhline(mean_pos, color="red", linestyle=":", linewidth=1.5)
        ax.text(
            x, mean_pos,
            f"+{mean_pos:.1f}",
            color="red",
            fontsize=9,
            va="center",
            ha="right",
            backgroundcolor="white"
        )

    # ===== 负势均值 =====
    if mean_neg is not None:
        ax.axhline(mean_neg, color="blue", linestyle=":", linewidth=1.5)
        ax.text(
            x, mean_neg,
            f"{mean_neg:.1f}",
            color="blue",
            fontsize=9,
            va="center",
            ha="right",
            backgroundcolor="white"
        )

    ax.text(
        x, amp,
        f"AMP:{amp:.1f}--",
        color="red",
        fontsize=9,
        va="center",
        ha="right",
        backgroundcolor="white"
    )


def visualize_match_with_signed_contour(
    match_data,
    window=3,
    trend_window=5
):
    main = match_data["main"]
    subs = match_data["subs"]

    total = 1 + len(subs)
    plt.figure(figsize=(14, 3 * total))

    # ===== 主图 =====
    ax = plt.subplot(total, 1, 1)
    main_contour = plot_series_with_contour(
        main["series"],
        window=window,
        title=f"MAIN: {main['id']}"
    )
    # print(main_contour)
    ax.set_ylim(-100, 100)
    # ⭐ 叠加均值线
    draw_mean_lines(ax, main["series"])

    # 只在主图显示 legend（避免太乱）
    # ax.legend(loc="upper right", fontsize=9)
    # ===== 子图 =====
    for i, sub in enumerate(subs, start=2):
        ax = plt.subplot(total, 1, i)
        sub_contour = plot_series_with_contour(
            sub["series"],
            window=window,
            title=f"SUB: {sub['id']}"
        )
        ax.set_ylim(-100, 100)
        draw_mean_lines(ax, sub["series"])
    plt.tight_layout()
    plt.show()

def plot_trend_segments_bar(segments):
    """
    根据 trend segments 画区间柱状图
    """
    plt.figure(figsize=(12, 4))

    for seg in segments:
        start = seg["start"]
        width = seg["end"] - seg["start"]
        value = seg["value"]
        trend = seg["trend"]

        color = "green" if trend == "+" else "red"

        plt.bar(
            start,
            value,
            width=width,
            align="edge",
            color=color,
            alpha=0.6,
            edgecolor="black"
        )

        # 可选：在柱子中间标注 + / -
        plt.text(
            start + width / 2,
            value * 0.5,
            trend,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="white"
        )

    plt.axhline(0, color="black", linewidth=1)
    plt.ylim(-100, 100)
    plt.xlabel("Time / Index")
    plt.ylabel("Trend Value (mean contour)")
    plt.title("Trend Segments Bar Visualization")

    plt.tight_layout()
    plt.show()
def visualize_contour(match_data):
    main = match_data["main"]
    subs = match_data["subs"]
    total = 1 + len(subs)
    plt.figure(figsize=(14, 3 * total))

    # main
    ax = plt.subplot(total, 1, 1)
    main_contour = extract_signed_area_contour(main["series"])


def format_filename(match_id):
    """将 match_id 转换为 Windows 合法的安全文件名"""
    # 将 2025/05/10 替换为 20250510 或 2025_05_10
    # 这里建议直接去掉斜杠，符合你要求的 20250518 格式
    return match_id.replace("/", "").replace(":", "_")
# if __name__ == '__main__':
#     excel_path = "../data/2.xlsx"
#     data = load_match_groups(excel_path)
#
#     match_id = "2025/05/10-161VS211-61"
#
#     visualize_match_with_signed_contour(
#         data[match_id],
#         window=3
#     )

def format_filename(match_id):
    """将 match_id 转换为 Windows 合法的安全文件名"""
    # 按照你的要求：2025/05/18 -> 20250518
    return match_id.replace("/", "").replace(":", "_")


import matplotlib

matplotlib.use('Agg')  # 强制使用非交互式后端，拦截所有 plt.show()
import matplotlib.pyplot as plt
import os


def format_filename(match_id):
    return match_id.replace("/", "").replace(":", "_")


def batch_process_visualizations(data, id_list, output_folder="visual_results"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for m_id in id_list:
        m_id = m_id.strip()
        if m_id not in data:
            continue

        try:
            # 1. 清理之前的残余画布
            # 1. 清理
            plt.close('all')

            # 2. 调用原函数（它内部应该已经设好了 figsize）
            visualize_match_with_signed_contour(data[m_id], window=3)

            # 3. 获取当前画布
            fig = plt.gcf()

            if fig.get_axes():
                safe_filename = format_filename(m_id)
                save_path = os.path.join(output_folder, f"{safe_filename}.png")

                # 直接保存，不要 set_size_inches
                # bbox_inches='tight' 会自动裁掉多余白边
                plt.savefig(save_path, bbox_inches='tight', dpi=120)
                print(f"  ---> 成功保存（原尺寸）: {save_path}")

        except Exception as e:
            print(f"处理 ID [{m_id}] 出错: {e}")
        finally:
            plt.close('all')


if __name__ == '__main__':
    excel_path = "../data/2n.xlsx"
    data = load_match_groups(excel_path)

    # 你提供的 ID 列表
    match_ids = [
        "2025/04/27-276VS72-62",
        "2025/04/28-76VS78-60",
        "2025/05/02-4852VS109-60",
        "2025/05/03-13VS183-60",
        "2025/05/03-78VS86-60",
        "2025/05/04-84VS73-60",
        "2025/05/05-2783VS51-60",
        "2025/05/11-76VS79-61",
        "2025/05/11-174VS14-60",
        "2025/05/15-131VS63-60",
        "2025/05/15-64VS54-60",
        "2025/05/16-15VS32-68",
        "2025/05/17-1730VS796-62",
        "2025/05/18-29VS174-60",
        "2025/05/25-165VS29-67"
    ]

    # 执行批量保存
    batch_process_visualizations(data, match_ids)
    print("\n所有图像批量处理完成！")
