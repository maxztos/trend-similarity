import numpy as np
import matplotlib.pyplot as plt

from src.timeline_match import slide_and_match, get_timeline


def visualize_timeline_match(
    main_tl,
    sub_tl,
    match_result,
    title_main="Main",
    title_sub="Sub",
    y_lim=(-100, 100),
):
    """
    main_tl: np.ndarray
    sub_tl: np.ndarray
    match_result: {
        "direction": "left" | "right",
        "shift": int,
        "score": float
    }
    """

    direction = match_result["direction"]
    shift = match_result["shift"]
    score = match_result["score"]

    len_m = len(main_tl)
    len_s = len(sub_tl)

    # ===== 计算对齐后的 sub timeline 映射到 main 轴 =====
    aligned_sub = np.full(len_m, np.nan)

    if direction == "left":
        start = shift
        end = min(len_m, shift + len_s)
        aligned_sub[start:end] = sub_tl[: end - start]

    elif direction == "right":
        end = len_m - shift
        start = max(0, end - len_s)
        aligned_sub[start:end] = sub_tl[len_s - (end - start) :]

    x = np.arange(len_m)

    # ===== 开始画图 =====
    plt.figure(figsize=(14, 6))

    # --- Main ---
    plt.subplot(2, 1, 1)
    plt.bar(x, main_tl, width=0.9)
    plt.axhline(0, linestyle="--")
    plt.ylim(*y_lim)
    plt.title(title_main)
    plt.ylabel("Value")

    # --- Sub ---
    plt.subplot(2, 1, 2)
    plt.bar(x, aligned_sub, width=0.9)
    plt.axhline(0, linestyle="--")
    plt.ylim(*y_lim)
    plt.title(
        f"{title_sub} | direction={direction}, shift={shift}, score={score:.3f}"
    )
    plt.ylabel("Value")
    plt.xlabel("Time Index")

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # main_timeline / sub_timeline 已算好
    main_timeline, sub_timeline1, sub_timeline2 = get_timeline("2025/05/18-29VS174-60")
    matches = slide_and_match(main_timeline, sub_timeline2)

    best = max(matches, key=lambda x: x["score"])

    visualize_timeline_match(
        main_timeline,
        sub_timeline2,
        best,
        title_main="Main Match",
        title_sub="Sub Match"
    )
