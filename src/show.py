import matplotlib.pyplot as plt
import numpy as np
import cv2

from src.preprocess import extract_series_from_image, extract_bar_series


def plot_main_vs_sub(main_series, sub_series, score=None, title=None):
    main_series = np.asarray(main_series)
    sub_series = np.asarray(sub_series)

    x = np.arange(len(main_series))

    plt.figure(figsize=(12, 5))

    plt.plot(x, main_series, label="Main", linewidth=2)
    plt.plot(x, sub_series, label="Sub", linewidth=2, linestyle="--")

    plt.axhline(0, linestyle=":", linewidth=1)

    plt.xlabel("Bin Index")
    plt.ylabel("Value")

    if title:
        plt.title(title)
    elif score is not None:
        plt.title(f"Main vs Sub (Similarity Score = {score})")
    else:
        plt.title("Main vs Sub Comparison")

    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_difference(main_series, sub_series):
    diff = main_series - sub_series
    x = np.arange(len(diff))

    plt.figure(figsize=(12, 3))
    plt.bar(x, diff)
    plt.axhline(0, linestyle=":")
    plt.title("Difference (Main - Sub)")
    plt.xlabel("Bin Index")
    plt.ylabel("Diff")
    plt.tight_layout()
    plt.show()

def draw_series_on_image(
    img_path,
    series,
    meta,
    color=(0, 0, 255),
    thickness=2
):
    # 1️⃣ 读图
    img = cv2.imread(img_path)
    assert img is not None, f"Cannot read image: {img_path}"

    # 2️⃣ resize（必须和 extractor 一致）
    img = cv2.resize(img, (meta["width"], meta["height"]))

    h, w = meta["height"], meta["width"]
    bins = meta["bins"]

    x_ignore = meta["x_ignore"]
    y_top = meta["y_top"]
    y_zero = meta["y_zero"]
    y_bottom = meta["y_bottom"]

    usable_width = w - x_ignore
    bin_width = usable_width / bins
    pixels_per_unit = (y_bottom - y_top) / 200

    # 3️⃣ 计算曲线点
    points = []
    for i, v in enumerate(series):
        x = int(x_ignore + (i + 0.5) * bin_width)
        y = int(y_zero - v * pixels_per_unit)
        points.append((x, y))

    # 4️⃣ 画曲线
    for i in range(len(points) - 1):
        cv2.line(img, points[i], points[i + 1], color, thickness)

    # 5️⃣ 辅助线（可选）
    cv2.line(img, (0, y_zero), (w, y_zero), (0, 255, 0), 1)

    # ✅ 6️⃣ 一定要 return
    return img

def rescale_amplitude(series, target_min=-70, target_max=70):
    s = np.asarray(series, dtype=np.float32)

    s_min, s_max = s.min(), s.max()
    if s_max - s_min < 1e-6:
        return s

    s_norm = (s - s_min) / (s_max - s_min)
    return s_norm * (target_max - target_min) + target_min

if __name__ == '__main__':

    main_img = "D:/CProject/ImageProcessing/Trend-similarity/data/sample2/fig1.png"
    sub_img = "D:/CProject/ImageProcessing/Trend-similarity/data/sample2/fig1.png"
    main_series = extract_series_from_image(main_img)
    sub_series = extract_series_from_image(sub_img)

    series, meta = extract_bar_series(main_img, bins=60)
    series = rescale_amplitude(series)
    print("Extracted series length:", len(series))
    print("Meta info:", meta)

    # 3️⃣ 画回原图
    img_overlay = draw_series_on_image(
        main_img,
        series,
        meta,
        color=(0, 0, 255)  # 红色
    )

    # 4️⃣ 显示
    cv2.imshow("Overlay Result", img_overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()