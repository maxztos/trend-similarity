from scipy.ndimage import gaussian_filter1d
import numpy as np
import cv2
import matplotlib.pyplot as plt
# 预处理(归一化/对齐)
# 1.灰度化
#
# 2.二值化（柱子 vs 背景）
#
# 3.去掉坐标轴、网格线
#
# 4.找出柱子所在区域（ROI）
# ┌──────────────────────────┐
# │ 原始柱状图（image）       │
# ├──────────────────────────┤
# │ 原始序列（raw, len≈552） │
# ├──────────────────────────┤
# │ 插值后序列（resampled）  │
# └──────────────────────────┘
X_IGNORE_RATIO = 0.1
def extract_bar_series(img_path, bins=60, y_min=-100, y_max=100):
    import cv2, numpy as np

    img = cv2.imread(img_path)
    img = cv2.resize(img, (800, 600))  # 强烈推荐

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    h, w = binary.shape
    x_ignore = int(X_IGNORE_RATIO * w)
    usable_width = w - x_ignore
    bin_width = usable_width / bins

    # 比例定义（分辨率无关）
    y_top = int(0.1 * h)
    y_bottom = int(0.9 * h)
    y_zero = int(0.5 * h)

    pixels_per_unit = (y_bottom - y_top) / (y_max - y_min)

    series = []
    # bin_width = w / bins

    for i in range(bins):
        xs = int(x_ignore + i * bin_width)
        xe = int(x_ignore + (i + 1) * bin_width)
        col = binary[y_top:y_bottom, xs:xe]

        ys, _ = np.where(col > 0)

        if len(ys) == 0:
            series.append(0.0)
            continue

        y_pixel = y_top + ys.mean()
        value = (y_zero - y_pixel) / pixels_per_unit
        series.append(value)

    meta = {
        "width": 800,
        "height": 600,
        "bins": bins,
        "x_ignore": x_ignore,
        "y_top": y_top,
        "y_zero": y_zero,
        "y_bottom": y_bottom
    }

    return series, meta



def resample_series_interp_smooth(series, target_len=60, sigma=1.5):
    series= np.asarray(series, dtype=np.float32)

    # 先去一点毛刺
    series = gaussian_filter1d(series, sigma=sigma)

    x_old = np.linspace(0, 1, len(series))
    x_new = np.linspace(0, 1, target_len)

    return np.interp(x_new, x_old, series)

def visualize_extraction(img_path, raw_series, resampled_series):
    # 读取原图
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig = plt.figure(figsize=(12, 8))

    # 1️⃣ 原图
    ax1 = plt.subplot(3, 1, 1)
    ax1.imshow(img)
    ax1.axvline(x=img.shape[1] * X_IGNORE_RATIO,
                color="red", linestyle="--", linewidth=2)
    ax1.set_title("Original Bar Chart Image")
    ax1.axis("off")

    # 2️⃣ 原始序列（像素级）
    ax2 = plt.subplot(3, 1, 2)
    ax2.plot(raw_series)
    ax2.axhline(0, linestyle="--")
    ax2.set_title(f"Extracted Raw Series (len={len(raw_series)})")
    ax2.set_ylabel("Value")

    # 3️⃣ 插值 / 重采样后
    ax3 = plt.subplot(3, 1, 3)
    ax3.plot(resampled_series, marker="o")
    ax3.axhline(0, linestyle="--")
    ax3.set_title(f"Resampled Series (len={len(resampled_series)})")
    ax3.set_ylabel("Value")
    ax3.set_xlabel("Bin Index")

    plt.tight_layout()
    plt.show()

def extract_series_from_image(img_path, bins=60):
    raw, _ = extract_bar_series(img_path)
    x_ignore_px = int(0.12 * 600)
    series = resample_series_interp_smooth(raw, bins)
    series = np.clip(series, -100, 100)
    return series


if __name__ == '__main__':
    img_path = '../data/sample1/main.png'
    img_path1 = '../data/sample1/fig1.png'
    raw = extract_bar_series(img_path)
    resampled = resample_series_interp_smooth(raw, target_len=60)

    visualize_extraction(img_path, raw, resampled)

