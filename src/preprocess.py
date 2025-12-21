# 预处理(归一化/对齐)
# 1.灰度化
#
# 2.二值化（柱子 vs 背景）
#
# 3.去掉坐标轴、网格线
#
# 4.找出柱子所在区域（ROI）

def extract_bar_series(img_path, bins=60, y_min=-100, y_max=100):
    import cv2, numpy as np

    img = cv2.imread(img_path)
    img = cv2.resize(img, (800, 600))  # 强烈推荐

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    h, w = binary.shape

    # 比例定义（分辨率无关）
    y_top = int(0.1 * h)
    y_bottom = int(0.9 * h)
    y_zero = int(0.5 * h)

    pixels_per_unit = (y_bottom - y_top) / (y_max - y_min)

    series = []
    bin_width = w / bins

    for i in range(bins):
        xs = int(i * bin_width)
        xe = int((i + 1) * bin_width)
        col = binary[y_top:y_bottom, xs:xe]

        ys, _ = np.where(col > 0)

        if len(ys) == 0:
            series.append(0.0)
            continue

        y_pixel = y_top + ys.mean()
        value = (y_zero - y_pixel) / pixels_per_unit
        series.append(value)

    return series

if __name__ == '__main__':
    img_path = '../data/sample1/main.png'
    print(extract_bar_series(img_path))
