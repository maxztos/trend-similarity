import numpy as np

from src.dataloader import load_match_groups

def window_trend(
    contour_window,
    area_ratio=0.6,
    neutral_amp_ratio=0.15,
):
    """
    判断窗口趋势：
    + 正势
    - 负势
    0 均势
    """
    contour_window = np.asarray(contour_window)

    mean_abs = np.mean(np.abs(contour_window))
    mean_signed = np.mean(contour_window)

    pos_area = contour_window[contour_window > 0].sum()
    neg_area = -contour_window[contour_window < 0].sum()
    total_area = pos_area + neg_area + 1e-6

    # ===== 1️⃣ 均势优先判断 =====
    # 幅度小 + 均值接近 0
    if mean_abs < neutral_amp_ratio * total_area / len(contour_window):
        return '0'

    # ===== 2️⃣ 正 / 负势 =====
    if pos_area / total_area >= area_ratio and mean_signed > 0:
        return '+'

    if neg_area / total_area >= area_ratio and mean_signed < 0:
        return '-'

    # ===== 3️⃣ 兜底 =====
    return '0'

def contour_to_variable_trends(
    contour,
    window_size=15,
    step=3,
    area_ratio=0.6,
    min_len=5
):
    """
    根据轮廓生成不定长趋势段
    返回: ['+', '+', '0', '-', '-', '+'] （已合并）
    """
    contour = np.asarray(contour)

    trends = []
    last_trend = None
    stable_len = 0

    for i in range(0, len(contour) - window_size + 1, step):
        window = contour[i:i + window_size]
        t = window_trend(window, area_ratio)

        if t == last_trend:
            stable_len += step
        else:
            if last_trend is not None and stable_len >= min_len:
                trends.append(last_trend)
            last_trend = t
            stable_len = step

    if last_trend is not None and stable_len >= min_len:
        trends.append(last_trend)

    return trends

def contour_to_trend_sequence(
    contour,
    window_size=5,
    stride=5,
    area_ratio=0.6
):
    contour = np.asarray(contour)
    trends = []

    for start in range(0, len(contour) - window_size + 1, stride):
        window = contour[start:start + window_size]

        pos = window[window > 0].sum()
        neg = -window[window < 0].sum()
        total = pos + neg + 1e-6

        if pos / total >= area_ratio:
            trends.append('+')
        elif neg / total >= area_ratio:
            trends.append('-')
        else:
            trends.append('0')

    return trends


if __name__ == '__main__':
    excel_path = "../data/2.xlsx"
    data = load_match_groups(excel_path)

    match_id = "2025/05/05-2783VS51-60"
    # print(data[match_id])

    # contour = extract_signed_area_contour(series, window=5)
    # print(trends)
