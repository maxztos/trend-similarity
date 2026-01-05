import numpy as np

from src.config import TREND_CONFIG
from src.dataloader import load_match_groups

def window_trend(
    contour_window,
    area_ratio=0.6,
    neutral_ratio_band=0.15,
):
    w = np.asarray(contour_window)

    pos_area = np.sum(w[w > 0])
    neg_area = np.abs(np.sum(w[w < 0]))
    total = pos_area + neg_area + 1e-6

    pos_ratio = pos_area / total
    neg_ratio = neg_area / total

    # 1️⃣ 方向主导（最重要）
    if pos_ratio >= area_ratio:
        return "+"
    if neg_ratio >= area_ratio:
        return "-"

    # 2️⃣ 均势：正负接近
    if abs(pos_ratio - neg_ratio) < neutral_ratio_band:
        return "0"

    # 3️⃣ 否则：按占优方向
    return "+" if pos_ratio > neg_ratio else "-"

# segment → 趋势时间轴
def segments_to_timeline(segments, total_len = 60):
    """
    segments: [{'trend': '+', 'start': i, 'end': j}, ...]
    total_len: 时间轴总长度
    """
    timeline = np.zeros(total_len, dtype=int)

    for seg in segments:
        v = 1 if seg["trend"] == "+" else -1
        timeline[seg["start"]:seg["end"]] = v
    # 补充后面数据
    last_seg = segments[-1]
    last_end = last_seg["end"]
    if last_end < total_len:
        last_v = 1 if last_seg["trend"] == "+" else -1
        timeline[last_end:total_len] = last_v

    return timeline
# 权重函数
def build_late_weight(total_len, start=0.7, end=1.3):
    """
    越靠后，权重越大
    start/end 控制强调程度
    """
    return np.linspace(start, end, total_len)

# 后半段加权趋势一致率
def weighted_trend_similarity(timeline_a, timeline_b, weights):
    """
    timeline_a / b: +1 / -1
    weights: 后半段加权
    """
    assert len(timeline_a) == len(timeline_b)

    same = (timeline_a == timeline_b).astype(float)

    return np.sum(same * weights) / np.sum(weights)

# 主趋势一致性
def dominant_trend(timeline):
    s = np.sum(timeline)
    return 1 if s > 0 else -1

# 最终相似度
def trend_similarity_pipeline(segA, segB, total_len = 60,
                              w_start=0.7, w_end=1.3,
                              dom_bonus=0.15):
    """
    返回 0~1 相似度
    """
    # 1. 展开时间轴
    a = segments_to_timeline(segA, total_len)
    b = segments_to_timeline(segB, total_len)

    # 2. 后半段权重
    weights = build_late_weight(total_len, w_start, w_end)

    # 3. 基础相似度
    base = weighted_trend_similarity(a, b, weights)

    # 4. 主趋势加成
    if dominant_trend(a) == dominant_trend(b):
        base += dom_bonus

    return min(base, 1.0)

def contour_to_variable_trends(
    contour,
    window_size=12,
    step=4,
    area_ratio=0.6,
    neutral_amp_ratio=0.25,
    min_len=6
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

    # 收尾 —— 尾段趋势强制保留一次
    if last_trend is not None:
        if stable_len >= min_len or len(trends) == 0 or last_trend != trends[-1]:
            trends.append(last_trend)

    return trends

def contour_to_trend_segments(contour, min_area=10.0):
    """
    只输出 + / -
    0 作为断点，不参与趋势
    小面积段自动并入相邻趋势

    返回:
    [
      {"trend": "+", "start": 0, "end": 18, "value": 12.3},
      {"trend": "-", "start": 18, "end": 39, "value": -8.7},
      {"trend": "+", "start": 39, "end": 52, "value": 21.5},
    ]
    """
    contour = np.asarray(contour)

    raw_segments = []

    def seg_info(seg):
        pos = np.sum(seg[seg > 0])
        neg = np.abs(np.sum(seg[seg < 0]))
        area = pos + neg
        trend = "+" if pos >= neg else "-"
        value = np.mean(seg) if len(seg) > 0 else 0.0
        return trend, area, value

    start = None
    prev_sign = 0

    # --- 切原始段（0 强制断） ---
    for i in range(len(contour)):
        s = np.sign(contour[i])

        if s == 0:
            if start is not None:
                seg = contour[start:i]
                trend, area, value = seg_info(seg)
                raw_segments.append({
                    "trend": trend,
                    "start": start,
                    "end": i,
                    "area": area,
                    "value": value
                })
                start = None
                prev_sign = 0
            continue

        if start is None:
            start = i
            prev_sign = s
            continue

        if s != prev_sign:
            seg = contour[start:i]
            trend, area, value = seg_info(seg)
            raw_segments.append({
                "trend": trend,
                "start": start,
                "end": i,
                "area": area,
                "value": value
            })
            start = i
            prev_sign = s

    # 收尾
    if start is not None:
        seg = contour[start:]
        trend, area, value = seg_info(seg)
        raw_segments.append({
            "trend": trend,
            "start": start,
            "end": len(contour),
            "area": area,
            "value": value
        })

    # --- 合并小面积段 ---
    merged = []

    for seg in raw_segments:
        if seg["area"] < min_area:
            if merged:
                merged[-1]["end"] = seg["end"]
                merged[-1]["value"] = np.mean(
                    contour[merged[-1]["start"]:merged[-1]["end"]]
                )
            continue

        if merged and merged[-1]["trend"] == seg["trend"]:
            merged[-1]["end"] = seg["end"]
            merged[-1]["value"] = np.mean(
                contour[merged[-1]["start"]:merged[-1]["end"]]
            )
        else:
            merged.append({
                "trend": seg["trend"],
                "start": seg["start"],
                "end": seg["end"],
                "value": seg["value"]
            })

    return merged

def contour_to_trends_by_zero_crossing(
    contour,
    min_area=10.0
):
    """
    根据 contour 与 x 轴相交点切分趋势
    小于 min_area 的段直接忽略

    返回: ['+', '-', '+', ...]
    """
    contour = np.asarray(contour)

    trends = []

    start = 0
    prev_sign = np.sign(contour[0])

    for i in range(1, len(contour)):
        curr_sign = np.sign(contour[i])

        # 忽略正好为 0 的点
        if curr_sign == 0:
            continue

        if prev_sign == 0:
            prev_sign = curr_sign
            continue

        # 发生过零
        if curr_sign != prev_sign:
            segment = contour[start:i]

            pos_area = np.sum(segment[segment > 0])
            neg_area = np.abs(np.sum(segment[segment < 0]))
            total_area = pos_area + neg_area

            # ⭐ 面积太小，直接忽略
            if total_area >= min_area:
                if pos_area > neg_area:
                    trends.append("+")
                elif neg_area > pos_area:
                    trends.append("-")

            start = i
            prev_sign = curr_sign

    # 最后一段
    segment = contour[start:]
    pos_area = np.sum(segment[segment > 0])
    neg_area = np.abs(np.sum(segment[segment < 0]))
    total_area = pos_area + neg_area

    if total_area >= min_area:
        if pos_area > neg_area:
            trends.append("+")
        elif neg_area > pos_area:
            trends.append("-")

    return trends

if __name__ == '__main__':
    excel_path = "../data/2.xlsx"
    data = load_match_groups(excel_path)

    match_id = "2025/05/18-29VS174-60"
    # print(data[match_id])

    # contour = extract_signed_area_contour(series, window=5)
    # print(trends)
