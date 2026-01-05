import numpy as np
import itertools

from src.show import plot_trend_segments_bar
from src.trend_segmentation import contour_to_variable_trends, contour_to_trend_segments, segments_to_timeline

contour = np.array([
 -0.63877839, -3.65605799, -0.2283914,  5.71128901,  8.81330704,
  2.93971508,  1.56721206,  4.81516771, 16.35244655, 25.01348133,
  28.78660366, 19.21008192, 20.19047365, 26.95237741, 36.19047265,
  46.47618811, 54.66666449, 59.6666663,  59.38095202, 54.38095147,
  45.33333297, 40.66666612, 32.28571202, 37.9523795,  44.4285697,
  55.61904653, 67.61904771, 78.7142859,  75.71428571, 69.00000036,
  57.33333279, 47.38095147, 37.66666549, 33.42856943, 30.04761669,
  25.3809502,  20.38094988, 16.85714095, 15.97994832, 14.22054983,
  12.04183404,  8.02621855,  2.97223806,  2.68941539,  9.6539419,
  17.28937637, 22.49188757, 18.9591823,  12.75744657,  5.61552775,
 -3.50023437, -12.79635233, -22.649963, -20.64018324, -10.22994098,
  0.69904811,  7.2491775,   8.37451384, 17.33485499, 42.3370533
])

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
    trends = contour_to_variable_trends(
        contour,
        window_size=12,
        step=4,
        area_ratio=0.7,
        neutral_amp_ratio=0.15,
        min_len=6
    )
    result = contour_to_trends_by_zero_crossing(contour)
    segments = contour_to_trend_segments(contour, min_area=10)
    # print(segments_to_timeline(segments,60))
    plot_trend_segments_bar(segments)
    for s in segments:
        print(s)
    # print(result)