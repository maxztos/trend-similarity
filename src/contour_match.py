import numpy as np
from fastdtw import fastdtw
from src.dataloader import load_match_groups
from src.show import extract_signed_area_contour
from src.trend_segmentation import contour_to_trend_segments


def get_contour(match_data, window=3):
    # ===== Main =====
    main = match_data["main"]
    main_series = main["series"]

    main_contour = extract_signed_area_contour(
        main_series,
        window=window
    )

    # ===== Subs =====
    subs_output = []

    for sub in match_data["subs"]:
        sub_series = sub["series"]

        sub_contour = extract_signed_area_contour(
            sub_series,
            window=window
        )

        subs_output.append({
            "id": sub["id"],
            "contour": sub_contour
        })

    return {
        "main": {
            "id": main["id"],
            "contour": main_contour
        },
        "subs": subs_output
    }
# DTW 可以灵活拉伸 / 压缩其中一个序列
def calculate_dtw_distance(s1, s2):
    """
    计算两个序列之间的 DTW 距离
    距离越小，相似度越高
    """
    n, m = len(s1), len(s2)
    # 构建累积距离矩阵
    dtw_matrix = np.full((n + 1, m + 1), float('inf'))
    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # 计算当前点的欧氏距离（代价）
            cost = abs(s1[i-1] - s2[j-1])
            # 状态转移：取左、下、左下三个方向的最小值
            last_min = min(dtw_matrix[i-1, j],    # 插入
                           dtw_matrix[i, j-1],    # 删除
                           dtw_matrix[i-1, j-1])  # 匹配
            dtw_matrix[i, j] = cost + last_min

    return dtw_matrix[n, m]

def slide_and_match(main_timeline, sub_timeline):
    """
    在主时间轴上滑动子时间轴，计算每一个位置的匹配得分
    """
    matches = []
    n = len(main_timeline)
    m = len(sub_timeline)

    # 确保子序列不比主序列长
    if m > n:
        return [{"offset": 0, "score": 0}]

    # 滑动窗口：从左到右移动 sub_timeline
    for i in range(n - m + 1):
        # 提取当前主序列的窗口部分
        window = main_timeline[i: i + m]

        # 计算相似度得分 (这里使用简单的负相关距离转得分，或者余弦相似度)
        # 方案：计算两个向量的点积，并进行归一化
        dot_product = np.dot(window, sub_timeline)
        norm_window = np.linalg.norm(window)
        norm_sub = np.linalg.norm(sub_timeline)

        if norm_window == 0 or norm_sub == 0:
            score = 0
        else:
            # 余弦相似度：范围 [-1, 1]
            score = dot_product / (norm_window * norm_sub)

        matches.append({
            "offset": i,  # 起始索引位置
            "score": float(score)
        })

    return matches


# 模拟打印函数
def print_match_results(results):
    for res in results:
        print(
            f"子 ID: {res['sub_id']} | 最佳偏移: {res['best_match']['offset']} | 最高得分: {res['best_match']['score']:.4f}")


if __name__ == '__main__':

    excel_path = "../data/2.xlsx"
    data = load_match_groups(excel_path)
    match_id = "2025/05/05-2783VS51-60"
    match_data = data[match_id]
    data = get_contour(match_data)
    main_con = data["main"]["contour"]
    # print(main_con)
    results = []

    for sub in data["subs"]:
        # 1. 直接计算子序列与主序列的 DTW 距离
        dist = calculate_dtw_distance(main_con, sub["contour"])

        # 2. 计算归一化距离（可选，消除长度影响）
        # 归一化得分 = 距离 / (主序列长度 + 子序列长度)
        norm_dist = dist / (len(main_con) + len(sub["contour"]))

        results.append({
            "sub_id": sub["id"],
            "distance": dist,
            "norm_distance": norm_dist
        })

    # 3. 按距离从小到大排序（最相似的在前）
    results.sort(key=lambda x: x["norm_distance"])

    # 打印结果
    for res in results:
        print(f"ID: {res['sub_id']} | DTW 距离: {res['distance']:.2f} | 归一化距离: {res['norm_distance']:.4f}")