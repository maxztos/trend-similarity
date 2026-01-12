import numpy as np

from src.scoring import series_stats, apply_penalties
from src.utils.dataloader import load_match_groups
from src.show import extract_signed_area_contour

def get_contour(match_data, window=3):
    # ===== Main =====
    main = match_data["main"]
    main_series = main["series"]
    main_nums = main["nums"]

    main_contour = extract_signed_area_contour(
        main_series,
        window=window
    )

    # ===== Subs =====
    subs_output = []

    for sub in match_data["subs"]:
        sub_series = sub["series"]
        sub_nums = sub["nums"]

        sub_contour = extract_signed_area_contour(
            sub_series,
            window=window
        )

        subs_output.append({
            "id": sub["id"],
            "series": sub_series,
            "contour": sub_contour,
            "nums": sub_nums
        })

    return {
        "main": {
            "id": main["id"],
            "series": main_series,
            "contour": main_contour,
            "nums": main_nums
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

def dtw_to_score(avg_dist, scale=20.0):
    """
    avg_dist: DTW 每步平均距离
    scale: 人工可调，≈“能接受的差异”
    """
    score = 100 * np.exp(-avg_dist / scale)
    return float(score)

# 输出匹配后的评分结果，计算得分
def match_results(excel_path):

    # 1. 加载所有比赛分组数据
    data = load_match_groups(excel_path)

    all_results = []          # 存所有 sub 的评分结果
    match_count = 0           # match_id 数量
    total_sub_count = 0       # sub 总数

    # 2. 遍历所有 match_id
    for match_id, match_data in data.items():
        try:
            match_count += 1

            if match_data["main"] is None:
                continue

            main_nums = match_data["main"]["nums"]

            # 获取轮廓数据
            contour_data = get_contour(match_data)

            main_con = contour_data["main"]["contour"]
            main_series = contour_data["main"]["series"]
            main_series_stats = series_stats(main_series)

            # 3. 遍历所有子序列（不选 best，全记录）
            for sub in contour_data["subs"]:
                total_sub_count += 1

                dist = calculate_dtw_distance(main_con, sub["contour"])
                avg_dist = dist / max(len(main_con), len(sub["contour"]))
                base_score = dtw_to_score(avg_dist)

                sub_series_stats = series_stats(sub["series"])

                final_score, penalties, total_penalty = apply_penalties(
                    base_score,
                    main_series_stats,
                    sub_series_stats,
                    main_con,
                    sub["contour"],
                )

                all_results.append({
                    "match_id": match_id,
                    "sub_id": sub["id"],
                    "main_nums": main_nums,
                    "sub_nums": sub["nums"],
                    # "distance": dist,
                    # "avg_distance": avg_dist,
                    # "base_score": base_score,
                    "final_score": final_score,
                    # "total_penalty": total_penalty,
                    # "penalties": penalties
                })

        except Exception:
            # 单个 match 出问题，不影响整体
            continue

    # 4. 只输出总体统计
    print(f"Processed match count: {match_count}")
    print(f"Total sub entries processed: {total_sub_count}")

    return {
        "results": all_results,
        "stats": {
            "match_count": match_count,
            "total_sub_count": total_sub_count
        }
    }


if __name__ == '__main__':

    excel_path = "../data/2.xlsx"
    data = load_match_groups(excel_path)
    match_id = "2025/05/10-161VS211-61"
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

        avg_dist = dist / max(len(main_con), len(sub["contour"]))
        score = dtw_to_score(avg_dist)

        results.append({
            "sub_id": sub["id"],
            "distance": dist,
            "norm_distance": norm_dist,
            "score": score
        })

    # 3. 按距离从小到大排序（最相似的在前）
    results.sort(key=lambda x: x["norm_distance"])

    # 打印结果
    for res in results:
        print(f"ID: {res['sub_id']} | 距离: {res['distance']:.2f} | 归一化距离: {res['norm_distance']:.4f} | 得分: {res['score']:.2f}")