import numpy as np
from tslearn.metrics import soft_dtw
from src.preprocess import preprocess_data
from src.scoring import series_stats, apply_penalties
from src.utils.dataloader import load_match_groups
from src.show import extract_signed_area_contour
from tslearn.metrics import soft_dtw


def soft_dtw_tail_score(
    main,
    sub,
    tail_ratio=0.7,
    gamma=0.5,
    temperature=1.0,
    eps=1e-6,
):
    """
    Soft-DTW 相似度评分（只看后半段走势）

    参数
    ----
    main, sub : dict
        {
            "series": np.ndarray  # 一维时间序列
        }

    tail_ratio : float
        只取后多少比例（默认 30%）

    gamma : float
        Soft-DTW 平滑系数，越小越严格（推荐 0.3 ~ 1.0）

    temperature : float
        score 衰减温度，越小分数掉得越快

    返回
    ----
    score : float
        0~1，越大越相似
    dist : float
        Soft-DTW 原始距离（调试用）
    """

    x = main
    y = sub

    # ---------- 1. 长度保护 ----------
    L = min(len(x), len(y))
    if L < 5:
        # 太短了，工程上一般不直接杀，给个中性分
        return 0.5, None

    tail_len = max(3, int(np.ceil(L * tail_ratio)))

    x_tail = x[-tail_len:]
    y_tail = y[-tail_len:]

    # ---------- 2. 去均值（去绝对高度） ----------
    x_tail = x_tail - np.mean(x_tail)
    y_tail = y_tail - np.mean(y_tail)

    # ---------- 3. 标准化（去振幅） ----------
    x_std = np.std(x_tail)
    y_std = np.std(y_tail)

    if x_std < eps or y_std < eps:
        # 几乎是平线，走势信息不足
        return 0.5, None

    x_tail = x_tail / (x_std + eps)
    y_tail = y_tail / (y_std + eps)

    # ---------- 4. Soft-DTW 距离 ----------
    # tslearn 需要 (T, 1) 形状
    x_tail = x_tail.reshape(-1, 1)
    y_tail = y_tail.reshape(-1, 1)

    dist = soft_dtw(x_tail, y_tail, gamma=gamma)

    # ---------- 5. 距离 → soft score ----------
    # 指数映射，连续、可解释
    score = np.exp(-dist / temperature)

    return float(score), float(dist)
    # return float(score)




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

def derivative(x):
    """
    Keogh-style derivative for DDTW
    """
    x = np.asarray(x, dtype=np.float32)
    if len(x) < 3:
        return np.zeros_like(x)

    d = np.zeros_like(x)
    for i in range(1, len(x) - 1):
        d[i] = (
            (x[i] - x[i - 1]) +
            (x[i + 1] - x[i - 1]) / 2
        ) / 2

    return d

def sliding_tail_ddtw(
    main_series,
    sub_series,
    tail_ratio_main=0.4,
    sub_ratio_range=(0.3, 0.5),
    sub_ratio_step=0.02,
):
    """
    Right-aligned sliding-tail DDTW
    - 右端点固定
    - 只滑动左端点
    """

    main = np.asarray(main_series, dtype=np.float32)
    sub  = np.asarray(sub_series, dtype=np.float32)

    N, M = len(main), len(sub)
    if N < 10 or M < 10:
        return np.inf, None

    # ---------- main 尾部（固定） ----------
    main_len = int(N * tail_ratio_main)
    if main_len < 5:
        return np.inf, None

    main_tail = main[-main_len:]
    main_d = derivative(main_tail)

    best_dist = np.inf
    best_cfg = None

    # ---------- sub：右对齐，左端滑动 ----------
    for r in np.arange(sub_ratio_range[0], sub_ratio_range[1] + 1e-6, sub_ratio_step):
        sub_len = int(M * r)

        if sub_len < 5:
            continue

        sub_tail = sub[-sub_len:]   # 右端点固定
        sub_d = derivative(sub_tail)

        dist = calculate_dtw_distance(main_d, sub_d)

        if dist < best_dist:
            best_dist = dist
            best_cfg = {
                "main_len": main_len,
                "sub_len": sub_len,
                "sub_ratio": round(r, 3),
            }

    return best_dist, best_cfg

# 输出匹配后的评分结果，计算得分
def match_results(excel_path):

    # 1. 加载所有比赛分组数据
    raw_data = load_match_groups(excel_path)
    # data = load_match_groups(excel_path)
    # 对源数据预处理
    data, stats = preprocess_data(raw_data)

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
                ddist, _ = sliding_tail_ddtw(main_con, sub["contour"])
                # dist = segmented_dtw_w(main_con, sub["contour"])

                avg_dist = dist / max(len(main_con), len(sub["contour"]))

                base_score = dtw_to_score(avg_dist)
                # base_score = dtw_to_score(dist)
                sub_series_stats = series_stats(sub["series"])

                soft_dtw, sdist= soft_dtw_tail_score(main_con, sub["series"])
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
                    "soft_dtw": soft_dtw,
                    "distance": dist,
                    "d_distance": ddist,
                    # "avg_distance": avg_dist,
                    # "base_score": base_score,
                    "final_score": final_score,  #直接分数
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

from collections import defaultdict
import numpy as np

def print_match_results(
    match_results,
    score_threshold=None,
    sub_id_list=None,
):
    """
    人类可读方式打印 match_results

    参数：
    - match_results : list[dict]
    - score_threshold : 只打印 final_score >= threshold 的（可选）
    - sub_id_list : 只打印指定 sub_id 的结果（可选，list / set）
    """

    grouped = defaultdict(list)

    # ---------- 预处理 ----------
    sub_id_set = set(sub_id_list) if sub_id_list is not None else None

    # ---------- 分组 + 过滤 ----------
    for r in match_results:

        # sub_id 过滤
        if sub_id_set is not None and r.get("sub_id") not in sub_id_set:
            continue

        # score 过滤
        if score_threshold is not None:
            if r.get("final_score") is None or r["final_score"] < score_threshold:
                continue

        grouped[r["match_id"]].append(r)

    # ---------- 打印 ----------
    for match_id, items in grouped.items():

        # ⭐⭐⭐ 按 dist 升序排序 ⭐⭐⭐
        items = sorted(
            items,
            key=lambda r: (
                r.get("distance") is None,
                r.get("distance", float("inf"))
            )
        )

        print("\n" + "=" * 70)
        print(f"MATCH {match_id}  (subs: {len(items)})")
        print("=" * 70)

        for r in items:
            score = r.get("final_score")
            soft = r.get("soft_dtw")
            dist = r.get("distance")
            d_dist = r.get("d_distance")

            score_str = f"{score:.2f}" if score is not None else "None"
            soft_str  = f"{soft:.3f}" if soft is not None else "None"
            dist_str  = f"{dist:.3f}" if dist is not None else "None"
            d_dist_str = f"{d_dist:.3f}" if d_dist is not None else "None"

            print(
                f"{r['sub_id']:<30} "
                f"score={score_str:<7} "
                f"soft_dtw={soft_str:<7} "
                f"dist={dist_str:<7} "
                f"d_dist={d_dist_str:<7}"
                # f"sum={d_dist+d_dist:<5}"
            )

        print()

if __name__ == "__main__":
    excel_path = "../data/2n.xlsx"
    data = match_results(excel_path)
    sub_id_list = [
        "2025/04/27-109VS796",
        "2022/03/12-832VS70",
        "2025/04/27-276VS72",
        "2024/01/20-84VS76",
        "2025/04/28-76VS78",
        "2024/01/05-71VS278",
        "2023/02/19-64VS67",
        "2025/05/03-13VS183",
        "2022/10/28-309VS246",
        "2025/05/03-78VS86",
        "2022/09/17-1501VS271",
        "2025/05/04-84VS73",
        "2023/04/02-29VS18",
        "2025/05/05-2783VS51",
        "2023/12/07-2332VS148",
        "2023/09/02-54VS64",
        "2022/02/19-41VS109",
        "2025/05/10-161VS211",
        "2022/02/26-143VS71",
        "2025/05/10-246VS309",
        "2023/10/29-269VS86",
        "2025/05/11-174VS14",
        "2023/05/20-300VS76",
        "2025/05/11-76VS79",
        "2023/10/07-51VS55",
        "2025/05/11-86VS269",
        "2023/10/28-15VS189",
        "2025/05/13-58VS2783",
        "2023/05/13-143VS300",
        "2025/05/15-64VS54",
        "2022/10/09-47VS50",
        "2025/05/18-29VS174",
        "2022/05/06-278VS87",
        "2025/05/18-55VS53",
        "2022/10/09-41VS796",
        "2025/05/25-165VS29",
        "2024/05/10-60VS2783"
    ]
    print_match_results(data["results"],0)


# if __name__ == '__main__':
#
#     excel_path = "../data/2.xlsx"
#     data = load_match_groups(excel_path)
#     match_id = "2025/05/10-161VS211-61"
#     match_data = data[match_id]
#     data = get_contour(match_data)
#     main_con = data["main"]["contour"]
#     # print(main_con)
#     results = []
#
#     for sub in data["subs"]:
#         # 1. 直接计算子序列与主序列的 DTW 距离
#         dist = calculate_dtw_distance(main_con, sub["contour"])
#
#         # 2. 计算归一化距离（可选，消除长度影响）
#         # 归一化得分 = 距离 / (主序列长度 + 子序列长度)
#         norm_dist = dist / (len(main_con) + len(sub["contour"]))
#
#         avg_dist = dist / max(len(main_con), len(sub["contour"]))
#         score = dtw_to_score(avg_dist)
#
#         results.append({
#             "sub_id": sub["id"],
#             "distance": dist,
#             "norm_distance": norm_dist,
#             "score": score
#         })
#
#     # 3. 按距离从小到大排序（最相似的在前）
#     results.sort(key=lambda x: x["norm_distance"])
#
#     # 打印结果
#     for res in results:
#         print(f"ID: {res['sub_id']} | 距离: {res['distance']:.2f} | 归一化距离: {res['norm_distance']:.4f} | 得分: {res['score']:.2f}")