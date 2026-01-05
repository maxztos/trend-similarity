import numpy as np
import pandas as pd

from src.dataloader import load_match_groups
from src.show import extract_signed_area_contour, visualize_match_with_signed_contour
from src.trend_segmentation import contour_to_variable_trends, contour_to_trend_segments, segments_to_timeline, \
    trend_similarity_pipeline


def trend_match_score(a, b):
    if a == b:
        return 1.0
    if '0' in (a, b):
        return 0.5
    return 0.0

def contour_similarity_l1(a, b, eps=1e-6):
    a = np.asarray(a)
    b = np.asarray(b)

    target_len = max(len(a), len(b))
    a = resample_contour(a, target_len)
    b = resample_contour(b, target_len)

    diff = np.abs(a - b).mean()
    scale = np.maximum(np.abs(a).mean(), np.abs(b).mean())

    if scale < eps:
        return 1.0

    sim = 1.0 - diff / (scale + eps)
    return float(np.clip(sim, 0.0, 1.0))

def split_contour_by_energy(contour, ratio=0.5, min_len=5):
    """
    根据轮廓“绝对面积能量”自适应切分前/后段

    ratio   : 后段起始所占累计能量比例（0.5 = 后半能量）
    min_len : 防止切分过短
    """
    contour = np.asarray(contour, dtype=float)
    n = len(contour)

    if n < min_len * 2:
        return contour, np.array([])

    energy = np.abs(contour)
    total_energy = energy.sum()

    if total_energy < 1e-6:
        split_idx = n // 2
    else:
        cum_energy = np.cumsum(energy)
        split_idx = int(np.searchsorted(cum_energy, total_energy * ratio))

    # 安全保护
    split_idx = max(min_len, min(split_idx, n - min_len))

    return contour[:split_idx], contour[split_idx:]
def split_weighted_contour_similarity(
    main_contour,
    sub_contour,
    energy_ratio=0.5,
    w_front=0.3,
    w_back=0.7
):
    # 1️⃣ 对齐长度
    target_len = max(len(main_contour), len(sub_contour))
    main = resample_contour(main_contour, target_len)
    sub = resample_contour(sub_contour, target_len)

    # 2️⃣ 能量自适应切分
    main_front, main_back = split_contour_by_energy(
        main, ratio=energy_ratio
    )
    sub_front, sub_back = split_contour_by_energy(
        sub, ratio=energy_ratio
    )

    # 3️⃣ 相似度计算（安全兜底）
    sim_front = (
        contour_similarity_l1(main_front, sub_front)
        if len(main_front) > 0 and len(sub_front) > 0
        else 0.0
    )

    sim_back = (
        contour_similarity_l1(main_back, sub_back)
        if len(main_back) > 0 and len(sub_back) > 0
        else 0.0
    )

    score = w_front * sim_front + w_back * sim_back

    return {
        "sim_front": float(sim_front),
        "sim_back": float(sim_back),
        "score": float(score)
    }

def backward_trend_similarity(
    main_trend,
    sub_trend,
    max_compare=None,
    decay=0.9
):
    """
    从后向前进行趋势匹配
    - 不改变原始趋势长度
    - 越靠后权重越大
    """
    i = len(main_trend) - 1
    j = len(sub_trend) - 1

    scores = []
    weights = []

    w = 1.0

    while i >= 0 and j >= 0:
        score = trend_match_score(main_trend[i], sub_trend[j])
        scores.append(score)
        weights.append(w)

        w *= decay
        i -= 1
        j -= 1

        if max_compare and len(scores) >= max_compare:
            break

    if not scores:
        return 0.0

    weighted_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
    return float(weighted_score)

def score_matches_by_trend_topk(
    data: dict,
    window: int = 3,
    eps: float = 1e-3,
    topk: int = 5,
    decay: float = 0.9,
    max_compare: int | None = None
):
    """
    对 data 中每个 match：
    - 计算 main / sub 的趋势序列
    - 使用后向趋势匹配打分
    - 返回 topk 的 subs
    """

    results = {}

    for match_id1, match_data in data.items():
        main = match_data["main"]
        subs = match_data.get("subs", [])

        # 1️⃣ main 趋势
        main_contour = extract_signed_area_contour(
            main["series"],
            window=window
        )
        main_trend = contour_to_variable_trends(
            main_contour,
            # eps=eps
        )

        scored_subs = []

        for sub in subs:
            # 2️⃣ sub 趋势
            sub_contour = extract_signed_area_contour(
                sub["series"],
                window=window
            )
            sub_trend = contour_to_variable_trends(
                sub_contour,
                # eps=eps
            )

            # 3️⃣ 后向趋势匹配评分（重点看后半段）
            score = backward_trend_similarity(
                main_trend,
                sub_trend,
                decay=decay,
                max_compare=max_compare
            )

            scored_subs.append({
                "id": sub["id"],
                "score": float(score),
                "trend": sub_trend
            })

        # 4️⃣ 按分数排序取 TopK
        scored_subs.sort(key=lambda x: x["score"], reverse=True)

        results[match_id1] = {
            "main_id": main["id"],
            "main_trend": main_trend,
            "top_subs": scored_subs[:topk]
        }

    return results
def score_matches_by_contour(
    matches,
    window=15,
    split_ratio=0.5
):
    for match_id, match_data in matches.items():
        main_series = match_data["main"]["series"]

        main_contour = extract_signed_area_contour(
            main_series,
            window=window
        )

        match_data["main"]["contour"] = main_contour

        for sub in match_data["subs"]:
            sub_series = sub["series"]

            sub_contour = extract_signed_area_contour(
                sub_series,
                window=window
            )

            result = split_weighted_contour_similarity(
                main_contour,
                sub_contour
            )

            sub["contour"] = sub_contour
            sub["score"] = result["score"]
            sub["sim_front"] = result["sim_front"]
            sub["sim_back"] = result["sim_back"]

    return matches
def sort_subs_by_score(matches, descending=True):
    for match_data in matches.values():
        match_data["subs"].sort(
            key=lambda x: x["score"],
            reverse=descending
        )
def resample_contour(contour, target_len):
    contour = np.asarray(contour)

    if len(contour) == target_len:
        return contour

    x_old = np.linspace(0, 1, len(contour))
    x_new = np.linspace(0, 1, target_len)

    return np.interp(x_new, x_old, contour)
def export_contour_scores_to_excel(
    excel_path: str,
    output_path: str,
    window: int = 15,
    split_ratio: float = 0.5,
    w_front: float = 0.3,
    w_back: float = 0.7,
    verbose: bool = True
):
    """
    使用轮廓分段相似度，对所有比赛对打分并导出 Excel
    """

    # 1️⃣ 加载原始数据
    matches = load_match_groups(excel_path)

    # 2️⃣ 计算轮廓 + 评分
    score_matches_by_contour(
        matches,
        window=window
    )

    # ⭐ 关键一步：每一组内部按 score 降序排序
    sort_subs_by_score(matches, descending=True)

    # 3️⃣ 展平写表（此时顺序已经是对的）
    rows = []

    for match_id, match_data in matches.items():
        main = match_data.get("main")
        if main is None:
            continue
        rank = 1
        for sub in match_data.get("subs", []):
            rows.append({
                "match_id1": match_id,
                "main_id": main["id"],
                "sub_id": sub["id"],
                "sim_front": float(sub["sim_front"]),
                "sim_back": float(sub["sim_back"]),
                "score": float(sub["score"]),
                "rank": rank
            })
            rank += 1

    df = pd.DataFrame(rows)

    if verbose:
        print("📊 评分条目数：", len(df))
        print(df.head())

    # 4️⃣ 导出 Excel
    df.to_excel(output_path, index=False)

    if verbose:
        print(f"✅ 已成功导出到 {output_path}")

    return df

def print_top5_results(results):
    for match_id, info in results.items():
        print("=" * 80)
        print(f"MAIN MATCH: {match_id}")
        print(f"MAIN ID   : {info['main_id']}")
        # print(f"MAIN TREND: {info['main_trend']}")
        print("-" * 80)

        for i, sub in enumerate(info["top_subs"], start=1):
            print(
                f"{i:>2}. sub_id={sub['id']} | "
                f"score={sub['score']:.4f} | "
                # f"trend={sub['trend']}"
            )
        print()

# if __name__ == '__main__':
#     excel_path = "../data/2.xlsx"
#     output_path = "../data/contour_scores.xlsx"
#
#     export_contour_scores_to_excel(
#         excel_path=excel_path,
#         output_path=output_path,
#         window=3,
#         split_ratio=0.5,
#         w_front=0.3,
#         w_back=0.7
#     )

def score_matches_by_segments(
    data: dict,
    window: int = 3,
    topk: int = 5,
):
    """
    对data中每个match:
    提取轮廓-序列化趋势-匹配打分
    :param data:
    :param topk:
    :return:
    """
    result = {}
    for match_id, match_data in data.items():
        main = match_data["main"]
        subs = match_data.get("subs", [])
        # main趋势
        main_contour = extract_signed_area_contour(
            main["series"],
            window=window
        )
        main_seg = contour_to_trend_segments(main_contour)


        # print(segments_to_timeline(contour_to_trend_segments(main_contour),60))

        scored_subs = []
        for sub in subs:
            sub_contour = extract_signed_area_contour(
                sub["series"],
                window=window
            )
            sub_seg = contour_to_trend_segments(sub_contour)
            # print(segments_to_timeline(contour_to_trend_segments(sub_contour),60))

            score = trend_similarity_pipeline(main_seg, sub_seg)

            scored_subs.append({
                "id": sub["id"],
                "score": float(score)
            })
            scored_subs.sort(key=lambda x: x["score"], reverse=True)

            result[match_id] = {
                "main_id": main["id"],
                "top_subs": scored_subs[:topk],
            }
    return result


if __name__ == "__main__":
    excel_path = "../data/2.xlsx"
    data = load_match_groups(excel_path)

    top5_results = score_matches_by_trend_topk(
        data,
        window=3,
        topk=5
    )
    result = score_matches_by_segments(data)
    print_top5_results(result)
    # 看某一场
    # match_id = "2025/05/05-2783VS51-60"
    # print_topk_results_compatible(top5_results)
    # print_top5_results(top5_results)
