import numpy as np

from src.dataloader import load_match_groups
from src.show import extract_signed_area_contour
from src.trend_segmentation import contour_to_trend_segments, segments_to_timeline
from scipy.stats import pearsonr


def calculate_trend_similarity(main_arr, sub_arr):
    """
    计算两个等长数组的趋势相似度。

    指标 1: 符号匹配率 (Sign Accuracy) - 最重要，判断 +/- 是否一致
    指标 2: 相关系数 (Correlation) - 判断波形走势是否一致
    """
    # 移除 NaN (由滑动产生的空值)
    valid_mask = ~np.isnan(main_arr) & ~np.isnan(sub_arr)
    m = main_arr[valid_mask]
    s = sub_arr[valid_mask]

    if len(m) == 0: return 0, 0  # 无重叠区域

    # 1. 计算符号匹配率 (忽略 0 的情况，或者把 0 视为一种状态)
    # np.sign 返回 -1, 0, 1
    same_sign_count = np.sum(np.sign(m) == np.sign(s))
    sign_accuracy = same_sign_count / len(m)

    # 2. 计算皮尔逊相关系数 (用于辅助判断波形)
    # 如果标准差为0 (直线)，相关系数无定义，设为0
    if np.std(m) == 0 or np.std(s) == 0:
        correlation = 0
    else:
        correlation, _ = pearsonr(m, s)

    # return sign_accuracy, correlation
    final_score = 0.7 * sign_accuracy + 0.3 * correlation
    return final_score

def get_timelines(match_data , window=3):
    # excel_path = "../data/2.xlsx"
    # data = load_match_groups(excel_path)
    # match_data = data[match_id]

    # ===== Main =====
    main = match_data["main"]
    main_series = main["series"]

    main_contour = extract_signed_area_contour(
        main_series,
        window=window
    )
    main_seg = contour_to_trend_segments(main_contour)
    main_timeline = segments_to_timeline(main_seg)

    # ===== Subs =====
    subs_output = []

    for sub in match_data["subs"]:
        sub_series = sub["series"]

        sub_contour = extract_signed_area_contour(
            sub_series,
            window=window
        )
        sub_seg = contour_to_trend_segments(sub_contour)
        sub_timeline = segments_to_timeline(sub_seg)

        subs_output.append({
            "id": sub["id"],
            "timeline": sub_timeline
        })

    return {
        "main": {
            "id": main["id"],
            "timeline": main_timeline
        },
        "subs": subs_output
    }


def weighted_cosine_similarity(a, b, tail_weight=1.5):
    """
    a, b: 已对齐、等长的一维数组
    后半段加权（强调趋势结尾）
    """

    if len(a) == 0 or len(b) == 0:
        return 0.0

    n = len(a)
    weights = np.ones(n)
    weights[n // 2 :] *= tail_weight

    a_w = a * weights
    b_w = b * weights

    denom = np.linalg.norm(a_w) * np.linalg.norm(b_w)
    if denom == 0:
        return 0.0

    return float(np.dot(a_w, b_w) / denom)
def slide_and_match(
    main_tl,
    sub_tl,
    max_slide=2,
    min_overlap=10
):
    """
    sub 在 main 上滑动（中心对齐）
    """
    results = []

    main_tl = np.asarray(main_tl)
    sub_tl = np.asarray(sub_tl)

    Lm = len(main_tl)
    Ls = len(sub_tl)

    # === 中心对齐基准 ===
    base = (Lm - Ls) // 2

    for shift in range(-max_slide, max_slide + 1):
        # sub 在 main 上的位置
        m_start = base + shift
        m_end   = m_start + Ls

        # 裁剪到 main 范围
        mm_start = max(0, m_start)
        mm_end   = min(Lm, m_end)

        # 对应 sub 的裁剪
        ss_start = mm_start - m_start
        ss_end   = ss_start + (mm_end - mm_start)

        if mm_end - mm_start < min_overlap:
            continue

        a = main_tl[mm_start:mm_end]
        b = sub_tl[ss_start:ss_end]

        score = calculate_trend_similarity(a, b)
        # score = calculate_visual_score(a, b)


        results.append({
            "direction": "right" if shift > 0 else "left" if shift < 0 else "center",
            "shift": shift,
            "score": score
        })

    return results

def print_match_results(results, score_fmt="{:.3f}"):
    """
    批量打印匹配结果（按 score 降序）
    results: [
        {
          'sub_id': str,
          'best_match': {'score': float, ...}
        },
        ...
    ]
    """

    # 按 score 降序排序（没有 score 的排最后）
    results_sorted = sorted(
        results,
        key=lambda r: r.get("best_match", {}).get("score", -1),
        reverse=True
    )

    for r in results_sorted:
        sub_id = r.get("sub_id", "UNKNOWN")
        score = r.get("best_match", {}).get("score", None)

        if score is None:
            print(f"{sub_id} | score: N/A")
        else:
            print(f"{sub_id} | score: {score_fmt.format(score)}")

if __name__ == '__main__':

    data = get_timelines("2025/05/15-64VS54-60")
    main_tl = data["main"]["timeline"]
    # print(main_tl)
    # print(data["subs"][-2]["timeline"])
    # print(data["subs"][-1]["timeline"])
    results = []

    for sub in data["subs"]:
        matches = slide_and_match(main_tl, sub["timeline"])
        best = max(matches, key=lambda x: x["score"])

        results.append({
            "sub_id": sub["id"],
            "best_match": best
        })

    print_match_results(results)