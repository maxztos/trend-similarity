import numpy as np
import pandas as pd

from src.dataloader import load_match_groups
from src.show import extract_signed_area_contour, visualize_match_with_signed_contour


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

def split_weighted_contour_similarity(
    main_contour,
    sub_contour,
    split_ratio=0.5,
    w_front=0.3,
    w_back=0.7
):
    # 对齐长度
    target_len = max(len(main_contour), len(sub_contour))
    main = resample_contour(main_contour, target_len)
    sub = resample_contour(sub_contour, target_len)

    split_idx = int(target_len * split_ratio)

    sim_front = contour_similarity_l1(
        main[:split_idx],
        sub[:split_idx]
    )

    sim_back = contour_similarity_l1(
        main[split_idx:],
        sub[split_idx:]
    )

    score = w_front * sim_front + w_back * sim_back

    return {
        "sim_front": sim_front,
        "sim_back": sim_back,
        "score": score
    }

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
                sub_contour,
                split_ratio=split_ratio
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

if __name__ == "__main__":
    excel_path = "../data/2.xlsx"
    data = load_match_groups(excel_path)

    data = score_matches_by_contour(
        data,
        window=3
    )

    sort_subs_by_score(data)

    match_id = "2025/05/18-55VS53-60"
    visualize_match_with_signed_contour(
        data[match_id],
        window=3
    )
