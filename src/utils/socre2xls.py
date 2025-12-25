import pandas as pd

from src.dataloader import load_match_groups, normalize_length
from src.metrics import *

if __name__ == '__main__':

    excel_path = "../data/2.xlsx"
    output_path = "../data/scores_without_labels.xlsx"

    groups = load_match_groups(excel_path)

    rows = []  # ⭐ 必须初始化

    for match_id, group in groups.items():
        main = group["main"]

        # 安全检查（非常推荐）
        if main is None:
            print(f"⚠️ match_id1={match_id} 没有 main，跳过")
            continue

        target_len = len(main["series"])
        main_series = main["series"]

        for sub in group["subs"]:
            sub_series = normalize_length(sub["series"], target_len)

            # 子相似度
            cosine = cosine_similarity(main_series, sub_series)
            pearson = pearson_similarity(main_series, sub_series)
            dtw = dtw_similarity(main_series, sub_series, alpha=0.3 * len(main_series))
            amplitude = amplitude_similarity(main_series, sub_series)

            # 当前旧评分机制
            final_score = final_similarity_score(main_series, sub_series)

            rows.append({
                "match_id1": match_id,
                "main_id": main["id"],
                "sub_id": sub["id"],
                "cosine": float(cosine),
                "pearson": float(pearson),
                "dtw": float(dtw),
                "amplitude": float(amplitude),
                "final_score": float(final_score),
                "human_label": 0  # 先占位
            })

    # 👉 转成 DataFrame
    df = pd.DataFrame(rows)

    # 👉 简单自检（强烈建议）
    print("样本数：", len(df))
    print(df.head())

    # 👉 导出 Excel
    df.to_excel(output_path, index=False)

    print(f"✅ 已成功导出到 {output_path}")