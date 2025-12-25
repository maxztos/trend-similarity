import pandas as pd
from src.dataloader import load_match_groups, normalize_length
from src.metrics import (
    cosine_similarity,
    pearson_similarity,
    dtw_similarity,
    amplitude_similarity,
)
from src.calibration import SimilarityCalibrator
def score_from_raw_excel(
    input_excel: str,
    output_excel: str,
    model_path: str
):
    # 1️⃣ 解析原始Excel数据
    groups = load_match_groups(input_excel)

    # 2️⃣ 加载训练好的校准模型
    calibrator = SimilarityCalibrator()
    calibrator.load(model_path)

    rows = []

    # 3️⃣ 遍历每个 match_id1
    for match_id1, group in groups.items():
        main = group["main"]
        if main is None:
            continue

        main_series = main["series"]
        target_len = len(main_series)

        for sub in group["subs"]:
            sub_series = normalize_length(
                sub["series"], target_len
            )

            # 4️⃣ 计算基础特征（程序内部）
            cosine = cosine_similarity(main_series, sub_series)
            pearson = pearson_similarity(main_series, sub_series)
            dtw = dtw_similarity(main_series, sub_series, alpha=0.3 * len(main_series))
            amplitude = amplitude_similarity(main_series, sub_series)

            # 5️⃣ 校准评分（最终分数）
            df_feat = pd.DataFrame([{
                "cosine": cosine,
                "pearson": pearson,
                "dtw": dtw,
                "amplitude": amplitude
            }])

            calibrated_score = calibrator.predict_score(df_feat)[0]

            rows.append({
                "match_id1": match_id1,
                "main_id": main["id"],
                "sub_id": sub["id"],
                "calibrated_score": calibrated_score
            })

    # 6️⃣ 输出结果
    df_out = pd.DataFrame(rows)
    df_out = df_out.sort_values(
        by=["match_id1", "calibrated_score"],
        ascending=[True, False]
    )

    df_out.to_excel(output_excel, index=False)

    print(f"✅ 已完成评分，结果输出到 {output_excel}")


if __name__ == "__main__":
    input_excel = "data/2.xlsx"
    output_excel = "data/enterprise_scored.xlsx"
    model_path = "data/similarity_calibrator.joblib"

    score_from_raw_excel(
        input_excel=input_excel,
        output_excel=output_excel,
        model_path=model_path
    )
