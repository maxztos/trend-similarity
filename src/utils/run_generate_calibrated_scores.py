import pandas as pd
from src.calibration import SimilarityCalibrator

if __name__ == "__main__":
    input_path = "../../data/scores_with_labels.xlsx"
    model_path = "../../data/similarity_calibrator.joblib"
    output_path = "../../data/scores_with_calibrated.xlsx"

    # 1️⃣ 读入带人工标注的数据
    df = pd.read_excel(input_path)

    required_cols = [
        "cosine", "pearson", "dtw", "amplitude"
    ]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"❌ 缺少列: {c}")

    # 2️⃣ 加载校准模型
    calibrator = SimilarityCalibrator()
    calibrator.load(model_path)

    # 3️⃣ 预测校准后的分数
    df["calibrated_score"] = calibrator.predict_score(df)

    # 4️⃣（可选）按 match_id1 + 分数排序，方便人工查看
    df = df.sort_values(
        by=["match_id1", "calibrated_score"],
        ascending=[True, False]
    )

    # 5️⃣ 导出 Excel
    df.to_excel(output_path, index=False)

    print(f"✅ 已生成 {output_path}")
    print(df.head())
