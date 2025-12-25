"""
模型训练
"""
import pandas as pd
from src.calibration import SimilarityCalibrator

df = pd.read_excel("data/scores_with_labels.xlsx")

calibrator = SimilarityCalibrator()
calibrator.fit(df)

calibrator.save("data/similarity_calibrator.joblib")

print("✅ Calibration model trained and saved")

if __name__ == '__main__':
    clf = calibrator.model.named_steps["clf"]
    scaler = calibrator.model.named_steps["scaler"]

    weights = clf.coef_[0]
    features = ["cosine", "pearson", "dtw", "amplitude"]

    print("\n=== 校准权重（越大越符合人工判断）===")
    for f, w in zip(features, weights):
        print(f"{f:10s}: {w:.4f}")

