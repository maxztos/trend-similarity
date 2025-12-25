import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
"""
使用逻辑回归学习。。。

"""

FEATURE_COLS = [
    "cosine",
    "pearson",
    "dtw",
    "amplitude"
]

LABEL_COL = "human_label"


class SimilarityCalibrator:
    """
    使用逻辑回归，将多种相似度子分数
    校准为一个更贴近人工判断的最终分数
    """

    def __init__(self):
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                class_weight="balanced",
                max_iter=1000
            ))
        ])

    def fit(self, df: pd.DataFrame):
        X = df[FEATURE_COLS].values
        y = df[LABEL_COL].values
        self.model.fit(X, y)

    def predict_score(self, df: pd.DataFrame) -> np.ndarray:
        """
        输出 0~100 的校准相似度分数
        """
        X = df[FEATURE_COLS].values
        prob = self.model.predict_proba(X)[:, 1]
        return prob * 100

    def save(self, path: str):
        joblib.dump(self.model, path)

    def load(self, path: str):
        self.model = joblib.load(path)
