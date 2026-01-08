import numpy as np
import pandas as pd

from src.dataloader import load_match_groups


# 惩罚上下偏移
def penalty_amp(amp_main, amp_sub, tol=15, scale=15):
    delta = abs(amp_main - amp_sub)

    if delta <= tol:
        return 0.0, delta

    # ⭐ 指数爆炸
    p = np.exp((delta - tol) / scale) - 1
    return min(1.0, p), delta


def penalty_mean(mu_main, mu_sub, tol=10, scale=30):
    delta = abs(mu_main - mu_sub)
    if delta <= tol:
        return 0.0, delta
    return min(1.0, (delta - tol) / scale), delta



def trend_sign(x, eps=1e-3):
    if x > eps:
        return 1
    if x < -eps:
        return -1
    return 0

def penalty_last_trend(cont_main, cont_sub, last_n=10, tol=3, scale=10):
    a = cont_main[-last_n:]
    b = cont_sub[-last_n:]

    sa = np.array([trend_sign(x) for x in a])
    sb = np.array([trend_sign(x) for x in b])

    mismatch = int(np.sum(sa != sb))

    if mismatch <= tol:
        return 0.0, mismatch

    return min(1.0, (mismatch - tol) / scale), mismatch
def penalty_to_score(penalty_ratio, min_penalty=5, max_penalty=30):
    """
    penalty_ratio: 0~1
    return: 实际扣分（>= min_penalty）
    """
    if penalty_ratio <= 0:
        return 0.0

    return min_penalty + penalty_ratio * (max_penalty - min_penalty)
def penalty_asym(delta, scale=10):
    return min(1.0, delta / scale)
def series_stats(series):
    x = np.asarray(series, dtype=np.float32)

    pos = x[x > 0]
    neg = x[x < 0]

    mu_all = float(x.mean())
    mu_pos = float(pos.mean()) if len(pos) else 0.0
    mu_neg = float(neg.mean()) if len(neg) else 0.0

    amp = mu_pos - mu_neg   # ⭐ 你现在定义的 amp

    return {
        "mean": mu_all,
        "amp": amp,
        "mu_pos": mu_pos,
        "mu_neg": mu_neg
    }
def apply_penalties(
    base_score,
    main_series_stats,
    sub_series_stats,
    main_contour,
    sub_contour
):
    penalties = []

    # ===== AMP（来自原始序列）=====
    p, delta = penalty_amp(
        main_series_stats["amp"],
        sub_series_stats["amp"]
    )
    if p > 0:
        penalties.append({
            "type": "amp",
            "delta": delta,
            "penalty": p
        })

    # ===== MEAN（来自原始序列）=====
    p, delta = penalty_mean(
        main_series_stats["mean"],
        sub_series_stats["mean"]
    )
    if p > 0:
        penalties.append({
            "type": "mean",
            "delta": delta,
            "penalty": p
        })

    # ===== TREND（来自轮廓）=====
    p, mismatch = penalty_last_trend(
        main_contour,
        sub_contour
    )
    if p > 0:
        penalties.append({
            "type": "trend",
            "mismatch": mismatch,
            "penalty": p
        })

    # ===== 权重融合 =====
    total_penalty = (
            sum(p["penalty"] for p in penalties if p["type"] == "amp") +
            sum(p["penalty"] for p in penalties if p["type"] == "mean") +
            sum(p["penalty"] for p in penalties if p["type"] == "trend")
    )

    final_score = base_score * (1 - total_penalty)

    return final_score, penalties, total_penalty

if __name__ == "__main__":
    excel_path = "../data/2.xlsx"
    data = load_match_groups(excel_path)

