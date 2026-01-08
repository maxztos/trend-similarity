import numpy as np
import pandas as pd

from src.dataloader import load_match_groups


# 惩罚上下偏移
def penalty_amp(amp_main, amp_sub, tol=10, scale=20):
    """
    振幅差异惩罚函数
    - 差异在 tol 内：不扣分 (0)
    - 差异超过 tol：扣 5-10 分
    """
    delta = abs(amp_main - amp_sub)

    if delta <= tol:
        return 0.0, delta

    # 计算超出阈值的程度 (x)
    # x = 0 时（刚超标），p = 5
    # x 增大时，p 趋近于 10
    x = delta - tol

    # 方案：使用指数增长后截断，或者线性增长
    # 这里采用平滑的线性映射：在 scale 范围内从 5 分增加到 10 分
    p = 5 + (5 * (1 - np.exp(-x / scale)))

    # 确保扣分值在 5 到 10 之间
    penalty_score = round(min(10.0, p), 2)

    return penalty_score, delta


def penalty_mean(mu_main, mu_sub, tol=10, scale=20):
    """
    均值偏移惩罚函数
    - delta <= tol: 不扣分 (0.0)
    - delta > tol: 扣除 5-10 分
    - scale: 控制从 5 分增长到 10 分的斜率（差异达到 tol + scale 时扣满 10 分）
    """
    delta = abs(mu_main - mu_sub)

    if delta <= tol:
        return 0.0, delta

    # 计算超出部分的比例 (0.0 到 1.0 之间)
    # 使用 (delta - tol) / scale 并截断在 1.0
    ratio = min(1.0, (delta - tol) / scale)

    # 映射到 15-25 分区间
    # 基础分 15 + (额外最高 5 分 * 比例)
    penalty_score = 15.0 + (5.0 * ratio)

    return round(penalty_score, 2), delta



def trend_sign(x, eps=1e-3):
    if x > eps:
        return 1
    if x < -eps:
        return -1
    return 0


def penalty_trend(cont_main, cont_sub, head_n=3, tail_n=10, tol=2, scale=7):
    # --- 1. 趋势一致性计算 (原有逻辑) ---
    h_a, h_b = cont_main[:head_n], cont_sub[:head_n]
    sh_a = np.array([trend_sign(x) for x in h_a])
    sh_b = np.array([trend_sign(x) for x in h_b])
    mismatch_head = int(np.sum(sh_a != sh_b))

    t_a, t_b = cont_main[-tail_n:], cont_sub[-tail_n:]
    st_a = np.array([trend_sign(x) for x in t_a])
    st_b = np.array([trend_sign(x) for x in t_b])
    mismatch_tail = int(np.sum(st_a != st_b))

    total_mismatch = mismatch_head + mismatch_tail

    # --- 2. 新增：零点交叉强约束 (Zero-Crossing Constraint) ---
    # 定义内部函数计算零点交叉数
    def count_zero_crossings(seq):
        # 通过判断相邻点乘积是否小于 0 来识别过零点
        return np.sum(np.diff(np.sign(seq)) != 0)

    # 我们只关注后半段 (例如后 1/2 或后 20 个点，这里取 tail_n 相关范围)
    # 如果你想判断整个后半段，建议传入长度的一半
    z_main = count_zero_crossings(cont_main[-(len(cont_main) // 2):])
    z_sub = count_zero_crossings(cont_sub[-(len(cont_sub) // 2):])

    z_diff = abs(z_main - z_sub)

    # 强约束逻辑
    zc_penalty = 0.0
    if z_diff >= 4:
        zc_penalty = 5.0  # 直接扣 5 分

    # --- 3. 综合判定 ---
    # 趋势不匹配的得分 (6-10分)
    trend_penalty = 0.0
    if total_mismatch > tol:
        ratio = min(1.0, (total_mismatch - tol) / scale)
        trend_penalty = 6.0 + (4.0 * ratio)

    # 最终取两者中的最大惩罚，或者累加（推荐取最大或有条件的累加）
    # 这里我们采用：如果触发了强约束，则在原有基础上保底 5 分
    final_penalty = trend_penalty + zc_penalty

    return round(final_penalty, 2), total_mismatch


def penalty_asym(cont_main, cont_sub, tol=0.15, scale=0.3):
    """
    改进版：衡量平均强度不匹配惩罚
    - tol: 允许的平均强度差异比例 (例如 0.15 代表 15% 差异)
    """
    # 1. 计算平均强度（均值），消除长度影响
    mean_main = np.mean(np.abs(cont_main))
    mean_sub = np.mean(np.abs(cont_sub))

    # 2. 计算相对差异 (Relative Delta)
    # 使用比例而不是绝对值，更通用
    if mean_main == 0: return 0.0, 0
    delta_ratio = abs(mean_main - mean_sub) / mean_main

    # 3. 判定逻辑
    # 如果两个波形平均高度相差在 15% 以内，不扣分
    if delta_ratio <= tol:
        return 0.0, delta_ratio

    # 4. 计算惩罚 (5-10分)
    ratio = min(1.0, (delta_ratio - tol) / scale)
    penalty_score = 5.0 + (5.0 * ratio)

    return round(penalty_score, 2), round(delta_ratio, 4)
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

    # ===== AMP（振幅：扣除 5-10 分）=====
    p_amp, delta_amp = penalty_amp(
        main_series_stats["amp"],
        sub_series_stats["amp"]
    )
    if p_amp > 0:
        penalties.append({
            "type": "amp",
            "delta": delta_amp,
            "penalty": p_amp
        })

    # ===== MEAN（均值：扣除 5-10 分）=====
    p_mean, delta_mean = penalty_mean(
        main_series_stats["mean"],
        sub_series_stats["mean"]
    )
    if p_mean > 0:
        penalties.append({
            "type": "mean",
            "delta": delta_mean,
            "penalty": p_mean
        })

    # ===== TREND（末端趋势：扣除 5-10 分）=====
    p_trend, mismatch = penalty_trend(
        main_contour,
        sub_contour
    )
    if p_trend > 0:
        penalties.append({
            "type": "trend",
            "mismatch": mismatch,
            "penalty": p_trend
        })
    # p_asym, delta_asym = penalty_asym(
    #     main_contour,
    #     sub_contour
    # )
    # if p_asym > 0:
    #     penalties.append({
    #         "type": "asym",
    #         "mismatch": delta_asym,
    #         "penalty": p_asym
    #     })
    # ===== 累加所有扣分值 =====
    total_penalty = sum(p["penalty"] for p in penalties)

    # ===== 核心修改：直接减去扣得分 =====
    # 使用 max(0, ...) 确保最终得分不会出现负数
    final_score = max(0.0, base_score - total_penalty)

    return round(final_score, 2), penalties, total_penalty

if __name__ == "__main__":
    excel_path = "../data/2.xlsx"
    data = load_match_groups(excel_path)

