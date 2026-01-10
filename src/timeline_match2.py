import numpy as np

from src.utils.dataloader import load_match_groups
from src.timeline_match import get_timelines

def timeline_to_runs(tl, min_len=1):
    runs = []
    curr_sign = np.sign(tl[0])
    start = 0

    for i in range(1, len(tl)):
        if np.sign(tl[i]) != curr_sign:
            length = i - start
            if length >= min_len:
                runs.append((curr_sign, length))
            start = i
            curr_sign = np.sign(tl[i])

    # 收尾
    length = len(tl) - start
    if length >= min_len:
        runs.append((curr_sign, length))

    return runs

def run_similarity(r1, r2, len_tol=0.5):
    s1, l1 = r1
    s2, l2 = r2

    if s1 != s2:
        return 0.0

    ratio = min(l1, l2) / max(l1, l2)

    if ratio < (1 - len_tol):
        return 0.0

    return ratio
def match_run_sequences(main_runs, sub_runs):
    """
    return best_score
    """
    best = 0.0

    for shift in range(-len(sub_runs), len(main_runs)):
        score = 0.0
        matched = 0

        for i, sr in enumerate(sub_runs):
            j = i + shift
            if 0 <= j < len(main_runs):
                score += run_similarity(main_runs[j], sr)
                matched += 1

        if matched > 0:
            best = max(best, score / matched)

    return best

def coverage_score(main_runs, sub_runs):
    return min(len(sub_runs), len(main_runs)) / max(len(sub_runs), len(main_runs))

def final_trend_score(main_tl, sub_tl,
                      w_match=0.5,
                      w_cover=0.5):
    """
    返回一个 0~1 的最终相似度
    """
    main_runs = timeline_to_runs(main_tl)
    sub_runs  = timeline_to_runs(sub_tl)

    if len(main_runs) == 0 or len(sub_runs) == 0:
        return 0.0

    match_score = match_run_sequences(main_runs, sub_runs)
    cover_score = coverage_score(main_runs, sub_runs)

    final = w_match * match_score + w_cover * cover_score
    return final


def lcs_length(a, b):
    dp = [[0]*(len(b)+1) for _ in range(len(a)+1)]
    for i in range(len(a)):
        for j in range(len(b)):
            if a[i] == b[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
    return dp[-1][-1]


if __name__ == '__main__':

    excel_path = "../data/2.xlsx"
    excel_data = load_match_groups(excel_path)
    match_data = excel_data["2025/05/15-64VS54-60"]
    data = get_timelines(match_data)
    main_tl = data["main"]["timeline"]
    # print(main_tl)
    # print(data["subs"][-2]["timeline"])
    # print(data["subs"][-1]["timeline"])
    results = []

    for sub in data["subs"]:
        score = final_trend_score(
            main_tl,
            sub["timeline"]
        )

        results.append({
            "sub_id": sub["id"],
            "score": score
        })

    # 按 score 排序
    results.sort(key=lambda x: x["score"], reverse=True)

    # 打印
    for r in results:
        print(f"{r['sub_id']} | score: {r['score']:.3f}")