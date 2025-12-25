import pandas as pd

def topk_hit_rate(df, score_col, k=1):
    """
    计算 Top-K 命中率
    """
    hit = 0
    total = 0

    for match_id, g in df.groupby("match_id1"):
        # 只评估“确实有人工相似样本”的组
        if g["human_label"].sum() == 0:
            continue

        total += 1

        g_sorted = g.sort_values(
            by=score_col,
            ascending=False
        )

        topk = g_sorted.head(k)

        if topk["human_label"].sum() > 0:
            hit += 1

    return hit / total if total > 0 else 0

if __name__ == "__main__":
    df = pd.read_excel("../../data/scores_with_calibrated.xlsx")

    for k in [1, 3, 5]:
        hit_old = topk_hit_rate(df, "final_score", k)
        hit_new = topk_hit_rate(df, "calibrated_score", k)

        print(f"\nTop-{k} 命中率：")
        print(f"  旧机制      : {hit_old:.3f}")
        print(f"  校准后      : {hit_new:.3f}")
        print(f"  提升        : {hit_new - hit_old:+.3f}")
