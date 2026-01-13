
import numpy as np

from src.caculate import aggregate_new_nums_single, generate_nums
from src.contour_match import match_results

excel_path = "../data/3n.xlsx"
results = match_results(excel_path)

best = {
    "threshold": None,
    "group_mean_sum": -np.inf,
    "stats": None,
}

records = []  # 方便你之后画图 / 打印

for th in np.arange(30.0, 60.0 + 1e-6, 1):
    th = round(th, 1)

    processed = generate_nums(results["results"], threshold=th)

    # 没有结果直接跳过
    if not processed:
        continue

    stats = aggregate_new_nums_single(processed)

    # ---- 计算 分组均值总和 ----
    group_mean_sum = 0.0
    for g in stats["group"].values():
        if g["count"] > 0:
            group_mean_sum += g["mean"]

    records.append({
        "threshold": th,
        "group_mean_sum": group_mean_sum,
        "match_cnt": len(processed),
    })

    # ---- 更新最优 ----
    if group_mean_sum > best["group_mean_sum"]:
        best["threshold"] = th
        best["group_mean_sum"] = group_mean_sum
        best["stats"] = stats

print("\n===== threshold 扫描结果 =====")
print(f"{'th':>6} {'mean_sum':>12} {'matches':>8}")
print("-" * 30)

for r in records:
    print(
        f"{r['threshold']:>6.1f} "
        f"{r['group_mean_sum']:>12.4f} "
        f"{r['match_cnt']:>8}"
    )
