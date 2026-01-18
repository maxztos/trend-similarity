import numpy as np

from src.caculate import aggregate_new_nums_single, generate_nums
from src.contour_match import match_results
from src.ncc_match import ncc_match_results

excel_path = "../data/3n.xlsx"
# results = match_results(excel_path)
results = ncc_match_results(excel_path)

for th in np.arange(75.0, 86.0 + 1e-6, 1):
    th = round(th, 1)

    processed = generate_nums(results["results"], threshold=th)
    if not processed:
        continue

    stats = aggregate_new_nums_single(processed)

    # ---- 计算分组均值总和 ----
    group_mean_sum = sum(
        g["mean"] for g in stats["group"].values() if g["count"] > 0
    )

    # ================== 阈值头 ==================
    print("\n" + "=" * 70)
    print(
        f"THRESHOLD = {th:.1f}   "
        f"matches = {len(processed)}   "
        f"group_mean_sum = {group_mean_sum:.4f}"
    )
    print("=" * 70)

    # ================== 分组统计 ==================
    print(f"{'组':<15} {'总和':>10} {'均值':>10} {'次数':>8}")
    print("-" * 45)

    for name, g in stats["group"].items():
        print(
            f"{name:<15} "
            f"{g['sum']:>10.3f} "
            f"{g['mean']:>10.3f} "
            f"{g['count']:>8}"
        )
