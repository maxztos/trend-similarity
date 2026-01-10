from src.contour_match import get_contour, calculate_dtw_distance, dtw_to_score
from src.utils.dataloader import load_match_groups
from src.scoring import series_stats, apply_penalties

# test dell git
if __name__ == '__main__':
    # Excel文件路径
    excel_path = "data/2n.xlsx"

    # 结果保存路径
    # output_xlsx_path = "data/match_results2n.xlsx"
    # 存储用于导出的平铺数据
    excel_rows = []
    # 1. 加载所有比赛分组数据
    data = load_match_groups(excel_path)
    # print(data)

    # 2. 存储所有match_id的最小DTW结果（方便后续查看/分析）
    all_min_results = []

    # 3. 遍历所有match_id（不再硬编码单个ID）
    for match_id in data.keys():
        try:
            # 获取当前match_id对应的比赛数据
            match_data = data[match_id]
            # print(match_data["main"]["series"])
            # 获取轮廓数据（主序列+子序列）
            contour_data = get_contour(match_data)

            main_con = contour_data["main"]["contour"]
            main_series = contour_data["main"]["series"]
            main_series_stats = series_stats(main_series)
            # 存储当前match_id下所有子序列的DTW结果
            sub_results = []

            # 遍历当前比赛的所有子序列
            for sub in contour_data["subs"]:

                dist = calculate_dtw_distance(main_con, sub["contour"])
                avg_dist = dist / max(len(main_con), len(sub["contour"]))
                base_score = dtw_to_score(avg_dist)

                sub_series_stats = series_stats(sub["series"])
                # 惩罚项计算

                final_score, penalties, total_penalty = apply_penalties(
                    base_score,
                    main_series_stats,
                    sub_series_stats,
                    main_con,
                    sub["contour"]
                )

                sub_results.append({
                    "sub_id": sub["id"],
                    "distance": dist,
                    "avg_distance": avg_dist,
                    "base_score": base_score,
                    "final_score": final_score,
                    "total_penalty": total_penalty,
                    "penalties": penalties
                })

            # ========== 按最终得分降序排序所有子序列 ==========
            if sub_results:
                # 按 final_score 降序（分数越高越相似）
                sub_results_sorted = sorted(
                    sub_results,
                    key=lambda x: x["final_score"],
                    reverse=True
                )

                print(f"\n=====================================")
                print(f"主数据ID: {match_id}")
                print(f"所有子序列匹配结果（按最终得分降序）:")
                print(
                    f"{'子ID':<18} "
                    # f"{'DTW':>8} "
                    f"{'Base':>7} "
                    f"{'Penalty':>8} "
                    f"{'Final':>7}"
                )
                print("-" * 65)

                for res in sub_results_sorted:
                    print(
                        f"{res['sub_id']:<18} "
                        # f"{res['distance']:>8.1f} "
                        f"{res['base_score']:>7.2f} "
                        f"{res['total_penalty']:>8.2f} "
                        f"{res['final_score']:>7.2f}"
                    )

                    # ===== 打印扣分项（如有）=====
                    if res["penalties"]:
                        for p in res["penalties"]:
                            if p["type"] == "amp":
                                print( #| Δamp={p['delta']:.1f} | penalty = {p["penalty"]}
                                    f"    - AMP惩罚 "
                                )
                            elif p["type"] == "mean":
                                print(# | Δmean={p['delta']:.1f}  | penalty = {p["penalty"]}
                                    f"    - MEAN惩罚 "
                                )
                            elif p["type"] == "trend":
                                print(# | mismatch={p['mismatch']}  | penalty = {p["penalty"]}
                                    f"    - 趋势惩罚 "
                                )
                            # elif p["type"] == "asym":
                            #     print(
                            #         f"    - 同步惩罚"
                            #     )

                # ===== 当前 match_id 下的最佳匹配 =====
                best = sub_results_sorted[0]
                best["match_id"] = match_id
                all_min_results.append(best)

            else:
                print(f"\n=====================================")
                print(f"主数据ID: {match_id} | 无有效子序列数据")
                all_min_results.append({
                    "match_id": match_id,
                    "sub_id": None,
                    "distance": None,
                    "base_score": None,
                    "final_score": None
                })

        except Exception as e:
            #捕获单个match_id处理失败的异常
            print(f"\n=====================================")
            print(f"处理主数据ID[{match_id}]时出错: {str(e)}")
            all_min_results.append({
                "match_id": match_id,
                "sub_id": None,
                "distance": None,
                "norm_distance": None,
                "score": None
            })
