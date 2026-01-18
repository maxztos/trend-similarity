from src.contour_match import get_contour, calculate_dtw_distance, dtw_to_score, ncc
from src.preprocess import preprocess_data
from src.utils.dataloader import load_match_groups
from src.scoring import series_stats, apply_penalties
def ncc_match_results(excel_path):
    # excel_path = "data/2n.xlsx"

    # 结果保存路径
    # output_xlsx_path = "data/match_results2n.xlsx"
    # 存储用于导出的平铺数据
    raw_data = load_match_groups(excel_path)
    # data = load_match_groups(excel_path)
    # 对源数据预处理
    data, stats = preprocess_data(raw_data)

    # 2. 存储所有match_id的最小DTW结果（方便后续查看/分析）
    # all_min_results = []
    match_count = 0  # match_id 数量
    total_sub_count = 0  # sub 总数
    sub_results = []

    # 3. 遍历所有match_id（不再硬编码单个ID）
    # for match_id in data.keys():
    for match_id, match_data in data.items():
        try:
            match_count += 1
            if match_data["main"] is None:
                continue
            # 获取当前match_id对应的比赛数据
            # match_data = data[match_id]
            # print(match_data["main"]["series"])
            # 获取轮廓数据（主序列+子序列）
            main_nums = match_data["main"]["nums"]
            contour_data = get_contour(match_data)

            main_con = contour_data["main"]["contour"]
            main_series = contour_data["main"]["series"]
            main_series_stats = series_stats(main_series)
            # 存储当前match_id下所有子序列的DTW结果
            # sub_results = []

            # 遍历当前比赛的所有子序列
            for sub in contour_data["subs"]:
                total_sub_count += 1
                # ========== 方法0：DTW ==========
                dist = calculate_dtw_distance(main_con, sub["contour"])
                avg_dist = dist / max(len(main_con), len(sub["contour"]))
                base_score0 = dtw_to_score(avg_dist)

                # ========== 方法1：NCC(Max Cross-Correlation) ==========
                base_score1 = ncc(main_con, sub["contour"], max_lag=10)

                sub_series_stats = series_stats(sub["series"])

                # ========== 分别计算惩罚项 & 最终分 ==========
                final_score0, penalties0, total_penalty0 = apply_penalties(
                    base_score0,
                    main_series_stats,
                    sub_series_stats,
                    main_con,
                    sub["contour"],
                )

                final_score1, penalties1, total_penalty1 = apply_penalties(
                    base_score1,
                    main_series_stats,
                    sub_series_stats,
                    main_con,
                    sub["contour"],
                )

                # 用“更高的最终分”作为该 sub 的推荐方法（仅用于展示，不影响你后续自定义）
                best_method = "DTW" if final_score0 >= final_score1 else "NCC"

                sub_results.append({
                    "match_id": match_id,
                    "sub_id": sub["id"],
                    "main_nums": main_nums,
                    "sub_nums": sub["nums"],

                    # DTW
                    "distance": dist,
                    "avg_distance": avg_dist,
                    "base_score0": base_score0,
                    "final_score0": final_score0,
                    "total_penalty0": total_penalty0,
                    "penalties0": penalties0,

                    # NCC
                    "base_score1": base_score1,
                    "final_score1": final_score1,
                    "total_penalty1": total_penalty1,
                    "penalties1": penalties1,

                    "best_method": best_method,
                })

            # ========== 按最终得分降序排序所有子序列 ==========
            # if sub_results:
            #     # 按 final_score 降序（分数越高越相似）
            #     # 默认按两种方法里“更高的最终分”排序，方便对比
            #     sub_results_sorted = sorted(
            #         sub_results,
            #         key=lambda x: max(x["final_score0"], x["final_score1"]),
            #         reverse=True,
            #     )
            #
            #     print(f"\n=====================================")
            #     print(f"主数据ID: {match_id}")
            #     print(f"所有子序列匹配结果（按最终得分降序）:")
            #     print(
            #         f"{'子ID':<18} "
            #         f"{'DTW_Base':>9} "
            #         f"{'DTW_Pen':>8} "
            #         f"{'DTW_Final':>9} "
            #         f"{'NCC_Base':>9} "
            #         f"{'NCC_Pen':>8} "
            #         f"{'NCC_Final':>9} "
            #         f"{'Best':>5}"
            #     )
            #     print("-" * 105)
            #
            #     for res in sub_results_sorted:
            #         print(
            #             f"{res['sub_id']:<18} "
            #             f"{res['base_score0']:>9.2f} "
            #             f"{res['total_penalty0']:>8.2f} "
            #             f"{res['final_score0']:>9.2f} "
            #             f"{res['base_score1']:>9.2f} "
            #             f"{res['total_penalty1']:>8.2f} "
            #             f"{res['final_score1']:>9.2f} "
            #             f"{res['best_method']:>5}"
            #         )
            #
            #         # ===== 打印扣分项（如有）=====
            #         # 这里分别打印 DTW 与 NCC 的扣分项，便于对比
            #         for tag, ps in [("DTW", res["penalties0"]), ("NCC", res["penalties1"])]:
            #             if not ps:
            #                 continue
            #             print(f"    [{tag}] 扣分项:")
            #             for p in ps:
            #                 if p["type"] == "amp":
            #                     print("      - AMP惩罚")
            #                 elif p["type"] == "mean":
            #                     print("      - MEAN惩罚")
            #                 elif p["type"] == "trend":
            #                     print("      - 趋势惩罚")
            #
            #     # ===== 当前 match_id 下的最佳匹配 =====
            #     # 当前 match_id 下的最佳匹配（按 max(DTW_Final, NCC_Final) 排序）
            #     best = sub_results_sorted[0]
            #     best["match_id"] = match_id
            #     all_min_results.append(best)
            #
            # else:
            #     print(f"\n=====================================")
            #     print(f"主数据ID: {match_id} | 无有效子序列数据")
            #     all_min_results.append({
            #         "match_id": match_id,
            #         "sub_id": None,
            #         "distance": None,
            #         "base_score": None,
            #         "final_score": None
            #     })

        except Exception as e:
            # 捕获单个match_id处理失败的异常
            print(f"\n=====================================")
            print(f"处理主数据ID[{match_id}]时出错: {str(e)}")
            sub_results.append({
                "match_id": match_id,
                "sub_id": None,
                "distance": None,
                "norm_distance": None,
                "score": None
            })
    return {
        "results": sub_results,
        "stats": {
            "match_count": match_count,
            "total_sub_count": total_sub_count
        }
}