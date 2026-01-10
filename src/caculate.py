import numpy as np

from src.contour_match import match_results

def generate_masked_nums_np(main_nums, sub_nums):
    main_nums = np.asarray(main_nums, dtype=np.float32)
    sub_nums  = np.asarray(sub_nums, dtype=np.float32)

    return np.where(sub_nums >= 0, main_nums, np.nan)
# 对于每个match中的sub :
    #   if final_socre >= threshold
    #       生成一组nums[a11,a22,b11,b22,c11,c22,c33]: sub中的 if a11 >= 0 时选择 main中对应的a11 如果如负数则为空值
    #
def generate_nums(results,threshold = 48):
    """
        对 match_results 的输出进行后处理：
        - final_score >= threshold 的 sub
        - 生成 new_nums
        """
    processed = []
    ok = 0

    for r in results:
        final_score = r.get("final_score")

        # 跳过无效或未评分的
        if final_score is None:
            continue

        if final_score >= threshold:
            ok += 1

            new_nums = generate_masked_nums_np(
                r["main_nums"],
                r["sub_nums"]
            )
            print(new_nums)
            processed.append({
                "match_id": r["match_id"],
                # "sub_id": r["sub_id"],
                # "final_score": final_score,
                "new_nums": new_nums
            })
    print(ok)
    return processed
NUM_KEYS = ["a11", "a22", "b11", "b22", "c11", "c22", "c33"]

def aggregate_new_nums_single(processed):
    """
    对 processed 中的 new_nums 进行维度级统计：
    - sum   : 忽略 NaN（-1 / 0 参与）
    - mean  : 忽略 NaN
    - count : 非 NaN 的数量
    """

    if not processed:
        return {
            "sum": None,
            "mean": None,
            "count": None
        }

    # 1. 强制转为 float ndarray，防止 object dtype
    stack = np.stack([
        np.asarray(x["new_nums"], dtype=np.float64)
        for x in processed
    ])

    # 2. 统计（NaN-aware）
    sum_vals = np.nansum(stack, axis=0)
    count_vals = np.sum(~np.isnan(stack), axis=0)

    # nanmean 在“全 NaN 列”会给 warning，这里主动处理
    mean_vals = np.divide(
        sum_vals,
        count_vals,
        out=np.full_like(sum_vals, np.nan, dtype=np.float64),
        where=count_vals > 0
    )

    return {
        "sum": sum_vals,
        "mean": mean_vals,
        "count": count_vals
    }

def aggregate_new_nums_grouped(processed):
    """
    按组统计 new_nums：
    A = a11 + a22
    B = b11 + b22
    C = c11 + c22 + c33

    规则：
    - -1 / 0 / 正数 都是合法值
    - NaN 表示空
    - 组内：有值就加；全空才是 NaN
    """

    if not processed:
        return {
            "A": {"sum": np.nan, "mean": np.nan, "count": 0},
            "B": {"sum": np.nan, "mean": np.nan, "count": 0},
            "C": {"sum": np.nan, "mean": np.nan, "count": 0},
        }

    # 结果暂存
    A_vals, B_vals, C_vals = [], [], []

    for item in processed:
        nums = np.asarray(item["new_nums"], dtype=np.float64)

        # --- A 组 ---
        a_vals = nums[[0, 1]]
        a_valid = a_vals[~np.isnan(a_vals)]
        A_vals.append(np.sum(a_valid) if a_valid.size > 0 else np.nan)

        # --- B 组 ---
        b_vals = nums[[2, 3]]
        b_valid = b_vals[~np.isnan(b_vals)]
        B_vals.append(np.sum(b_valid) if b_valid.size > 0 else np.nan)

        # --- C 组 ---
        c_vals = nums[[4, 5, 6]]
        c_valid = c_vals[~np.isnan(c_vals)]
        C_vals.append(np.sum(c_valid) if c_valid.size > 0 else np.nan)

    def stats(arr):
        arr = np.asarray(arr, dtype=np.float64)
        count = np.sum(~np.isnan(arr))
        total = np.nansum(arr)
        mean = total / count if count > 0 else np.nan
        return {
            "sum": total,
            "mean": mean,
            "count": int(count)
        }

    return {
        "A": stats(A_vals),
        "B": stats(B_vals),
        "C": stats(C_vals),
    }


if __name__ == '__main__':
    excel_path= "../data/2n.xlsx"
    results = match_results(excel_path)
    # print(results)
    processed = generate_nums(results["results"])
    # print(processed)
    stats = aggregate_new_nums_grouped(processed)

    print(stats["A"])
    # {'sum': 12.4, 'mean': 0.89, 'count': 14}

    print(stats["B"])
    # {'sum': 3.2, 'mean': 0.27, 'count': 12}

    print(stats["C"])
    # {'sum': 18.6, 'mean': 1.55, 'count': 12}

