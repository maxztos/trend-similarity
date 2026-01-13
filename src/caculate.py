import numpy as np

from src.contour_match import match_results

def vote_pair_k_ge_3(v1, v2):
    """
    v1, v2: shape (k,)
    返回 0 / 1 / None
    """
    v1 = v1[v1 != 0]
    v2 = v2[v2 != 0]

    p1 = np.sum(v1 > 0)
    n1 = np.sum(v1 < 0)
    p2 = np.sum(v2 > 0)
    n2 = np.sum(v2 < 0)

    if p1 > n1:
        return 0
    if p2 > n2:
        return 1
    return None

def vote_c_k_ge_3(sub_nums, main_nums):
    counts = []
    for i in [4,5,6]:
        v = sub_nums[:, i]
        v = v[v != 0]
        counts.append(np.sum(v > 0))

    max_cnt = max(counts)
    if max_cnt == 0 or counts.count(max_cnt) > 1:
        return None

    idx = counts.index(max_cnt)
    return 4 + idx

def vote_pair(sub_vals, main_vals, idx1, idx2):
    """
    sub_vals : shape (k, 2) -> 副数据的 a11 / a22
    main_vals: 主数据
    idx1, idx2: 在 main_nums 中的索引
    """
    k = sub_vals.shape[0]

    # -------- k == 1 --------
    if k == 1:
        if sub_vals[0, 0] > 0:
            return idx1
        if sub_vals[0, 1] > 0:
            return idx2
        return None

    # -------- k == 2 --------
    if k == 2:
        # a11 异号 → 矛盾
        if sub_vals[0, 0] * sub_vals[1, 0] < 0:
            return None

        # 同号
        if sub_vals[0, 0] > 0:
            return idx1
        if sub_vals[0, 1] > 0:
            return idx2
        return None

    # -------- k >= 3 --------
    pos = np.sum(sub_vals[:, 0] > 0)
    neg = np.sum(sub_vals[:, 0] < 0)

    if pos > neg:
        return idx1

    return None

def sign(x):
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0

def generate_masked_nums_np(main_nums, sub_nums):
    main_nums = np.asarray(main_nums, dtype=np.float32)
    sub_nums  = np.asarray(sub_nums, dtype=np.float32)

    return np.where(sub_nums > 0, main_nums, np.nan)
# 对于每个match中的sub :
    #   if final_socre >= threshold
    #       生成一组nums[a11,a22,b11,b22,c11,c22,c33]: sub中的 if a11 >= 0 时选择 main中对应的a11 如果如负数则为空值
    #

def generate_nums(results, threshold=45):
    import numpy as np
    from collections import defaultdict
    print(f"threshold={threshold}")

    def sign_conflict(v):
        """
        判断是否存在正负冲突（忽略 0）
        """
        v = v[v != 0]
        return np.any(v > 0) and np.any(v < 0)

    grouped = defaultdict(list)

    # ---- 分组 + 阈值过滤 ----
    for r in results:
        if isinstance(r, dict) and r.get("final_score") is not None and r["final_score"] >= threshold:
            grouped[r["match_id"]].append(r)

    processed = []

    for match_id, subs in grouped.items():
        main = np.asarray(subs[0]["main_nums"], dtype=float)
        sub  = np.stack([np.asarray(s["sub_nums"], dtype=float) for s in subs])
        k = sub.shape[0]

        new = np.full(7, np.nan)

        # =====================================================
        # a 组（a11 / a22）
        # =====================================================
        if k == 1:
            if sub[0, 0] > 0:
                new[0] = main[0]
            elif sub[0, 1] > 0:
                new[1] = main[1]

        elif k == 2:
            if not sign_conflict(sub[:, 0]):  # 只看 a11
                if sub[0, 0] > 0:
                    new[0] = main[0]
                elif sub[0, 1] > 0:
                    new[1] = main[1]

        else:  # k >= 3
            a11 = sub[:, 0]
            a22 = sub[:, 1]

            a11 = a11[a11 != 0]
            a22 = a22[a22 != 0]

            if np.sum(a11 > 0) > np.sum(a11 < 0):
                new[0] = main[0]
            elif np.sum(a22 > 0) > np.sum(a22 < 0):
                new[1] = main[1]
            # 否则整组不选

        # =====================================================
        # b 组（b11 / b22）——完全同 a 组
        # =====================================================
        if k == 1:
            if sub[0, 2] > 0:
                new[2] = main[2]
            elif sub[0, 3] > 0:
                new[3] = main[3]

        elif k == 2:
            if not sign_conflict(sub[:, 2]):  # 只看 b11
                if sub[0, 2] > 0:
                    new[2] = main[2]
                elif sub[0, 3] > 0:
                    new[3] = main[3]

        else:
            b11 = sub[:, 2]
            b22 = sub[:, 3]

            b11 = b11[b11 != 0]
            b22 = b22[b22 != 0]

            if np.sum(b11 > 0) > np.sum(b11 < 0):
                new[2] = main[2]
            elif np.sum(b22 > 0) > np.sum(b22 < 0):
                new[3] = main[3]

        # =====================================================
        # c 组（c11 / c22 / c33）
        # =====================================================
        if k == 1:
            for i in (4, 5, 6):
                # if main[i] <= 0.6:
                #     continue
                if sub[0, i] > 1.1:
                    new[i] = main[i]
                    break

        elif k == 2:
            for i in (4, 5, 6):
                # if main[i] <= 0.6:
                #     continue
                if not sign_conflict(sub[:, i]):
                    if sub[0, i] > 1.1:
                        new[i] = main[i]
                        break
        else:
            pos_counts = []
            valid_idx = []

            for i in (4, 5, 6):
                # if main[i] <= 0.6:
                #     continue

                v = sub[:, i]
                v = v[v != 0]

                if len(v) == 0:
                    continue

                pos_counts.append(np.sum(v > 0))
                valid_idx.append(i)

            if pos_counts:
                best_i = valid_idx[np.argmax(pos_counts)]
                new[best_i] = main[best_i]

        # else:
        #     pos_counts = []
        #     for i in (4, 5, 6):
        #         if main[i] <= 0.6:
        #             continue
        #         v = sub[:, i]
        #         v = v[v != 0]
        #         pos_counts.append(np.sum(v > 0))
        #
        #     max_pos = max(pos_counts)
        #     if max_pos > 0 and pos_counts.count(max_pos) == 1:
        #         idx = pos_counts.index(max_pos)
        #         new[4 + idx] = main[4 + idx]

        processed.append({
            "match_id": match_id,
            "new_nums": new
        })

    return processed

def generate_nums_old1(results, threshold=45):
    import numpy as np
    from collections import defaultdict

    grouped = defaultdict(list)

    # ---- 分组 + 阈值过滤 ----
    for r in results:
        if r.get("final_score") is not None and r["final_score"] >= threshold:
            grouped[r["match_id"]].append(r)

    processed = []

    for match_id, subs in grouped.items():
        main = np.asarray(subs[0]["main_nums"], dtype=float)
        sub  = np.stack([np.asarray(s["sub_nums"], dtype=float) for s in subs])
        k = sub.shape[0]

        new = np.full(7, np.nan)

        # =====================================================
        # a 组（a11 / a22）
        # =====================================================
        if k == 1:
            if sub[0, 0] > 0:
                new[0] = main[0]
            elif sub[0, 1] > 0:
                new[1] = main[1]

        elif k == 2:
            # 只看 a11 是否异号
            if np.sign(sub[0, 0]) * np.sign(sub[1, 0]) >= 0:
                if sub[0, 0] > 0:
                    new[0] = main[0]
                elif sub[0, 1] > 0:
                    new[1] = main[1]
            # 异号 → 整组不选

        else:  # k >= 3
            pos = np.sum(sub[:, 0] > 0)
            neg = np.sum(sub[:, 0] < 0)
            if pos > neg:
                new[0] = main[0]
            # 否则整组不选（不会选 a22）

        # =====================================================
        # b 组（b11 / b22）——完全同 a 组
        # =====================================================
        if k == 1:
            if sub[0, 2] > 0:
                new[2] = main[2]
            elif sub[0, 3] > 0:
                new[3] = main[3]

        elif k == 2:
            if np.sign(sub[0, 2]) * np.sign(sub[1, 2]) >= 0:
                if sub[0, 2] > 0:
                    new[2] = main[2]
                elif sub[0, 3] > 0:
                    new[3] = main[3]

        else:
            pos = np.sum(sub[:, 2] > 0)
            neg = np.sum(sub[:, 2] < 0)
            if pos > neg:
                new[2] = main[2]

        # =====================================================
        # c 组（c11 / c22 / c33）
        # =====================================================
        if k == 1:
            for i in (4, 5, 6):
                if sub[0, i] > 0:
                    new[i] = main[i]
                    break

        elif k == 2:
            # 任一维度异号 → 该维度不选
            for i in (4, 5, 6):
                if np.sign(sub[0, i]) * np.sign(sub[1, i]) >= 0:
                    if sub[0, i] > 0:
                        new[i] = main[i]
                        break

        else:
            pos = [np.sum(sub[:, i] > 0) for i in (4, 5, 6)]
            max_pos = max(pos)
            if pos.count(max_pos) == 1:
                idx = pos.index(max_pos)
                new[4 + idx] = main[4 + idx]

        processed.append({
            "match_id": match_id,
            "new_nums": new
        })

    return processed


def generate_nums_old(results, threshold=45):
    """
    对 match_results 的输出进行后处理：
    - final_score >= threshold
    - 每个 match_id 只保留一个结果
    - 优先选择 new_nums 中 (>0) 个数最多的那个
    """

    best_per_match = {}
    ok = 0

    for r in results:
        final_score = r.get("final_score")

        # 跳过无效或未评分
        if final_score is None or final_score < threshold:
            continue

        match_id = r["match_id"]

        new_nums = generate_masked_nums_np(
            r["main_nums"],
            r["sub_nums"]
        )

        # 统计 >0 的个数（NaN / 0 / -1 都不会被计入）
        positive_cnt = int(np.sum(np.asarray(new_nums) > 0))

        # 如果这个 match_id 还没记录，直接放
        if match_id not in best_per_match:
            best_per_match[match_id] = {
                "match_id": match_id,
                "new_nums": new_nums,
                "pos_cnt": positive_cnt
            }
            continue

        # 如果已有，比较 >0 个数
        if positive_cnt > best_per_match[match_id]["pos_cnt"]:
            best_per_match[match_id] = {
                "match_id": match_id,
                "new_nums": new_nums,
                "pos_cnt": positive_cnt
            }

    # 整理输出
    processed = []
    for v in best_per_match.values():
        ok += 1
        processed.append({
            "match_id": v["match_id"],
            "new_nums": v["new_nums"]
        })

    print(ok)
    return processed
NUM_KEYS = ["a11", "a22", "b11", "b22", "c11", "c22", "c33"]

def aggregate_new_nums_single(processed):
    """
    对 processed 中的 new_nums 进行统计：
    1) 逐维度：sum / mean / count
    2) 分组统计：
       - A: a11 + a22
       - B: b11 + b22
       - C: c11 + c22 + c33
    """

    if not processed:
        return {
            "sum": None,
            "mean": None,
            "count": None,
            "group": None
        }

    # ===== 1. 堆叠 =====
    stack = np.stack([
        np.asarray(x["new_nums"], dtype=np.float64)
        for x in processed
    ])  # shape: (N, 7)

    # ===== 2. 逐维度统计 =====
    sum_vals = np.nansum(stack, axis=0)
    count_vals = np.sum(~np.isnan(stack), axis=0)

    mean_vals = np.divide(
        sum_vals,
        count_vals,
        out=np.full_like(sum_vals, np.nan),
        where=count_vals > 0
    )

    # ===== 3. 分组统计 =====
    def group_stats(indices):
        g_sum = np.sum(sum_vals[indices])
        g_count = np.sum(count_vals[indices])

        g_mean = g_sum / g_count if g_count > 0 else np.nan
        return {
            "sum": g_sum,
            "mean": g_mean,
            "count": int(g_count)
        }

    group = {
        "A(a11+a22)": group_stats([0, 1]),
        "B(b11+b22)": group_stats([2, 3]),
        "C(c11+c22+c33)": group_stats([4, 5, 6])
    }

    return {
        "sum": sum_vals,
        "mean": mean_vals,
        "count": count_vals,
        "group": group
    }

def print_match_results(results, processed, threshold=48):
    """
    打印每个 match 的 a11~c33 选择结果，用于人工核对规则
    """
    import numpy as np
    from collections import defaultdict

    # ---- 原始副数据分组 ----
    grouped = defaultdict(list)
    for r in results:
        if r.get("final_score") is not None and r["final_score"] >= threshold:
            grouped[r["match_id"]].append(r)

    processed_map = {x["match_id"]: x["new_nums"] for x in processed}

    names = ["a11", "a22", "b11", "b22", "c11", "c22", "c33"]

    for match_id, subs in grouped.items():
        print("=" * 80)
        print(f"match_id: {match_id}")
        print(f"副数据条数 k = {len(subs)}")

        # ---- 打印副数据 ----
        for i, s in enumerate(subs):
            sub = np.asarray(s["sub_nums"], dtype=float)
            print(f"  sub[{i}]: ", end="")
            for name, v in zip(names, sub):
                print(f"{name}={v:6.2f}", end="  ")
            print()

        # ---- 打印最终选择 ----
        new = processed_map.get(match_id)
        print("\n>>> 最终 new_nums：")
        for name, v in zip(names, new):
            if np.isnan(v):
                print(f"  {name}: ❌ 未选")
            else:
                print(f"  {name}: ✅ 选中 → {v}")

        # ---- 分组总结 ----
        print("\n>>> 分组结果：")
        print("  a 组:", end=" ")
        if not np.isnan(new[0]):
            print("选 a11")
        elif not np.isnan(new[1]):
            print("选 a22")
        else:
            print("未选")

        print("  b 组:", end=" ")
        if not np.isnan(new[2]):
            print("选 b11")
        elif not np.isnan(new[3]):
            print("选 b22")
        else:
            print("未选")

        print("  c 组:", end=" ")
        if not np.isnan(new[4]):
            print("选 c11")
        elif not np.isnan(new[5]):
            print("选 c22")
        elif not np.isnan(new[6]):
            print("选 c33")
        else:
            print("未选")

        print()


if __name__ == '__main__':
    excel_path = "../data/3n.xlsx"
    results = match_results(excel_path)
    # print(results)
    processed = generate_nums(results["results"], threshold=52.0)
    stats = aggregate_new_nums_single(processed)

    labels = ["a11", "a22", "b11", "b22", "c11", "c22", "c33"]

    print("\n===== 单维度统计 =====")
    print(f"{'维度':<6} {'总和':>10} {'均值':>10} {'次数':>8}")
    print("-" * 38)

    for i, name in enumerate(labels):
        s = stats["sum"][i]
        m = stats["mean"][i]
        c = stats["count"][i]

        print(f"{name:<6} {s:>10.3f} {m:>10.3f} {int(c):>8}")

    print("\n===== 分组统计 =====")
    print(f"{'组':<15} {'总和':>10} {'均值':>10} {'次数':>8}")
    print("-" * 45)

    for name, g in stats["group"].items():
        print(
            f"{name:<15} "
            f"{g['sum']:>10.3f} "
            f"{g['mean']:>10.3f} "
            f"{g['count']:>8}"
        )


