import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def calculate_visual_score(arr_main, arr_sub, threshold=5.0):
    """
    计算两段数据的【视觉相似度】，模拟人眼判断逻辑。
    """
    length = len(arr_main)
    if length == 0: return -999  # 无重叠

    # --- 1. 时间权重 (Time Weighting) ---
    # 生成一个从 0.5 到 1.5 的线性权重，越靠后权重越高
    # 这满足了你 "尤其是后半段" 的需求
    time_weights = np.linspace(0.5, 1.5, length)

    # --- 2. 预处理：噪音过滤 (Noise Filtering) ---
    # 模拟人眼：太小的数值看起来就是0，忽略其正负
    # 使用 np.where，如果绝对值小于阈值，强制置为 0
    m_clean = np.where(np.abs(arr_main) < threshold, 0, arr_main)
    s_clean = np.where(np.abs(arr_sub) < threshold, 0, arr_sub)

    scores = []

    for i in range(length):
        m = m_clean[i]
        s = s_clean[i]

        # --- 3. 逐点打分逻辑 (Scoring Logic) ---

        # 情况 A: 都在死区内 (都是0) -> 视为匹配
        if m == 0 and s == 0:
            point_score = 1.0

        # 情况 B: 其中一个是0，另一个有值 -> 小幅扣分 (没有完全反向，只是程度不同)
        elif m == 0 or s == 0:
            point_score = 0.5

            # 情况 C: 同号 (同正或同负) -> 匹配
        elif np.sign(m) == np.sign(s):
            point_score = 1.0
            # (可选进阶: 如果幅度也差不多，可以给 1.2 分)

        # 情况 D: 异号 (一正一负) -> 严重不匹配
        else:
            # 这是一个关键点：大柱子反向要重罚，小柱子反向轻罚
            # 计算反向的严重程度
            magnitude = (abs(m) + abs(s)) / 2
            # 如果幅度很大 (比如 > 50)，分数给负分；幅度小，给 0 分
            if magnitude > 30:
                point_score = -1.0  # 严厉惩罚
            else:
                point_score = 0.0  # 普通不匹配

        scores.append(point_score)

    # --- 4. 计算加权平均分 ---
    # 分数 * 时间权重
    weighted_scores = np.array(scores) * time_weights

    # 归一化：除以权重的总和，让最终分数落在 [-1, 1] 之间方便理解
    # 1.0 = 完美匹配, 0.0 = 不相关, -1.0 = 完全相反
    final_score = np.sum(weighted_scores) / np.sum(time_weights)

    return final_score


def match_timelines_visual(main_data, sub_data, max_shift=20):
    """
    滑动匹配主函数
    """
    s_main = pd.Series(main_data)
    s_sub = pd.Series(sub_data)

    results = []

    # 左右滑动范围
    for shift in range(-max_shift, max_shift + 1):

        # 移动 sub
        s_sub_shifted = s_sub.shift(shift)

        # 获取重叠部分的索引
        valid_idx = s_main.notna() & s_sub_shifted.notna()

        m_segment = s_main[valid_idx].values
        s_segment = s_sub_shifted[valid_idx].values

        # 只有当重叠长度足够时才计算 (比如至少重叠一半)
        if len(m_segment) > len(main_data) * 0.5:
            # 调用上面的视觉评分算法
            score = calculate_visual_score(m_segment, s_segment, threshold=8.0)

            results.append({
                'shift': shift,
                'score': score
            })

    # 找出最高分
    df_results = pd.DataFrame(results)
    if df_results.empty:
        return None, df_results

    best_match = df_results.loc[df_results['score'].idxmax()]
    return best_match, df_results


# ================= 模拟测试 =================

# 第一个NumPy数组变量
np_arr1 = np.array([
    -22.71159862, -22.71159862, -22.71159862, -22.71159862, -22.71159862,
    -22.71159862, -22.71159862,  27.43769847,  27.43769847,  27.43769847,
    27.43769847,  27.43769847,  27.43769847,  27.43769847,  27.43769847,
    27.43769847,  27.43769847,  27.43769847,  27.43769847, -14.96571427,
    -14.96571427, -14.96571427, -14.96571427,   4.9193651,    4.9193651,
    4.9193651,  -10.79976189, -10.79976189, -10.79976189, -10.79976189,
    10.34940476,  10.34940476,  10.34940476,  10.34940476, -17.3373545,
    -17.3373545,  -17.3373545,  -17.3373545,  -17.3373545,  -17.3373545,
    -17.3373545,  -17.3373545,  -17.3373545,   15.675048,    15.675048,
    15.675048,    15.675048,    15.675048,   -28.85816672, -28.85816672,
    -28.85816672, -28.85816672, -28.85816672, -28.85816672, -28.85816672,
    -28.85816672, -28.85816672, -28.85816672,   9.22809523,   9.22809523
])

# 第二个NumPy数组变量
np_arr2 = np.array([
    -38.47653055, -38.47653055, -38.47653055, -38.47653055, -38.47653055,
    -38.47653055, -38.47653055,  46.54480523,  46.54480523,  46.54480523,
    46.54480523,  46.54480523,  46.54480523,  46.54480523,  46.54480523,
    46.54480523,  46.54480523,  46.54480523,  -6.33999999,  -6.33999999,
    21.46492045,  21.46492045,  21.46492045,  21.46492045,  21.46492045,
    21.46492045, -11.12161902, -11.12161902, -11.12161902, -11.12161902,
    -11.12161902,  13.26406927,  13.26406927,  13.26406927,  13.26406927,
    13.26406927,  13.26406927,  13.26406927,  13.26406927,  13.26406927,
    13.26406927,  13.26406927,  13.26406927,  -7.73920654,  -7.73920654,
    -7.73920654,   6.51214247,   6.51214247, -33.82057142, -33.82057142,
    -33.82057142, -33.82057142, -33.82057142,  37.06920654,  37.06920654,
    37.06920654
])

# 第三个NumPy数组变量
np_arr3 = np.array([
    57.88683674,  57.88683674,  57.88683674,  57.88683674,  57.88683674,
    57.88683674,  57.88683674,  57.88683674,  57.88683674,  57.88683674,
    57.88683674,  57.88683674,  57.88683674,  57.88683674,  -6.13404751,
    -6.13404751,  23.51893564,  23.51893564,  23.51893564,  23.51893564,
    23.51893564,  23.51893564,  23.51893564,  23.51893564,  23.51893564,
    23.51893564,  23.51893564,  23.51893564,  23.51893564,  23.51893564,
    23.51893564,  23.51893564,  23.51893564, -22.46353741, -22.46353741,
    -22.46353741, -22.46353741, -22.46353741, -22.46353741, -22.46353741,
    3.2165079,    3.2165079,    3.2165079,    3.2165079,    3.2165079,
    3.2165079,    3.2165079,    3.2165079,    3.2165079,  -22.91407742,
    -22.91407742, -22.91407742, -22.91407742, -22.91407742, -22.91407742,
    -22.91407742, -22.91407742
])
# 1. 造数据
np.random.seed(0)
# Main: 前半段乱动，后半段有几个明显的大波峰
# main_arr = np.concatenate([np.random.randn(30) * 10, [50, 60, 50, -40, -50, -40, 20]])
main_arr = np_arr1

# Sub: 把 Main 向右移 5 格，并且在前半段加很多干扰，但保留后半段的大趋势
real_offset = 5
# sub_arr = np.roll(main_arr, real_offset)
# sub_arr[:30] = np.random.randn(30) * 10  # 把前半段彻底改乱 (模拟上半场打得不一样)
# 此时，只有后半段趋势是一样的，且有一个偏移量
sub_arr = np_arr2

# 2. 运行匹配
best_res, df = match_timelines_visual(main_arr, sub_arr, max_shift=20)

# 3. 输出结果
print("=== 视觉感知匹配结果 ===")
print(f"最佳偏移量: {int(best_res['shift'])} (正数代表 Sub 滞后)")
print(f"视觉相似得分: {best_res['score']:.3f} (满分1.0)")

# 4. 绘图验证
plt.figure(figsize=(10, 5))
shift = int(best_res['shift'])

# 画 Main
plt.plot(main_arr, label='Main', color='black', linewidth=2)

# 画对齐后的 Sub
# 构建一个临时的对齐数组用于绘图
aligned_sub = pd.Series(sub_arr).shift(shift)
plt.plot(aligned_sub, label=f'Sub (Shifted {shift})', color='green', linestyle='--', linewidth=2)

# 标记一下重点关注的区域 (后半段)
plt.axvspan(len(main_arr) * 0.5, len(main_arr), color='yellow', alpha=0.1, label='Focus Area')

plt.title(f"Visual Alignment (Weighted on 2nd Half) - Score: {best_res['score']:.2f}")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()