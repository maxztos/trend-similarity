from collections import defaultdict

import numpy as np

from src.utils.dataloader import load_match_groups

class Color:
    RED = '\033[91m'      # 红：用于趋势错误
    YELLOW = '\033[93m'   # 黄：用于振幅错误
    CYAN = '\033[96m'     # 青：用于长度错误
    MAGENTA = '\033[95m'  # 紫：用于死线错误
    GREEN = '\033[92m'    # 绿：用于通过/豁免
    RESET = '\033[0m'     # 重置：必须加在结尾，否则后面全是彩色



# 相关判断规则
def rule_length(main, sub, cfg):
    """长度相关规则"""
    """
        长度规则：
        Sub 的长度不能比 Main 短太多。
        配置参数：
        - len_ratio: 最小长度比例 (默认 0.5，即 Sub 长度至少是 Main 的一半)
        """
    m_len = len(main["series"])
    s_len = len(sub["series"])

    # 获取阈值，默认 0.5
    ratio_threshold = cfg.get("len_ratio", 0.5)

    # 防止 Main 为空的情况
    if m_len == 0:
        return False

    ratio = s_len / m_len

    if ratio < ratio_threshold:
        print(
            f"{Color.CYAN}❌ [剔除] Sub ID: {sub['id']:<15} | 原因: 长度不足 (Sub: {s_len}, Main: {m_len}, 比例: {ratio:.2f} < {ratio_threshold}){Color.RESET}")
        return False

    return True

def rule_nums(main, sub, cfg):
    """nums 相关规则"""
    """
        数值/死线规则：
        检查 Sub 的标准差 (Standard Deviation)。
        如果标准差极小，说明信号是一条直线（死线）或无有效波动，
        这种数据没有匹配价值，且会导致后续计算（如相关系数）除以零报错。
        """
    s_s = sub["series"]

    # 计算标准差
    s_std = np.std(s_s)

    # 获取阈值，默认为 1
    # 只要波动小于这个数，就视为“死线”
    min_std = cfg.get("min_std", 1)

    if s_std < min_std:
        # 使用 MAGENTA (紫色) 标记死线错误
        print(
            f"{Color.MAGENTA}❌ [剔除] Sub ID: {sub['id']:<15} | 原因: 信号死线/无波动 (std: {s_std:.6f} < {min_std}){Color.RESET}")
        return False

    return True

def rule_amplitude(main, sub, cfg):
    """振幅相关规则"""
    m_amp = np.ptp(main["series"])  # ptp = max - min
    s_amp = np.ptp(sub["series"])

    if m_amp == 0: return True  # 主数据是一条直线，无法比较，跳过

    ratio = s_amp / m_amp

    # 默认允许 0.2倍 ~ 2.0倍 的差异
    min_r = cfg.get("amp_ratio_min", 0.4)
    max_r = cfg.get("amp_ratio_max", 1.6)

    if not (min_r <= ratio <= max_r):
        print(
            f"{Color.YELLOW}❌ [剔除] Sub ID: {sub['id']:<15} | 原因: 振幅不匹配 (Main: {m_amp:.1f}, Sub: {s_amp:.1f}, 比例: {ratio:.2f}){Color.RESET}")
        return False
    return True

def rule_trend(main, sub, cfg):
    """
    趋势规则（优化版）：
    1. 符号不一致数量检查 (硬指标)
    2. 相关性补救机制 (软指标，防止误杀)
    """
    main_s = main["series"]
    sub_s = sub["series"]

    # --- 配置参数 ---
    n = cfg.get("trend_tail", 30)  # 检查最后 30 个点
    limit = cfg.get("trend_limit", 15)  # 允许最多 15 个点符号不一致
    corr_limit = cfg.get("trend_corr", 0.2)  # 【新增】相关性豁免阈值
    # ----------------

    # 1. 长度保护
    if len(main_s) < n or len(sub_s) < n:
        return True

    # 2. 取尾部数据
    m_tail = main_s[-n:]
    s_tail = sub_s[-n:]

    # 3. 计算符号不一致 (原逻辑)
    m_sign = np.sign(m_tail)
    s_sign = np.sign(s_tail)

    # 只比较两者都不为0的点
    valid = (m_sign != 0) & (s_sign != 0)
    diff_cnt = np.sum(m_sign[valid] != s_sign[valid])

    # 4. 判断逻辑
    if diff_cnt <= limit:
        return True  # 直接通过

    # =========================================
    # 🚀 优化核心：进入“补救模式”
    # =========================================
    # 代码走到这里，说明 diff_cnt > limit，原本应该被杀掉。
    # 现在我们计算 Pearson 相关系数，看看是否冤枉了它。

    # 计算相关系数 (处理常数序列除0风险)
    if np.std(m_tail) == 0 or np.std(s_tail) == 0:
        corr = 0  # 无法计算相关性，视为不相关
    else:
        corr = np.corrcoef(m_tail, s_tail)[0, 1]

    if corr > corr_limit:
        # 虽然符号不对，但趋势高度相关，给予豁免！
        # 可以在这里打印一条特殊的日志，方便你知道谁被“救”回来了
        print(f"{Color.GREEN}⚠️ [豁免] Sub ID: {sub['id']:<15} | 符号不一致: {diff_cnt} (Fail) 但 相关系数: {corr:.2f}(Pass){Color.RESET}")
        return True

    # 如果相关系数也很差，那就真的剔除
    print(f"{Color.RED}❌ [剔除] Sub ID: {sub['id']:<15} | 原因: 趋势背离 (不一致: {diff_cnt}, 相关性: {corr:.2f}){Color.RESET}")
    return False

FILTER_RULES = [
    rule_length,
    rule_trend,
    rule_amplitude,
    rule_nums,
]

def preprocess_data(data: dict, cfg: dict = None):
    """
    预筛选框架
    - 输入 data
    - 输出 data（结构不变）
    """
    if cfg is None:
        cfg = {}

    new_data = {}

    # ---------- 调试信息 ----------
    stats = {
        "total_removed": 0,

        # 人类调试主视图
        # match_id -> { sub_id -> [rule1, rule2, ...] }
        "details": defaultdict(lambda: defaultdict(list)),

        # 规则统计视图
        # rule_name -> [(match_id, sub_id)]
        "by_rule": defaultdict(list),
    }
    print("=" * 60)
    print(f"Starting Preprocess... Config: {cfg}")
    print(
        f"Key: {Color.CYAN}Length{Color.RESET} | {Color.MAGENTA}Nums{Color.RESET} | {Color.YELLOW}Amplitude{Color.RESET} | {Color.RED}Trend{Color.RESET}")
    print("=" * 60)

    for match_id, block in data.items():
        main = block["main"]
        subs = block.get("subs", [])

        kept_subs = []

        for sub in subs:
            removed = False

            for rule in FILTER_RULES:
                if not rule(main, sub, cfg):
                    removed = True
                    stats["total_removed"] += 1
                    stats["details"][match_id].setdefault(sub["id"], []).append(rule.__name__)
                    stats["by_rule"][rule.__name__].append((match_id, sub["id"]))
                    break

            if not removed:
                kept_subs.append(sub)

        new_data[match_id] = {
            "main": main,
            "subs": kept_subs
        }

    return new_data, stats

def print_preprocess_stats(stats):
    print("\n========== Preprocess Report ==========")
    print(f"Total removed subs: {stats['total_removed']}")

    print("\n--- Removed by match (with reasons) ---")
    for match_id, subs in stats["details"].items():
        print(f"\n{match_id}:")
        for sub_id, rules in subs.items():
            rule_str = ", ".join(rules)
            print(f"  - {sub_id}  ❌ {rule_str}")

    print("\n--- Removed by rule (summary) ---")
    for rule, items in stats["by_rule"].items():
        print(f"{rule}: {len(items)}")

    print("\n=======================================\n")


if __name__ == '__main__':
    excel_path = '../data/2n.xlsx'
    data = load_match_groups(excel_path)

    config = {
        # --- 1. 长度规则 (Length Rule) ---
        "len_ratio": 0.5,  # Sub 长度至少是 Main 的 50%
        # 作用：防御性拦截极短或截断的数据

        # --- 2. 死线规则 (Nums Rule) ---
        "min_std": 10,  # 【你要求的设置】标准差小于 1 视为死线/无波动
        # 作用：剔除直线或波动极小的数据

        # --- 3. 振幅规则 (Amplitude Rule) ---
        "amp_ratio_min": 0.6,  # Sub 振幅最小是 Main 的 0.6 倍
        "amp_ratio_max": 1.5,  # Sub 振幅最大是 Main 的 1.5 倍
        # 作用：确保两者的能量/量级是“门当户对”的

        # --- 4. 趋势规则 (Trend Rule) ---
        "trend_tail": 30,  # 只检查最后 30 个数据点
        "trend_limit": 15,  # 在这 30 个点里，允许最多 15 个点符号相反
        "trend_corr": 0.2,  # 如果符号相反数量超标，但相关系数 > 0.2，则给予豁免
        # 作用：确保形状、走势高度一致
    }

    new_data, stats = preprocess_data(data, cfg=config)
    print_preprocess_stats(stats)

