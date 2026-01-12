from collections import defaultdict

import numpy as np

from src.utils.dataloader import load_match_groups


# 相关判断规则
def rule_length(main, sub, cfg):
    """长度相关规则"""
    return True


def rule_trend(main, sub, cfg):
    """
    趋势规则：
    后 10 个点中，符号不一致的数量 > 5 → 不通过
    """
    main_s = main["series"]
    sub_s  = sub["series"]

    n = cfg.get("trend_tail", 10)

    # 长度保护
    if len(main_s) < n or len(sub_s) < n:
        return True  # 工程上一般不在这里直接杀

    m_tail = main_s[-n:]
    s_tail = sub_s[-n:]

    # 符号（0 视为不参与）
    m_sign = np.sign(m_tail)
    s_sign = np.sign(s_tail)

    valid = (m_sign != 0) & (s_sign != 0)
    diff_cnt = np.sum(m_sign[valid] != s_sign[valid])

    return diff_cnt <= 5


def rule_amplitude(main, sub, cfg):
    """振幅相关规则"""
    return True


def rule_nums(main, sub, cfg):
    """nums 相关规则"""
    return True

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
    new_data, stats = preprocess_data(data)
    print_preprocess_stats(stats)
