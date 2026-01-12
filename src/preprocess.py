from src.utils.dataloader import load_match_groups


# 相关判断规则
def rule_length(main, sub, cfg):
    """长度相关规则"""
    return True


def rule_trend(main, sub, cfg):
    """趋势相关规则"""
    return True


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

    for match_id, block in data.items():
        main = block["main"]
        subs = block.get("subs", [])

        kept_subs = []

        for sub in subs:
            keep = True

            for rule in FILTER_RULES:
                if not rule(main, sub, cfg):
                    keep = False
                    break

            if keep:
                kept_subs.append(sub)

        # 保留原结构
        new_data[match_id] = {
            "main": main,
            "subs": kept_subs
        }

    return new_data

if __name__ == '__main__':
    excel_path = '../data/2n.xlsx'
    data = load_match_groups(excel_path)
    new_data = preprocess_data(data)
    print(new_data)
