import pandas as pd
import numpy as np
import ast
from collections import defaultdict

"""
解析xlsx数据
需要分析的数据结构
{
  match_id1: {
    "main": {
        "id": match_id2,
        "series": np.ndarray,
        "nums": [a11,a22,b11,b22,c11,c22,c33]
    },
    "subs": [
        {
            "id": match_id2,
            "series": np.ndarray,
            "nums": "nums": [a11,a22,b11,b22,c11,c22,c33]
        },
        ...
    ]
  },
  match_id2: {}
}

"""
def parse_s6(s):
    """
    安全解析 Excel 中的序列字段：
    - 支持 list / ndarray
    - 支持字符串形式的 list（含 nan / None）
    - 空值返回 None
    """

    # 1. Excel 空单元格
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None

    # 2. 已经是 list / ndarray
    if isinstance(s, (list, tuple, np.ndarray)):
        return np.asarray(s, dtype=np.float32)

    # 3. 字符串情况
    if isinstance(s, str):
        s = s.strip()
        if not s:
            return None

        # 关键：把 nan / None 显式转成 np.nan
        s = s.replace("nan", "np.nan").replace("None", "np.nan")

        try:
            return np.array(eval(s, {"np": np}), dtype=np.float32)
        except Exception as e:
            raise ValueError(f"无法解析序列字段: {s}") from e

    # 4. 兜底
    raise ValueError(f"不支持的类型: {type(s)}")

NUM_COLS = ["a11", "a22", "b11", "b22", "c11", "c22", "c33"]

"""
describe:提取Excel中的数据，封装成group变量
输入：Excel文件路径
输出：提取的Excel数据
"""
def load_match_groups(excel_path):
    df = pd.read_excel(excel_path)
    df = df.rename(columns={'match_id_2': 'match_id1', 'match_id': 'match_id2'})
    groups = defaultdict(lambda: {
        "main": None,
        "subs": []
    })

    for _, row in df.iterrows():
        mid1 = row["match_id1"]
        entry = {
            "id": row["match_id2"],
            "series": parse_s6(row["s6"]),
            "nums":np.array([row[col] for col in NUM_COLS], dtype=np.float32)
        }

        if row["Type"].lower() == "主数据":
            groups[mid1]["main"] = entry
        else:
            groups[mid1]["subs"].append(entry)

    return groups




if __name__ == '__main__':

    excel_path = "../../data/3.xlsx"
    groups = load_match_groups(excel_path)
    print(groups)


