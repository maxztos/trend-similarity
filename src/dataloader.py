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
        "series": np.ndarray
    },
    "subs": [
        {
            "id": match_id2,
            "series": np.ndarray
        },
        ...
    ]
  }
}

"""
def parse_s6(s):
    if isinstance(s, list):
        return np.array(s, dtype=np.float32)
    return np.array(ast.literal_eval(s), dtype=np.float32)

"""
describe:提取Excel中的数据，封装成group变量
输入：Excel文件路径
输出：提取的Excel数据
"""
def load_match_groups(excel_path):
    df = pd.read_excel(excel_path)

    groups = defaultdict(lambda: {
        "main": None,
        "subs": []
    })

    for _, row in df.iterrows():
        mid1 = row["match_id1"]
        entry = {
            "id": row["match_id2"],
            "series": parse_s6(row["s6"])
        }

        if row["Type"].lower() == "主数据":
            groups[mid1]["main"] = entry
        else:
            groups[mid1]["subs"].append(entry)

    return groups

def normalize_length(series, target_len):
    series = np.asarray(series, dtype=np.float32)

    if len(series) == target_len:
        return series

    x_old = np.linspace(0, 1, len(series))
    x_new = np.linspace(0, 1, target_len)
    return np.interp(x_new, x_old, series)




if __name__ == '__main__':

    excel_path = "../data/2.xlsx"
    groups = load_match_groups(excel_path)
    print(groups)


