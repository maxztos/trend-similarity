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
    if isinstance(s, list):
        return np.array(s, dtype=np.float32)
    return np.array(ast.literal_eval(s), dtype=np.float32)

NUM_COLS = ["a11", "a22", "b11", "b22", "c11", "c22", "c33"]

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
            "series": parse_s6(row["s6"]),
            "nums": np.array([row[col] for col in NUM_COLS], dtype=np.float32)
        }

        if row["Type"].lower() == "主数据":
            groups[mid1]["main"] = entry
        else:
            groups[mid1]["subs"].append(entry)

    return groups




if __name__ == '__main__':

    excel_path = "../../data/2.xlsx"
    groups = load_match_groups(excel_path)
    print(groups)


