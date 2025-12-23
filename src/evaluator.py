# 综合评估/打分
from src.metrics import final_similarity_score

"""
| 分数     | 含义         |
| ------ | ---------- |
| 90～100 | 极其相似（几乎同形） |
| 80～90  | 非常相似       |
| 70～80  | 相似         |
| 60～70  | 勉强相似       |
| < 60   | 差异明显       |

"""

#
def compare_main_vs_many(main, others):
    results = []
    for k, s in others.items():
        score = final_similarity_score(main, s)
        results.append((k, score))
    return sorted(results, key=lambda x: x[1], reverse=True)
