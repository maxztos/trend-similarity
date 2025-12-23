# 相似度算法
"""
相似度评分应该满足：

主图 vs N 幅图 可直接排序

对 整体趋势 / 形态 敏感

对 幅度轻微差异 不太敏感

输出 0～100 的直观分数

后续可以加指标，不推翻结构
| 指标           | 作用    | 原因      |
| ------------ | ----- | ------- |
| 余弦相似度        | 形态方向  | 对整体走势最稳 |
| Pearson 相关系数 | 趋势一致性 | 对峰谷同步敏感 |
| DTW 距离（归一化）  | 局部错位  | 抗轻微平移   |

"""
import numpy as np
from scipy.stats import pearsonr
from scipy.spatial.distance import cosine
from fastdtw import fastdtw
from math import exp

def cosine_similarity(a, b):
    return 1 - cosine(a, b)

def pearson_similarity(a, b):
    r, _ = pearsonr(a, b)
    return (r + 1) / 2

def dtw_similarity(a, b, alpha):
    dist, _ = fastdtw(a, b)
    return exp(-dist / alpha)

def final_similarity_score(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    cos = max(0.0, cosine_similarity(a, b))
    pear = pearson_similarity(a, b)
    dtw = dtw_similarity(a, b, alpha=0.3 * len(a))

    score = 0.4 * cos + 0.4 * pear + 0.2 * dtw
    return round(score * 100, 2)
