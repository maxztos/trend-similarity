# trend-similarity

## 柱形分布图的相似度分析



## 1. 输入数据格式

Excel 表格包含以下字段：

| 字段名    | 说明                                                  |
| --------- | ----------------------------------------------------- |
| Type      | `main` / `sub`                                        |
| match_id1 | 同一个对比组 ID                                       |
| match_id2 | 每条曲线的唯一 ID                                     |
| s6        | 柱状图对应的一维数值序列（如 `[0, 1, 40, -20, ...]`） |

## 2. 系统接口说明

数据处理

```python
load_match_groups(excel_path)
输出：
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
  },
  match_id2: {}
}
```

轮廓趋势

```
extract_signed_area_contour(series,window=3)		输出:contour
contour_to_trend_segments(contour)			
输出:
segments:[
      {"trend": "+", "start": 0, "end": 18, "value": 12.3},
      {"trend": "-", "start": 18, "end": 39, "value": -8.7},
      {"trend": "+", "start": 39, "end": 52, "value": 21.5},
    ]
segments_to_timeline(segments) 
输出一个数组
```



## 3. 项目结构说明

```
.
├── data/
│   ├── raw.xlsx          # 原始Excel数据
│   ├── similarity_calibrator.joblib # 训练好的校准模型
│
├── src/
│   ├── data_loader.py               # 原始 Excel → 结构化数据
│   ├── metrics.py                # 各类相似度计算函数
│   ├── calibration.py               # 相似度校准模型（逻辑回归）
│
├── run_score_from_raw_excel.py      # ⭐ 主评分入口脚本
│
├── README.md
```

## 4. 核心数据结构

解析后的数据以如下结构组织：

```
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
```

## 5. 如何运行评分（推荐使用方式）

### 5.1 准备数据

将的原始 Excel 放入**（注意修改文件名、Excel列名）**：

```
data/raw.xlsx   注意修改文件名、Excel列名
```

### 5.2 运行评分脚本

```
python run_score_from_raw_excel.py
```

------

### 5.3 输出结果

生成文件：

```
data/enterprise_scored.xlsx
```



| 方法                 | 问题               |
| -------------------- | ------------------ |
| 趋势符号序列 (+ - 0) | 信息损失太大       |
| segment + IoU        | 边界极其敏感       |
| timeline 滑动        | 人工对齐规则过多   |
| 容忍 mIoU            | 还是在补丁式修规则 |
| **DTW(contour)**     | ✅ 自动处理一切     |
