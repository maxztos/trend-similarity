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

## 2. 系统整体流程

```
企业原始 Excel
      ↓
数据解析（load_match_groups）
      ↓
主 / 副序列配对
      ↓
基础相似度特征计算
  - cosine
  - pearson
  - dtw
  - amplitude
      ↓
校准模型（逻辑回归）
      ↓
calibrated_score
      ↓
按 match_id1 排序输出结果
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
