from src.contour_match import match_results


# 输出表格字段：原有字段名+score、P（分数）、xianshi（1、0）

import pandas as pd

def write_scores_to_excel_v2(
    excel_path,
    results,
    out_path=None,
    score_col="final_score",
    flag_col="xianshi",
    threshold=50
):
    """
    在原始 Excel（match_id1 / match_id2）基础上
    写入 final_score + xianshi
    """

    # 1. 读取原始 Excel
    df = pd.read_excel(excel_path)

    # 2. 结果转 DataFrame
    res_df = pd.DataFrame(results)

    # 只保留需要字段
    res_df = res_df[["match_id", "sub_id", "final_score"]]

    # 3. merge（关键点在这里）
    df = df.merge(
        res_df,
        left_on=["match_id1", "match_id2"],
        right_on=["match_id", "sub_id"],
        how="left"
    )

    # 4. 删除多余字段（可选，但强烈建议）
    df.drop(columns=["match_id", "sub_id"], inplace=True)

    # 5. xianshi 标志
    df[flag_col] = (df[score_col] >= threshold).astype(int)

    # 6. 输出
    if out_path is None:
        out_path = excel_path.replace(".xlsx", "_with_score.xlsx")

    df.to_excel(out_path, index=False)

    print(f"✔ 已生成文件: {out_path}")
    print(f"✔ 显示阈值: {threshold}")

    return df



if __name__ == "__main__":
    excel_path = "../data/3n.xlsx"
    data = match_results(excel_path)
    print(data)
    ret = match_results(excel_path)
    results = ret["results"]

    write_scores_to_excel_v2(
        excel_path=excel_path,
        results=results,
        threshold=52  # 你自己定，比如 45 / 50 / 60
    )
