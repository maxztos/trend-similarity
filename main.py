import numpy as np

from src.contour_match import get_contour, calculate_dtw_distance
from src.dataloader import load_match_groups
from src.timeline_match import get_timelines

if __name__ == '__main__':
    # Excel文件路径
    excel_path = "data/2.xlsx"

    # 1. 加载所有比赛分组数据
    data = load_match_groups(excel_path)
    # print(data)

    # 2. 存储所有match_id的最小DTW结果（方便后续查看/分析）
    all_min_results = []

    # 3. 遍历所有match_id（不再硬编码单个ID）
    for match_id in data.keys():
        try:
            # 获取当前match_id对应的比赛数据
            # print(match_id)
            match_data = data[match_id]
            # 获取轮廓数据（主序列+子序列）
            contour_data = get_contour(match_data)
            # contour_data = get_timelines(match_data)
            # print(contour_data)
            main_con = contour_data["main"]["contour"]
            # main_con = contour_data["main"]["timeline"]

            # 存储当前match_id下所有子序列的DTW结果
            sub_results = []

            # 遍历当前比赛的所有子序列
            for sub in contour_data["subs"]:
                # # 跳过空的子序列（避免计算报错）
                # if not sub["contour"]:
                #     print(f"警告：match_id[{match_id}]的子ID[{sub['id']}]轮廓为空，跳过计算")
                #     continue

                # 计算DTW距离
                dist = calculate_dtw_distance(main_con, sub["contour"])
                # print(main_con, sub["contour"])
                # 计算归一化距离（消除长度影响）
                norm_dist = dist / (len(main_con) + len(sub["contour"]))

                sub_results.append({
                    "sub_id": sub["id"],
                    "distance": dist,
                    "norm_distance": norm_dist
                })

            # 4. 找到当前match_id的最小DTW距离结果
            if sub_results:  # 确保有子序列数据
                min_result = min(sub_results, key=lambda x: x["norm_distance"])
                # 补充match_id信息，方便溯源
                min_result["match_id"] = match_id
                all_min_results.append(min_result)

                # 打印当前match_id的最小DTW结果（格式清晰）
                print(f"=====================================")
                print(f"主数据ID: {match_id}")
                print(f"最优副数据ID: {min_result['sub_id']}")
                # print(f"最小DTW距离: {min_result['distance']:.2f}")
                # print(f"最小归一化DTW距离: {min_result['norm_distance']:.4f}")
            else:
                # 无有效子序列的情况
                print(f"=====================================")
                print(f"比赛ID: {match_id} | 无有效子序列数据，无法计算DTW距离")
                all_min_results.append({
                    "match_id": match_id,
                    "sub_id": None,
                    "distance": None,
                    "norm_distance": None
                })

        except Exception as e:
            # 捕获单个match_id处理失败的异常，不影响整体遍历
            print(f"=====================================")
            print(f"处理主数据ID[{match_id}]时出错: {str(e)}")
            all_min_results.append({
                "match_id": match_id,
                "sub_id": None,
                "distance": None,
                "norm_distance": None
            })