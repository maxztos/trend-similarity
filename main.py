# 主程序入口

from src.preprocess import extract_series_from_image
from src.metrics import final_similarity_score
from src.show import plot_main_vs_sub


def compare_main_vs_many(main_img, sub_imgs, bins=60):
    main_series = extract_series_from_image(main_img, bins)

    results = []
    for img in sub_imgs:
        sub_series = extract_series_from_image(img, bins)
        score = final_similarity_score(main_series, sub_series)
        results.append({
            "image": img,
            "score": score
        })

    return sorted(results, key=lambda x: x["score"], reverse=True)

"""
main_img: 主图路径(一张)
sub_imgs: 对比图路径列表(支持多张)
"""
if __name__ == "__main__":
    main_img = "D:/CProject/ImageProcessing/Trend-similarity/data/sample2/main.png"
    sub_imgs = [
        "D:/CProject/ImageProcessing/Trend-similarity/data/sample2/fig1.png",
        "D:/CProject/ImageProcessing/Trend-similarity/data/sample2/fig2.png",
    ]

    score_results = compare_main_vs_many(main_img, sub_imgs)

    for r in score_results:
        print(f"与主图的相似度得分{r['image'][-8:]}: {r['score']}")
    main_series = extract_series_from_image(main_img)

    results = []
    for img in sub_imgs:
        sub_series = extract_series_from_image(img)
        score = final_similarity_score(main_series, sub_series)
        results.append((img, sub_series, score))

    # 排序
    results.sort(key=lambda x: x[2], reverse=True)

    # 取最相似的一张
    best_img, best_series, best_score = results[0]

    print(f"相似度最高的图: {best_img[-8:]}, score={best_score}")

    plot_main_vs_sub(
        main_series,
        best_series,
        score=best_score,
        title=f"Main vs {best_img}"
    )


