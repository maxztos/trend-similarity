import numpy as np

from src.dataloader import load_match_groups
from src.timeline_match import get_timelines

def dilate_1d(mask, radius=5):
    """
    mask: bool array
    radius: 时间容忍度（±radius）
    """
    n = len(mask)
    out = np.zeros(n, dtype=bool)

    idx = np.where(mask)[0]
    for i in idx:
        l = max(0, i - radius)
        r = min(n, i + radius + 1)
        out[l:r] = True

    return out

def tolerant_iou(mask_a, mask_b):
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return inter / (union + 1e-6)
def tolerant_miou(a, b, tol=5):
    """
    a, b: timeline（连续值）
    tol: 时间轴容忍 ±tol
    """
    a = np.asarray(a)
    b = np.asarray(b)

    pos_a = dilate_1d(a > 0, tol)
    pos_b = dilate_1d(b > 0, tol)

    neg_a = dilate_1d(a < 0, tol)
    neg_b = dilate_1d(b < 0, tol)

    pos_iou = tolerant_iou(pos_a, pos_b)
    neg_iou = tolerant_iou(neg_a, neg_b)

    return 0.5 * (pos_iou + neg_iou)

def timeline_to_sign_mask(tl):
    """
    tl: np.ndarray
    return: np.ndarray of {-1, +1}
    """
    s = np.sign(tl)
    s[s == 0] = 0   # 可选：你现在基本没有 0
    # print(s)
    return s

def iou_binary(a, b, value):
    """
    a, b: np.ndarray
    value: +1 or -1
    """
    a_mask = (a == value)
    b_mask = (b == value)

    inter = np.logical_and(a_mask, b_mask).sum()
    union = np.logical_or(a_mask, b_mask).sum()

    if union == 0:
        return 1.0  # 都没有该趋势，视为一致
    return inter / union

def miou_score(a, b):
    """
    a, b: same length sign arrays
    """
    pos_iou = iou_binary(a, b, 1)
    neg_iou = iou_binary(a, b, -1)
    return 0.5 * (pos_iou + neg_iou)

def slide_miou(main_tl, sub_tl, max_slide=20, min_overlap=15):
    main_s = timeline_to_sign_mask(main_tl)
    sub_s  = timeline_to_sign_mask(sub_tl)

    len_m = len(main_s)
    len_s = len(sub_s)

    best = {
        "score": -1,
        "shift": 0,
        "direction": None
    }

    # ===== 子图从左往右滑 =====
    for shift in range(-max_slide, max_slide + 1):
        if shift >= 0:
            m_start = shift
            s_start = 0
        else:
            m_start = 0
            s_start = -shift

        overlap = min(len_m - m_start, len_s - s_start)
        if overlap < min_overlap:
            continue

        a = main_s[m_start:m_start + overlap]
        b = sub_s[s_start:s_start + overlap]

        # score = miou_score(a, b)
        score = tolerant_miou(a, b, tol=5)

        if score > best["score"]:
            best.update({
                "score": score,
                "shift": shift,
                "direction": "right" if shift > 0 else "left"
            })

    return best

if __name__ == '__main__':

    excel_path = "../data/2.xlsx"
    excel_data = load_match_groups(excel_path)
    match_data = excel_data["2025/05/15-64VS54-60"]
    data = get_timelines(match_data)
    main_tl = data["main"]["timeline"]
    # print(main_tl)
    # print(data["subs"][-2]["timeline"])
    # print(data["subs"][-1]["timeline"])
    results = []

    for sub in data["subs"]:
        best_match = slide_miou(
            main_tl,
            sub["timeline"],
            max_slide=15,
            min_overlap=15
        )

        results.append({
            "sub_id": sub["id"],
            "score": best_match["score"],
            "shift": best_match["shift"],
            "direction": best_match["direction"]
        })

    # 按 mIoU score 降序排序
    results.sort(key=lambda x: x["score"], reverse=True)

    # 打印结果
    for r in results:
        print(
            f"{r['sub_id']} | "
            f"mIoU: {r['score']:.3f} | "
            f"shift: {r['shift']} | "
            f"dir: {r['direction']}"
        )