# -*- coding: utf-8 -*-
"""手画掩码校验器 —— 画完存成 PNG，跑这一个脚本就知道能不能用。

用法:
    .\\.venv\\Scripts\\python.exe check_mask.py           # 校验 images-2027/map_mask27.png（现役掩码）
    .\\.venv\\Scripts\\python.exe check_mask.py 我画的.png              # 校验任意文件
    .\\.venv\\Scripts\\python.exe check_mask.py --ref images-2027/map_mask27.png  # 指定参照图算 IoU

只做四件事：格式 / 面积与分块 / 边界是否压在墨线上 / 与参照图差多少。
main.py 的读法（main.py:1455）是 `color[0]==color[1]==color[2]==0` 判地面，
所以像素必须**严格只有** (0,0,0) 和 (0,255,0) 两种颜色，抗锯齿、羽化、
JPEG、调色板 PNG 全部会污染判定。
"""
import sys, os, json
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import cv2

ROOT = r"E:\code\visual_robot\rader"
MAP = os.path.join(ROOT, "images-2027")
H, W, PX = 2800, 1500, 100.0          # 100 px = 1 m
TOL_PX = 5                            # 5 px = 50 mm，笔画宽 8 px 的中心线容差
INK_TOL_PX = 6                        # 边界找墨的容差（略放宽给手画）
PURE_BLACK = (0, 0, 0)
PURE_GREEN = (0, 255, 0)

problems, warns = [], []


def m_of(row, col):
    """像素 -> 米制：x 短轴向右，y 长轴自底边向上。"""
    return col / PX, (H - row) / PX


def load_chroma():
    """红/蓝两族墨线（色度判据，不用亮度：海报是 9 级灰阶浮雕）。"""
    src = cv2.imread(os.path.join(MAP, "red.jpg"), cv2.IMREAD_COLOR)
    b, g, r = [c.astype(np.int16) for c in cv2.split(src)]
    k = np.ones((3, 3), np.uint8)
    red = cv2.dilate(((r - 0.5 * (g + b)) > 45).astype(np.uint8), k, iterations=1) > 0
    blue = cv2.dilate(((b - 0.5 * (g + r)) > 45).astype(np.uint8), k, iterations=1) > 0
    return red, blue


def report(path):
    print("=" * 74)
    print("校验:", path)
    print("=" * 74)
    if not os.path.exists(path):
        print("文件不存在。")
        return False

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print("读不出来（不是图像文件？）")
        return False
    print("shape=%s  dtype=%s  %d bytes" % (img.shape, img.dtype, os.path.getsize(path)))

    # ---------- 1) 通道与尺寸 ----------
    if img.ndim == 2:
        problems.append("灰度/单通道 PNG。main.py 会读成 (H,W)，取 color[1] 直接崩。"
                        "必须存成 24-bit RGB（3 通道）PNG。")
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        warns.append("带 alpha 通道；已按 BGR 取前三通道判定，建议存成无 alpha 的 PNG。")
        img = img[:, :, :3]
    if img.shape[:2] != (H, W):
        problems.append("尺寸 %dx%d，应为 %dx%d（宽%d 高%d）。掩码与 red.jpg 必须逐像素对齐，"
                        "差一点整场坐标就偏。" % (img.shape[1], img.shape[0], W, H, W, H))
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_NEAREST)
        warns.append("已临时缩放到 %dx%d 继续检查，但请回去按原尺寸重存。" % (W, H))

    # ---------- 2) 严格二值 ----------
    flat = img.reshape(-1, 3)
    uniq, cnt = np.unique(flat, axis=0, return_counts=True)
    order = np.argsort(-cnt)
    uniq, cnt = uniq[order], cnt[order]
    print("\n[1] 颜色（按像素数排序，前 8）")
    bad_total = 0
    for c, n in zip(uniq[:8], cnt[:8]):
        tagc = tuple(int(v) for v in c)
        ok = tagc in (PURE_BLACK, PURE_GREEN)
        print("    BGR%-16s %9d  %6.3f%%  %s" % (str(tagc), n, 100.0 * n / flat.shape[0],
                                                 "OK" if ok else "<-- 非法"))
    for c, n in zip(uniq, cnt):
        if tuple(int(v) for v in c) not in (PURE_BLACK, PURE_GREEN):
            bad_total += int(n)
    if bad_total:
        problems.append("存在 %d 个非纯色像素（%.3f%%）。抗锯齿/羽化/JPEG 压缩都会这样，"
                        "main.py 会把它们全判成高地。用「无抗锯齿的铅笔/油漆桶」重画，"
                        "或最后做一次阈值化：非黑一律拉成 (0,255,0)。"
                        % (bad_total, 100.0 * bad_total / flat.shape[0]))
    else:
        print("    -> 严格二值，只有 (0,0,0) 与 (0,255,0)。")

    green = np.all(img == np.array(PURE_GREEN, np.uint8), axis=2)
    gfrac = float(green.mean())
    print("\n[2] 面积")
    print("    绿占比 %.6f   高地 %.3f m²" % (gfrac, gfrac * 15.0 * 28.0))
    print("    参照值 0.137010 / 57.544 m²（±15% 内算正常手画）")
    if not (0.1165 <= gfrac <= 0.1576):
        problems.append("绿占比 %.4f 落在 0.1165..0.1576 之外。要么漏画了整块，"
                        "要么把非高地也涂了。" % gfrac)

    # ---------- 3) 分块 ----------
    n, lab, stats, cent = cv2.connectedComponentsWithStats(green.astype(np.uint8), 8)
    blocks = []
    for i in range(1, n):
        a_m2 = stats[i, cv2.CC_STAT_AREA] / (PX * PX)
        if a_m2 < 0.5:
            continue
        x, y = m_of(cent[i][1], cent[i][0])
        x0, y1 = m_of(stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] - 1, stats[i, cv2.CC_STAT_LEFT])
        x1, y0 = m_of(stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] - 1)
        role = ("中央高地" if 4.0 <= x <= 11.0 and 9.5 <= y <= 18.5 and 1.0 <= (x1 - x0) <= 5.0
                else "红方梯形" if y < 14.0 else "蓝方梯形")
        blocks.append(dict(id=i, a=a_m2, cx=x, cy=y, x0=x0, x1=x1, y0=y0, y1=y1, role=role))
    print("\n[3] 连通块（≥0.5 m²）共 %d 块" % len(blocks))
    for b in blocks:
        print("    %-6s 面积 %6.2f m²  中心(%.2f, %5.2f)  x %.2f..%.2f  y %.2f..%.2f"
              % (b["role"], b["a"], b["cx"], b["cy"], b["x0"], b["x1"],
                 min(b["y0"], b["y1"]), max(b["y0"], b["y1"])))
    if len(blocks) < 3:
        problems.append("只找到 %d 块 ≥0.5 m² 的高地；规则要求 3 块（红梯形 / 蓝梯形 / 中央高地）。"
                        "两块粘在一起说明边界没分开。" % len(blocks))
    if len(blocks) > 4:
        warns.append("块数偏多（%d），检查是否有零碎溢出的绿色。" % len(blocks))
    touch = cv2.copyMakeBorder(green.astype(np.uint8), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    if touch[0].any() or touch[-1].any() or touch[:, 0].any() or touch[:, -1].any():
        problems.append("绿色贴到画布边缘 —— 场地边界外不该有高地。")

    # ---------- 4) 边界是否压在墨线上 ----------
    red_i, blue_i = load_chroma()
    ring = green & ~cv2.erode(green.astype(np.uint8), np.ones((3, 3), np.uint8), 1).astype(bool)
    any_ink = red_i | blue_i
    # 边界像素到最近墨点的距离（用距离变换，单位 px）
    dist_ink = cv2.distanceTransform((~any_ink).astype(np.uint8), cv2.DIST_L2, 5)
    dr = dist_ink[ring]
    print("\n[4] 边界 → 最近墨线（%d 个边界像素）" % ring.sum())
    if dr.size:
        for q in (50, 75, 90, 99):
            print("    p%-3d %7.1f mm" % (q, np.percentile(dr, q) * 10.0))
        print("    ≤40 mm: %.1f%%   ≤60 mm: %.1f%%"
              % (100.0 * (dr <= 4).mean(), 100.0 * (dr <= INK_TOL_PX).mean()))
        print("    说明：海报上只有 4 条边被画出来（远边/内竖/中竖/外竖），"
              "近边与斜边本来无墨，它们天然拉高这个数；看下面的色度归属更有用。")

    # ---------- 5) 色度归属：这块到底画在红墨还是蓝墨上 ----------
    # 只统计「附近真有墨」的边界像素：没被画出来的边（近边/斜边）本来无墨，
    # 算进去会把命中率摊薄，参照图自己也只有 0.75，判别力全丢。
    print("\n[5] 色度归属（分母只算压到墨的边界像素）")
    red_d = cv2.dilate(red_i.astype(np.uint8), np.ones((13, 13), np.uint8)) > 0
    blue_d = cv2.dilate(blue_i.astype(np.uint8), np.ones((13, 13), np.uint8)) > 0
    for b in blocks:
        m = (lab == b["id"]) & ring
        cnt_px = int(m.sum())
        if not cnt_px:
            continue
        own_img = red_d if b["role"] == "红方梯形" else blue_d
        for_img = blue_d if b["role"] == "红方梯形" else red_d
        n_own = int((m & own_img).sum())
        n_for = int((m & for_img).sum())
        n_ink = n_own + n_for
        cov = float(n_ink) / cnt_px
        if n_ink == 0:
            print("    %-6s 整条边界没压到任何墨（覆盖 0.000）" % b["role"])
            continue
        own = n_own / n_ink
        for_ = n_for / n_ink
        b["own"], b["foreign"] = own, for_
        flag = ""
        if b["role"] in ("红方梯形", "蓝方梯形"):
            if own < 0.85:
                flag = " <-- 压到的墨主要是对面颜色，红蓝画反/整体错位？"
                problems.append("%s 压到墨的边界里只有 %.1f%% 是本侧颜色，检查是否把半场画反。"
                                % (b["role"], 100.0 * own))
            if for_ > 0.15:
                flag = " <-- 蹭到对面颜色的墨"
                problems.append("%s 有 %.1f%% 的边界落在对面颜色的墨上，检查是否跨过中线。"
                                % (b["role"], 100.0 * for_))
            if cov < 0.30:
                flag += " (压墨覆盖偏低)"
                warns.append("%s 只有 %.0f%% 的边界落在墨线 6 px 内，其余边可能没对着笔画中心。"
                             % (b["role"], 100.0 * cov))
        print("    %-6s 压墨覆盖 %.3f  own %.3f  foreign %.3f%s"
              % (b["role"], cov, own, for_, flag))

    # ---------- 6) 与参照图比 ----------
    ref_path = REF
    if ref_path and os.path.exists(ref_path) and os.path.abspath(ref_path) != os.path.abspath(path):
        ref = cv2.imread(ref_path, cv2.IMREAD_COLOR)
        if ref is not None:
            rg = np.all(ref == np.array(PURE_GREEN, np.uint8), axis=2)
            if rg.shape == green.shape:
                inter = float((rg & green).sum())
                uni = float((rg | green).sum())
                iou = inter / uni if uni else 1.0
                print("\n[6] 与参照 %s" % os.path.basename(ref_path))
                print("    IoU %.4f   只你画 %.2f m²   只参照 %.2f m²"
                      % (iou, (green & ~rg).sum() / (PX * PX), (rg & ~green).sum() / (PX * PX)))
                if iou < 0.80:
                    warns.append("与参照图 IoU %.3f < 0.80：差异较大，若你有依据就没问题，"
                                 "否则对一下上面的逐块 bbox。" % iou)

    # ---------- 结论 ----------
    print("\n" + "=" * 74)
    if problems:
        print("结论：不通过（%d 条）" % len(problems))
        for i, p in enumerate(problems, 1):
            print("  %d. %s" % (i, p))
    else:
        print("结论：通过 —— 格式严格二值、三块齐全、边界压在墨线上。")
    for w in warns:
        print("  提醒: %s" % w)
    print("=" * 74)
    return not problems


def sanitize(src, dst=None):
    """强制二值化：绿色占优且够亮 -> (0,255,0)，其余一律 (0,0,0)。

    抗锯齿、羽化、半透明边、JPEG 振铃在这一步被吃掉。用别的软件画完
    先跑这个，比回去重画省事。注意判据是「绿通道压过红蓝」，
    所以直接画在彩色海报上的图不会被误判成高地。
    """
    img = cv2.imread(src, cv2.IMREAD_UNCHANGED)
    if img is None:
        print("--fix: 读不出来", src)
        return src
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[:2] != (H, W):
        print("--fix: 尺寸 %dx%d != %dx%d，按最近邻缩放后再二值化"
              % (img.shape[1], img.shape[0], W, H))
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_NEAREST)
    b, g, r = [c.astype(np.int16) for c in cv2.split(img)]
    green = (g >= 128) & (g > r + 32) & (g > b + 32)
    out = np.zeros((H, W, 3), np.uint8)
    out[green] = (0, 255, 0)
    if dst is None:
        dst = os.path.splitext(src)[0] + ".fixed.png"
    cv2.imwrite(dst, out)
    chk = cv2.imread(dst, cv2.IMREAD_COLOR)
    changed = int(np.any(img != out, axis=2).sum())
    print("--fix: 写出 %s\n       改动 %d 像素 (%.3f%%)  高地 %.3f m²  通道数 %d  颜色数 %d"
          % (dst, changed, 100.0 * changed / (H * W), green.sum() / (PX * PX),
             chk.shape[2] if chk.ndim == 3 else 1,
             len(np.unique(chk.reshape(-1, 3), axis=0))))
    return dst


if __name__ == "__main__":
    args = [a for a in sys.argv[1:]]
    REF = os.path.join(MAP, "map_mask27.png")
    if "--ref" in args:
        i = args.index("--ref")
        REF = args[i + 1]
        del args[i:i + 2]
    fix = "--fix" in sys.argv[1:]
    args = [a for a in sys.argv[1:] if a != "--fix"]
    target = args[0] if args else os.path.join(MAP, "map_mask27.png")
    if fix:
        target = sanitize(target)
        print()
    ok = report(target)
    sys.exit(0 if ok else 1)
