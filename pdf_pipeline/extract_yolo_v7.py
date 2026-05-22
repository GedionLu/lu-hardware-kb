#!/usr/bin/env python3
"""
V7.1: YOLOv8s 条码/qrcode 检测 + 下方文字扩展裁剪 (优化版)

V7 → V7.1 优化:
  1. 误检过滤: 全页小尺寸检测 + 超大嵌入图检测
  2. 共享文字去重: 多个码命中同段文字时只分配给最近的一个

用法:
  python3 scripts/extract_yolo_v7.py --pdf path/to.pdf --out /tmp/YOLO-V7/ [--dpi 300]
"""
import os, sys, json, argparse
import fitz
import numpy as np
from PIL import Image
from ultralytics import YOLO

# ---------- paths ----------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../weights/yolov8s-barcode-detection.pt")


def extract(pdf_path, out_dir, dpi=300, conf=0.25, skip_pages=None):
    if skip_pages is None:
        skip_pages = {1}

    os.makedirs(out_dir, exist_ok=True)
    model = YOLO(MODEL_PATH)
    doc = fitz.open(pdf_path)
    SCALE = dpi / 72.0

    image_index = []

    for pn in range(1, len(doc) + 1):
        if pn in skip_pages:
            continue

        page = doc[pn - 1]
        pr = page.rect

        # ============ 全页渲染 ============
        pix = page.get_pixmap(dpi=dpi)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        PAGE_W, PAGE_H = pix.width, pix.height

        # ============ 阶段1: 全页 YOLO ============
        results = model(img, conf=conf, verbose=False)[0]
        detections = []

        for box_data in results.boxes:
            x1, y1, x2, y2 = map(int, box_data.xyxy[0].tolist())
            cls = int(box_data.cls[0].item())
            detections.append({
                'px_box': (x1, y1, x2, y2),
                'cls': cls,
                'conf': box_data.conf[0].item(),
                'source': 'full',
            })

        # ============ 阶段2: 嵌入图兜底 ============
        for img_info in page.get_images(full=True):
            xref = img_info[0]
            base = doc.extract_image(xref)
            w, h = base['width'], base['height']

            if w * h > PAGE_W * PAGE_H * 0.5:
                continue

            rects = page.get_image_rects(xref)
            if not rects:
                continue
            r = rects[0]

            embed_px = (int(r.x0 * SCALE), int(r.y0 * SCALE),
                        int(r.x1 * SCALE), int(r.y1 * SCALE))

            already_detected = False
            for d in detections:
                dx1, dy1, dx2, dy2 = d['px_box']
                ox = max(0, min(dx2, embed_px[2]) - max(dx1, embed_px[0]))
                oy = max(0, min(dy2, embed_px[3]) - max(dy1, embed_px[1]))
                if ox > 10 and oy > 10:
                    already_detected = True
                    break
            if already_detected:
                continue

            try:
                from io import BytesIO
                embed_img = Image.open(BytesIO(base['image']))
                if embed_img.mode == 'RGBA':
                    bg = Image.new('RGB', embed_img.size, (255, 255, 255))
                    bg.paste(embed_img, mask=embed_img.split()[3])
                    embed_img = bg
                elif embed_img.mode != 'RGB':
                    embed_img = embed_img.convert('RGB')
                embed_np = np.array(embed_img)
                embed_results = model(embed_np, conf=conf, verbose=False)[0]
            except Exception as e:
                continue

            for box_data in embed_results.boxes:
                x1, y1, x2, y2 = map(int, box_data.xyxy[0].tolist())
                cls = int(box_data.cls[0].item())
                emb_conf = box_data.conf[0].item()

                scale_x = (r.x1 - r.x0) / max(w, 1)
                scale_y = (r.y1 - r.y0) / max(h, 1)

                px_x1 = int((r.x0 + x1 * scale_x) * SCALE)
                px_y1 = int((r.y0 + y1 * scale_y) * SCALE)
                px_x2 = int((r.x0 + x2 * scale_x) * SCALE)
                px_y2 = int((r.y0 + y2 * scale_y) * SCALE)

                embed_full_px = (int(r.x0 * SCALE), int(r.y0 * SCALE),
                                int(r.x1 * SCALE), int(r.y1 * SCALE))

                detections.append({
                    'px_box': (px_x1, px_y1, px_x2, px_y2),
                    'px_box_full': embed_full_px,
                    'cls': cls,
                    'conf': emb_conf,
                    'source': 'embed',
                })

        if not detections:
            continue

        # ============ 全页检测与嵌入图对齐 ============
        for det in detections:
            if det['source'] != 'full':
                continue
            dx1, dy1, dx2, dy2 = det['px_box']
            dw = dx2 - dx1
            dh = dy2 - dy1
            for img_info in page.get_images(full=True):
                xref = img_info[0]
                rects = page.get_image_rects(xref)
                if not rects:
                    continue
                r = rects[0]
                fx0, fy0, fx1, fy1 = int(r.x0*SCALE), int(r.y0*SCALE), int(r.x1*SCALE), int(r.y1*SCALE)
                fw = fx1 - fx0
                fh = fy1 - fy0
                ox = max(0, min(dx2, fx1) - max(dx1, fx0))
                oy = max(0, min(dy2, fy1) - max(dy1, fy0))
                if ox > 10 and oy > 10:
                    if fw > dw * 2 or fh > dh * 2:
                        continue
                    det['px_box'] = (min(dx1, fx0), min(dy1, fy0),
                                    max(dx2, fx1), max(dy2, fy1))
                    det['source'] = 'full+embed'
                    break

        # ============ 去重: 空间聚类 ============
        detections.sort(key=lambda d: d['conf'], reverse=True)
        kept = []
        for d in detections:
            dx1, dy1, dx2, dy2 = d['px_box']
            dh = dy2 - dy1
            is_dup = False
            for k in kept:
                if d['cls'] != k['cls']:
                    continue
                kx1, ky1, kx2, ky2 = k['px_box']
                oy = max(0, min(dy2, ky2) - max(dy1, ky1))
                max_h = max(dh, ky2 - ky1)
                y_overlap_ok = max_h > 0 and oy / max_h > 0.6
                contained = (dx1 >= kx1 and dy1 >= ky1 and dx2 <= kx2 and dy2 <= ky2) or \
                            (kx1 >= dx1 and ky1 >= dy1 and kx2 <= dx2 and ky2 <= dy2)
                if y_overlap_ok or contained:
                    is_dup = True
                    break
            if not is_dup:
                kept.append(d)
        detections = kept

        # ============ 获取文字块 ============
        blocks = page.get_text("blocks")
        text_blocks = [(b[0], b[1], b[2], b[3], b[4].strip())
                       for b in blocks if b[4].strip() and len(b[4].strip()) > 2]

        print(f"Page {pn}: {len(detections)} detections, {len(text_blocks)} text blocks")

        # ============ 对每个检测找文字 + 收集分配记录 ============
        assignments = []  # (det_idx, nearest_info, is_above)

        for det_idx, det in enumerate(detections):
            x1_px, y1_px, x2_px, y2_px = det['px_box']
            if 'px_box_full' in det and det['source'] == 'embed':
                fx1, fy1, fx2, fy2 = det['px_box_full']
                x1_px = min(x1_px, fx1)
                y1_px = min(y1_px, fy1)
                x2_px = max(x2_px, fx2)
                y2_px = max(y2_px, fy2)

            cls = det['cls']
            conf_val = det['conf']
            source = det['source']
            class_name = model.names[cls]

            bx0 = x1_px / SCALE
            by0 = y1_px / SCALE
            bx1 = x2_px / SCALE
            by1 = y2_px / SCALE
            bw = bx1 - bx0
            bh = by1 - by0

            if bw < 5 or bh < 5:
                continue

            # ---------- 误检过滤 ----------
            # 全页检测到的barcode框宽 < 55pt → 文本噪声误检
            if source == 'full' and class_name == 'barcode' and bw < 55:
                print(f"  [SKIP] {class_name:7s} too small ({bw:.0f}×{bh:.0f}pt) → false positive")
                continue
            # 嵌入图过大的检测 → 装饰元素
            if source in ('embed', 'full+embed') and 'px_box_full' in det:
                efx0, efy0, efx1, efy1 = det['px_box_full']
                embed_w = (efx1 - efx0) / SCALE
                embed_h = (efy1 - efy0) / SCALE
                if (embed_w > 120 or embed_h > 120):
                    coverage = (bw * bh) / (embed_w * embed_h) * 100 if embed_w * embed_h > 0 else 0
                    if coverage < 30:
                        print(f"  [SKIP] {class_name:7s} large embed ({embed_w:.0f}×{embed_h:.0f}pt, {coverage:.0f}%) → icon/footer")
                        continue

            # ============ 找下方最近文字 ============
            nearest = None
            ndist = 9999.0

            for tx0, ty0, tx1, ty1, text in text_blocks:
                overlap_x = max(0, min(bx1, tx1) - max(bx0, tx0))
                gap_left = bx0 - tx1
                gap_right = tx0 - bx1

                if overlap_x < 1:
                    if not (0 < gap_left < 5 or 0 < gap_right < 5):
                        continue

                text_mid_y = (ty0 + ty1) / 2
                det_mid_y = (by0 + by1) / 2

                if text_mid_y < det_mid_y:
                    continue

                dist = text_mid_y - by1

                deep_inside = overlap_x > 5 and ty1 < by0 + (by1 - by0) * 0.4
                if deep_inside:
                    continue

                if dist < ndist:
                    nearest = {
                        'rect': (tx0, ty0, tx1, ty1),
                        'text': text,
                        'dist': dist,
                    }
                    ndist = dist

            # ============ 扩展裁剪框 ============
            PAD = 5
            MAX_H_EXPAND = 30

            crop_x0 = max(0, bx0 - PAD)
            crop_y0 = max(0, by0 - PAD)
            crop_x1 = min(pr.width, bx1 + PAD)
            crop_y1 = min(pr.height, by1 + PAD)

            label_text = ""
            is_above = False

            if nearest and ndist < 150:
                tx0, ty0, tx1, ty1 = nearest['rect']
                new_x0 = min(crop_x0, tx0 - PAD)
                new_x1 = max(crop_x1, tx1 + PAD)
                if bx0 - new_x0 > MAX_H_EXPAND:
                    new_x0 = bx0 - MAX_H_EXPAND
                if new_x1 - bx1 > MAX_H_EXPAND:
                    new_x1 = bx1 + MAX_H_EXPAND
                crop_x0 = max(0, new_x0)
                crop_x1 = min(pr.width, new_x1)
                crop_y0 = max(0, min(crop_y0, ty0 - PAD))
                crop_y1 = min(pr.height, max(crop_y1, ty1 + PAD))
                label_text = nearest['text']
            elif not nearest:
                # 向上看 (标题可能在上方)
                for tx0, ty0, tx1, ty1, text in text_blocks:
                    overlap_x = max(0, min(bx1, tx1) - max(bx0, tx0))
                    if overlap_x < 1:
                        gap_left = bx0 - tx1
                        gap_right = tx0 - bx1
                        if not (0 < gap_left < 15 or 0 < gap_right < 15):
                            continue
                    if ty1 > by0 + 5:
                        continue
                    dist = by0 - ty1
                    if 0 < dist < 40 and dist < ndist:
                        nearest = {
                            'rect': (tx0, ty0, tx1, ty1),
                            'text': text,
                            'dist': dist,
                        }
                        ndist = dist
                if nearest:
                    is_above = True
                    tx0, ty0, tx1, ty1 = nearest['rect']
                    new_x0 = min(crop_x0, tx0 - PAD)
                    new_x1 = max(crop_x1, tx1 + PAD)
                    if bx0 - new_x0 > MAX_H_EXPAND:
                        new_x0 = bx0 - MAX_H_EXPAND
                    if new_x1 - bx1 > MAX_H_EXPAND:
                        new_x1 = bx1 + MAX_H_EXPAND
                    crop_x0 = max(0, new_x0)
                    crop_x1 = min(pr.width, new_x1)
                    crop_y0 = max(0, min(crop_y0, ty0 - PAD))
                    crop_y1 = min(pr.height, max(crop_y1, ty1 + PAD))
                    label_text = nearest['text']

            assignments.append({
                'det_idx': det_idx,
                'det': det,
                'class_name': class_name,
                'conf_val': conf_val,
                'source': source,
                'bx0': bx0, 'by0': by0, 'bx1': bx1, 'by1': by1,
                'bw': bw, 'bh': bh,
                'PAD': PAD,
                'MAX_H_EXPAND': MAX_H_EXPAND,
                'crop_x0': crop_x0, 'crop_y0': crop_y0,
                'crop_x1': crop_x1, 'crop_y1': crop_y1,
                'label_text': label_text,
                'nearest': nearest,
                'is_above': is_above,
                'PAGE_W': PAGE_W, 'PAGE_H': PAGE_H,
            })

        # ============ 共享文字去重 ============
        # 如果多个码命中同一段文字（下方或上方），只分配给最近的一个
        text_usage = {}  # (tx0, ty0, tx1, ty1) → [(dist, assignments_idx)]
        for a_idx, a in enumerate(assignments):
            if a['nearest']:
                key = a['nearest']['rect']
                if key not in text_usage:
                    text_usage[key] = []
                text_usage[key].append((a['nearest']['dist'], a_idx))

        shared_texts = {k: v for k, v in text_usage.items() if len(v) > 1}
        for rect, users in shared_texts.items():
            # 只保留最近的那个，其余清空文字和裁剪扩展
            users.sort(key=lambda x: x[0])  # 按dist排序，最小的最近
            closest_idx = users[0][1]
            for _, a_idx in users[1:]:
                a = assignments[a_idx]
                if a['label_text']:
                    print(f"  [DEDUP] {a['class_name']:7s} shared text \"{a['label_text'][:30]}...\" → cleared")
                a['label_text'] = ''
                a['nearest'] = None
                a['crop_x0'] = max(0, a['bx0'] - a['PAD'])
                a['crop_y0'] = max(0, a['by0'] - a['PAD'])
                a['crop_x1'] = min(pr.width, a['bx1'] + a['PAD'])
                a['crop_y1'] = min(pr.height, a['by1'] + a['PAD'])

        # ============ 渲染输出 + 最终过滤 ============
        for a in assignments:
            if not a['label_text'] and a['nearest'] and a['nearest']['dist'] >= 150:
                continue

            # 标签包含页眉页脚模式 → 误检
            if 'User Guide' in a['label_text'] and a['label_text'].strip()[-1].isdigit():
                print(f"  [SKIP] {a['class_name']:7s} page number text → false positive")
                continue

            px0 = max(0, int(a['crop_x0'] * SCALE))
            py0 = max(0, int(a['crop_y0'] * SCALE))
            px1 = min(a['PAGE_W'], int(a['crop_x1'] * SCALE))
            py1 = min(a['PAGE_H'], int(a['crop_y1'] * SCALE))

            if px1 - px0 < 10 or py1 - py0 < 10:
                continue

            crop_img = img[py0:py1, px0:px1]
            crop_h, crop_w = crop_img.shape[:2]

            x1_px = int(a['bx0'] * SCALE)
            y1_px = int(a['by0'] * SCALE)
            fname = f"page{pn}_{a['class_name']}_{x1_px}_{y1_px}.png"
            fpath = os.path.join(out_dir, fname)
            Image.fromarray(crop_img).save(fpath)

            image_index.append({
                'file': fname,
                'page': pn,
                'class': a['class_name'],
                'confidence': round(a['conf_val'], 3),
                'label': a['label_text'][:120],
                'source': a['source'],
                'crop_px': [px0, py0, px1, py1],
                'crop_pt': [round(a['crop_x0'], 1), round(a['crop_y0'], 1),
                            round(a['crop_x1'], 1), round(a['crop_y1'], 1)],
                'detect_pt': [round(a['bx0'], 1), round(a['by0'], 1),
                              round(a['bx1'], 1), round(a['by1'], 1)],
            })

            print(f"  [{a['source']:5s}] {a['class_name']:7s} {fname:45s} crop={crop_w:>4}×{crop_h:<4}  \"{a['label_text'][:50]}\"")

    doc.close()

    idx_path = os.path.join(out_dir, "index.json")
    with open(idx_path, 'w', encoding='utf-8') as f:
        json.dump(image_index, f, ensure_ascii=False, indent=2)

    print(f"\n✅ {os.path.basename(pdf_path)}: {len(image_index)} images → {out_dir}")
    return image_index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V7.1: YOLO barcode/qrcode + text expansion")
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--skip", type=str, default="1")
    args = parser.parse_args()

    skip = set(int(p.strip()) for p in args.skip.split(",") if p.strip())
    extract(args.pdf, args.out, dpi=args.dpi, conf=args.conf, skip_pages=skip)
