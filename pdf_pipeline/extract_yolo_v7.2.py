#!/usr/bin/env python3
"""
V7.2: YOLOv8s 条码检测 + pyzbar 解码 + 空间排序 + 结构化输出

V7.1 → V7.2 升级:
  1. 空间几何重组: 文字块 + 条码块按阅读顺序排列，输出结构化 JSON
  2. 补丁1: 分栏排版检测 (多栏 PDF 正确排序)
  3. 补丁2: 图文重叠过滤 (包含度剔除条码自带文字)
  4. 补丁3: 扫描件分流 (无文本层时 fallback)
  5. pyzbar 解码: 条码原始值写入元数据

用法:
  python3 scripts/extract_yolo_v7.2.py --pdf path/to.pdf --out /tmp/YOLO-V7.2/ [--dpi 300]
"""
import os, sys, json, argparse
from io import BytesIO

import fitz
import numpy as np
from PIL import Image

# ---------- optional imports ----------
import ctypes, ctypes.util, platform

# macOS: monkey-patch find_library so pyzbar can find homebrew-installed zbar
_original_find_library = ctypes.util.find_library

def _patched_find_library(name):
    result = _original_find_library(name)
    if result is None and name == "zbar" and platform.system() == "Darwin":
        for candidate in ["/opt/homebrew/opt/zbar/lib/libzbar.dylib",
                          "/usr/local/opt/zbar/lib/libzbar.dylib"]:
            if os.path.exists(candidate):
                return candidate
    return result

ctypes.util.find_library = _patched_find_library

try:
    from pyzbar.pyzbar import decode as zbar_decode
    HAS_PYZBAR = True
except (ImportError, OSError, Exception):
    HAS_PYZBAR = False
    print("⚠️  pyzbar not available — barcode decoding disabled", file=sys.stderr)

from ultralytics import YOLO

# ---------- paths ----------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "yolov8s-barcode-detection.pt")


# ═══════════════════════════════════════════════════════════════════
# PATCH 1: 分栏排版检测
# ═══════════════════════════════════════════════════════════════════

def detect_columns(text_blocks, page_width, page_height):
    """
    从文字块的 X 轴分布判断是否多栏排版。
    返回: (split_x, num_columns)
      - split_x: 分界线 x 坐标 (单栏时为 None)
      - num_columns: 1 或 2
    """
    if len(text_blocks) < 5:
        return None, 1

    x_centers = sorted((b['x0'] + b['x1']) / 2 for b in text_blocks)

    # 找最大的 X 轴间隙
    max_gap = 0
    gap_idx = 0
    for i in range(len(x_centers) - 1):
        gap = x_centers[i + 1] - x_centers[i]
        if gap > max_gap:
            max_gap = gap
            gap_idx = i

    # 间隙占页面宽度 > 12% → 双栏
    if max_gap > page_width * 0.12:
        split_x = (x_centers[gap_idx] + x_centers[gap_idx + 1]) / 2
        left_count = sum(1 for b in text_blocks if (b['x0'] + b['x1']) / 2 < split_x)
        right_count = len(text_blocks) - left_count
        if left_count >= 2 and right_count >= 2:
            print(f"  📐 双栏检测: split_x={split_x:.0f}pt, 左={left_count}块, 右={right_count}块")
            return split_x, 2

    return None, 1


# ═══════════════════════════════════════════════════════════════════
# PATCH 2: 图文重叠过滤
# ═══════════════════════════════════════════════════════════════════

def containment_ratio(inner_bbox, outer_bbox):
    """inner 被 outer 包含的面积 / inner 总面积"""
    ix0, iy0, ix1, iy1 = inner_bbox
    ox0, oy0, ox1, oy1 = outer_bbox

    overlap_x = max(0, min(ix1, ox1) - max(ix0, ox0))
    overlap_y = max(0, min(iy1, oy1) - max(iy0, oy0))
    overlap_area = overlap_x * overlap_y

    inner_area = (ix1 - ix0) * (iy1 - iy0)
    if inner_area <= 0:
        return 0
    return overlap_area / inner_area


def filter_overlapping_text(text_blocks, barcode_elements):
    """
    剔除与条码区域重叠的文字块 (条码自带的 HRI 文字等)。
    规则:
      a) text 被 barcode 包含度 > 0.5 → 剔除
      b) text 内容 == barcode 解码值 → 剔除
    """
    removed = []
    kept = []

    for tb in text_blocks:
        should_remove = False
        tb_bbox = (tb['x0'], tb['y0'], tb['x1'], tb['y1'])

        for be_ in barcode_elements:
            be_bbox = (be_['x0'], be_['y0'], be_['x1'], be_['y1'])
            cr = containment_ratio(tb_bbox, be_bbox)
            if cr > 0.5:
                should_remove = True
                break

            # 文字内容精确匹配解码值
            raw = be_.get('raw_value', '')
            if raw and tb['text'].strip() == raw.strip():
                should_remove = True
                break

        if should_remove:
            removed.append(tb)
        else:
            kept.append(tb)

    if removed:
        names = [r['text'][:30] for r in removed]
        print(f"  🧹 重叠过滤: 剔除 {len(removed)} 块 → {names}")

    return kept


# ═══════════════════════════════════════════════════════════════════
# PATCH 3: 扫描件分流
# ═══════════════════════════════════════════════════════════════════

def is_scanned_page(page):
    """
    通过 PyMuPDF 文本块数量快速判断是否为扫描件。
    原生 PDF 文本块通常几十到几百; 扫描件为 0 (或极少噪声)。
    """
    blocks = page.get_text("dict")["blocks"]
    text_blocks = [b for b in blocks if b["type"] == 0 and b.get("lines")]
    return len(text_blocks) < 5


# ═══════════════════════════════════════════════════════════════════
# 辅助: PyMuPDF 文本提取
# ═══════════════════════════════════════════════════════════════════

def get_text_blocks(page):
    """
    从 PyMuPDF 提取带坐标的文字块。

    返回: [{text, x0, y0, x1, y1, font_size, font_name}, ...]
    """
    blocks = page.get_text("dict")["blocks"]
    result = []

    for block in blocks:
        if block["type"] != 0:  # 非文本块 (图片)
            continue
        lines = block.get("lines", [])
        if not lines:
            continue

        # 合并 block 内所有 span 文本
        full_text = ""
        font_size = None
        font_name = None
        for line in lines:
            for span in line.get("spans", []):
                full_text += span["text"]
                if font_size is None:
                    font_size = span.get("size", 0)
                    font_name = span.get("font", "")

        full_text = full_text.strip()
        if not full_text or len(full_text) < 2:
            continue

        bbox = block["bbox"]  # [x0, y0, x1, y1]
        result.append({
            'text': full_text,
            'x0': bbox[0],
            'y0': bbox[1],
            'x1': bbox[2],
            'y1': bbox[3],
            'font_size': font_size or 0,
            'font_name': font_name or '',
        })

    return result


# ═══════════════════════════════════════════════════════════════════
# 辅助: pyzbar 解码
# ═══════════════════════════════════════════════════════════════════

def decode_with_pyzbar(img_array, conf_threshold=0):
    """对裁剪后的条码图片执行 pyzbar 解码。返回 (raw_value, type) 或 (None, None)。"""
    if not HAS_PYZBAR:
        return None, None

    try:
        pil_img = Image.fromarray(img_array)
        # 转灰度提升解码率
        if pil_img.mode != 'L':
            pil_img = pil_img.convert('L')
        results = zbar_decode(pil_img)
        if results:
            r = results[0]
            return r.data.decode('utf-8', errors='replace'), r.type
    except Exception:
        pass
    return None, None


# ═══════════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════════

def extract_v72(pdf_path, out_dir, dpi=300, conf=0.25, skip_pages=None):
    if skip_pages is None:
        skip_pages = {1}

    os.makedirs(out_dir, exist_ok=True)
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    model = YOLO(MODEL_PATH)
    doc = fitz.open(pdf_path)
    SCALE = dpi / 72.0

    all_pages = []
    total_elements = 0
    total_barcodes = 0

    for pn in range(1, len(doc) + 1):
        if pn in skip_pages:
            continue

        page = doc[pn - 1]
        pr = page.rect
        page_w = pr.width
        page_h = pr.height

        # ============ 全页渲染 ============
        pix = page.get_pixmap(dpi=dpi)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        PAGE_W, PAGE_H = pix.width, pix.height

        # ============ PATCH 3: 扫描件检测 ============
        scanned = is_scanned_page(page)
        if scanned:
            print(f"Page {pn}: 📄 扫描件模式")

        # ============ 文字提取 ============
        text_blocks = get_text_blocks(page) if not scanned else []
        # 过滤页眉页脚 (页面顶部/底部 5% 区域的短文本)
        header_margin = page_h * 0.05
        footer_margin = page_h * 0.95
        text_blocks_filtered = []
        for tb in text_blocks:
            is_header = tb['y1'] < header_margin and len(tb['text']) < 30
            is_footer = tb['y0'] > footer_margin and len(tb['text']) < 30
            if not is_header and not is_footer:
                text_blocks_filtered.append(tb)
        text_blocks = text_blocks_filtered

        # ============ 阶段1: 全页 YOLO ============
        results = model(img, conf=conf, verbose=False)[0]
        detections = []

        for box_data in results.boxes:
            x1, y1, x2, y2 = map(int, box_data.xyxy[0].tolist())
            cls = int(box_data.cls[0].item())
            detections.append({
                'px_box': (x1, y1, x2, y2),
                'cls': cls,
                'conf': float(box_data.conf[0].item()),
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

            already_detected = any(
                max(0, min(d['px_box'][2], embed_px[2]) - max(d['px_box'][0], embed_px[0])) > 10
                and max(0, min(d['px_box'][3], embed_px[3]) - max(d['px_box'][1], embed_px[1])) > 10
                for d in detections
            )
            if already_detected:
                continue

            try:
                embed_img = Image.open(BytesIO(base['image']))
                if embed_img.mode == 'RGBA':
                    bg = Image.new('RGB', embed_img.size, (255, 255, 255))
                    bg.paste(embed_img, mask=embed_img.split()[3])
                    embed_img = bg
                elif embed_img.mode != 'RGB':
                    embed_img = embed_img.convert('RGB')
                embed_np = np.array(embed_img)
                embed_results = model(embed_np, conf=conf, verbose=False)[0]
            except Exception:
                continue

            for box_data in embed_results.boxes:
                ex1, ey1, ex2, ey2 = map(int, box_data.xyxy[0].tolist())
                cls = int(box_data.cls[0].item())
                emb_conf = float(box_data.conf[0].item())

                scale_x = (r.x1 - r.x0) / max(w, 1)
                scale_y = (r.y1 - r.y0) / max(h, 1)

                px_x1 = int((r.x0 + ex1 * scale_x) * SCALE)
                px_y1 = int((r.y0 + ey1 * scale_y) * SCALE)
                px_x2 = int((r.x0 + ex2 * scale_x) * SCALE)
                px_y2 = int((r.y0 + ey2 * scale_y) * SCALE)

                embed_full_px = (int(r.x0 * SCALE), int(r.y0 * SCALE),
                                int(r.x1 * SCALE), int(r.y1 * SCALE))

                detections.append({
                    'px_box': (px_x1, px_y1, px_x2, px_y2),
                    'px_box_full': embed_full_px,
                    'cls': cls,
                    'conf': emb_conf,
                    'source': 'embed',
                })

        if not detections and not text_blocks:
            continue

        # ============ 全页检测与嵌入图对齐 ============
        for det in detections:
            if det['source'] != 'full':
                continue
            dx1, dy1, dx2, dy2 = det['px_box']
            dw, dh = dx2 - dx1, dy2 - dy1
            for img_info in page.get_images(full=True):
                xref = img_info[0]
                rects = page.get_image_rects(xref)
                if not rects:
                    continue
                r = rects[0]
                fx0, fy0 = int(r.x0 * SCALE), int(r.y0 * SCALE)
                fx1, fy1 = int(r.x1 * SCALE), int(r.y1 * SCALE)
                fw, fh = fx1 - fx0, fy1 - fy0
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

        # ============ 构建条码元素列表 ============
        barcode_elements = []

        for det_idx, det in enumerate(detections):
            x1_px, y1_px, x2_px, y2_px = det['px_box']
            cls = det['cls']
            class_name = model.names[cls]

            # 包含嵌入图边界
            if 'px_box_full' in det and det['source'] == 'embed':
                fx1, fy1_, fx2, fy2_ = det['px_box_full']
                x1_px = min(x1_px, fx1)
                y1_px = min(y1_px, fy1_)
                x2_px = max(x2_px, fx2)
                y2_px = max(y2_px, fy2_)

            bx0 = x1_px / SCALE
            by0 = y1_px / SCALE
            bx1 = x2_px / SCALE
            by1 = y2_px / SCALE
            bw = bx1 - bx0
            bh = by1 - by0

            if bw < 5 or bh < 5:
                continue

            # ---------- 误检过滤 ----------
            if det['source'] == 'full' and class_name == 'barcode' and bw < 55:
                print(f"  [SKIP] {class_name:7s} too small ({bw:.0f}×{bh:.0f}pt) → false positive")
                continue
            if det['source'] in ('embed', 'full+embed') and 'px_box_full' in det:
                efx0, efy0, efx1, efy1 = det['px_box_full']
                embed_w = (efx1 - efx0) / SCALE
                embed_h = (efy1 - efy0) / SCALE
                if embed_w > 120 or embed_h > 120:
                    coverage = (bw * bh) / (embed_w * embed_h) * 100 if embed_w * embed_h > 0 else 0
                    if coverage < 30:
                        print(f"  [SKIP] {class_name:7s} large embed ({embed_w:.0f}×{embed_h:.0f}pt, {coverage:.0f}%) → icon/footer")
                        continue

            # ---------- pyzbar 解码 ----------
            px0 = max(0, int(bx0 * SCALE))
            py0 = max(0, int(by0 * SCALE))
            px1 = min(PAGE_W, int(bx1 * SCALE))
            py1 = min(PAGE_H, int(by1 * SCALE))

            crop = img[py0:py1, px0:px1]
            raw_value, zbar_type = decode_with_pyzbar(crop) if crop.size > 0 else (None, None)

            # ---------- 裁剪保存 ----------
            # 扩展裁剪框包含附近标注文字
            PAD = 5
            crop_x0 = max(0, bx0 - PAD)
            crop_y0 = max(0, by0 - PAD)
            crop_x1 = min(page_w, bx1 + PAD)
            crop_y1 = min(page_h, by1 + PAD)

            ocr_label = ""
            for tb in text_blocks:
                tx0, ty0, tx1, ty1 = tb['x0'], tb['y0'], tb['x1'], tb['y1']
                overlap_x = max(0, min(bx1, tx1) - max(bx0, tx0))
                if overlap_x < max(bw * 0.3, 10):
                    continue
                # 文字在条码下方 0-60pt 范围内
                if 0 < ty0 - by1 < 60:
                    crop_y1 = max(crop_y1, ty1 + PAD)
                    ocr_label = tb['text']
                    break
                # 文字在条码上方 0-30pt 范围内
                if 0 < by0 - ty1 < 30:
                    crop_y0 = min(crop_y0, ty0 - PAD)
                    ocr_label = tb['text']
                    break

            # 保存图片
            fname = f"bc_page{pn}_{det_idx+1:02d}.png"
            fpath = os.path.join(img_dir, fname)

            save_px0 = max(0, int(crop_x0 * SCALE))
            save_py0 = max(0, int(crop_y0 * SCALE))
            save_px1 = min(PAGE_W, int(crop_x1 * SCALE))
            save_py1 = min(PAGE_H, int(crop_y1 * SCALE))

            if save_px1 - save_px0 >= 10 and save_py1 - save_py0 >= 10:
                save_crop = img[save_py0:save_py1, save_px0:save_px1]
                Image.fromarray(save_crop).save(fpath)

            barcode_elements.append({
                'type': 'barcode',
                'barcode_id': fname.replace('.png', ''),
                'class': class_name,
                'confidence': round(det['conf'], 3),
                'raw_value': raw_value or '',
                'zbar_type': zbar_type or '',
                'ocr_label': ocr_label,
                'image_path': f"images/{fname}",
                'x0': bx0,
                'y0': by0,
                'x1': bx1,
                'y1': by1,
                'source': det['source'],
            })

        # ============ PATCH 2: 重叠过滤 ============
        text_blocks = filter_overlapping_text(text_blocks, barcode_elements)

        # ============ PATCH 1: 分栏检测 ============
        split_x, num_cols = detect_columns(text_blocks, page_w, page_h)

        # ============ 空间排序 ============
        # 构建统一元素列表
        all_elements = []

        for tb in text_blocks:
            all_elements.append({
                'type': 'text',
                'content': tb['text'],
                'x0': tb['x0'],
                'y0': tb['y0'],
                'x1': tb['x1'],
                'y1': tb['y1'],
                'font_size': tb['font_size'],
            })

        for be_ in barcode_elements:
            all_elements.append({
                'type': 'barcode',
                'barcode_id': be_['barcode_id'],
                'class': be_['class'],
                'confidence': be_['confidence'],
                'raw_value': be_['raw_value'],
                'zbar_type': be_['zbar_type'],
                'ocr_label': be_['ocr_label'],
                'image_path': be_['image_path'],
                'x0': be_['x0'],
                'y0': be_['y0'],
                'x1': be_['x1'],
                'y1': be_['y1'],
                'source': be_['source'],
            })

        # 排序逻辑
        if num_cols == 2 and split_x is not None:
            # 双栏: 分左右桶，各桶内按 Y 排序，然后按 Y 波段交错合并
            left = [e for e in all_elements if (e['x0'] + e['x1']) / 2 < split_x]
            right = [e for e in all_elements if (e['x0'] + e['x1']) / 2 >= split_x]

            left.sort(key=lambda e: e['y0'])
            right.sort(key=lambda e: e['y0'])

            # 按 Y 波段合并: 同一波段内左栏优先
            merged = []
            li = ri = 0
            while li < len(left) and ri < len(right):
                # 如果右栏元素在当前左栏元素上方一个波段，先放右
                if right[ri]['y0'] < left[li]['y0'] - 20:
                    merged.append(right[ri])
                    ri += 1
                else:
                    # 把同一 Y 波段的所有元素放一起 (左优先)
                    band_top = left[li]['y0']
                    while li < len(left) and left[li]['y0'] < band_top + left[li]['y1'] - left[li]['y0'] + 10:
                        merged.append(left[li])
                        li += 1
                        if li >= len(left):
                            break
                    while ri < len(right) and right[ri]['y0'] < band_top + 30:
                        merged.append(right[ri])
                        ri += 1
            merged.extend(left[li:])
            merged.extend(right[ri:])
            all_elements = merged
        else:
            # 单栏: 按 Y 排序
            all_elements.sort(key=lambda e: e['y0'])

        # ============ 结构化 JSON 输出 ============
        elements_out = []
        for idx, elem in enumerate(all_elements):
            if elem['type'] == 'text':
                elements_out.append({
                    'type': 'text',
                    'content': elem['content'],
                    'bbox': [
                        round(elem['x0'], 1),
                        round(elem['y0'], 1),
                        round(elem['x1'], 1),
                        round(elem['y1'], 1),
                    ],
                })
            else:
                elements_out.append({
                    'type': 'barcode',
                    'barcode_id': elem['barcode_id'],
                    'barcode_raw_value': elem['raw_value'],
                    'image_path': elem['image_path'],
                    'ocr_label': elem['ocr_label'],
                    'class': elem['class'],
                    'confidence': elem['confidence'],
                    'zbar_type': elem['zbar_type'],
                    'bbox': [
                        round(elem['x0'], 1),
                        round(elem['y0'], 1),
                        round(elem['x1'], 1),
                        round(elem['y1'], 1),
                    ],
                })

        page_data = {
            'page': pn,
            'width': round(page_w, 1),
            'height': round(page_h, 1),
            'is_scanned': scanned,
            'columns': num_cols,
            'element_count': len(elements_out),
            'elements': elements_out,
        }
        all_pages.append(page_data)

        n_barcodes = sum(1 for e in elements_out if e['type'] == 'barcode')
        n_decoded = sum(1 for e in elements_out if e['type'] == 'barcode' and e.get('barcode_raw_value'))
        print(f"  Page {pn}: {len(elements_out)} elements ({len(elements_out) - n_barcodes} text + {n_barcodes} barcode, {n_decoded} decoded) | scanned={scanned} | cols={num_cols}")

        total_elements += len(elements_out)
        total_barcodes += n_barcodes

    doc.close()

    # ============ 输出 ============
    output = {
        'pdf': os.path.basename(pdf_path),
        'version': '7.2',
        'total_pages': len(all_pages),
        'total_elements': total_elements,
        'total_barcodes': total_barcodes,
        'pages': all_pages,
    }

    idx_path = os.path.join(out_dir, "index.json")
    with open(idx_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n✅ {os.path.basename(pdf_path)}: {total_elements} elements → {out_dir}")
    print(f"   index.json + {total_barcodes} barcode images")
    return output


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V7.2: YOLO barcode + spatial sort + structured output")
    parser.add_argument("--pdf", required=True, help="Input PDF path")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--dpi", type=int, default=300, help="Render DPI (default: 300)")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--skip", type=str, default="1", help="Comma-separated pages to skip (default: 1)")
    args = parser.parse_args()

    skip = set(int(p.strip()) for p in args.skip.split(",") if p.strip())
    extract_v72(args.pdf, args.out, dpi=args.dpi, conf=args.conf, skip_pages=skip)
