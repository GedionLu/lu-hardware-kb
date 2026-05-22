#!/usr/bin/env python3
"""
V7.3: YOLO 条码 + Plan B 图片提取 + 统一命名 + 统一输出格式

V7.2 → V7.3 升级:
  1. Plan B: PyMuPDF 提取非条码图片 (产品图/连接图) → type: "image"
  2. 统一命名: {model}_{category}_p{page}_s{idx}_{value}_{hash}.png
  3. 统一输出: kb-images/pdf/{barcode,qrcode,image}/ + image_index.json
  4. 型号提取: 从 PDF 文件名自动提取产品型号
  5. 对齐 DOCX image_index.json 格式

用法:
  python3 pdf_pipeline/extract_yolo_v7.3.py \\
    --pdf "19Series/197x/1972-EN-QS-01 Rev A.pdf" \\
    --out kb-images/pdf/ \\
    [--dpi 300] [--conf 0.25] [--skip 1]
"""
import os, sys, json, argparse, hashlib, re
from io import BytesIO

import fitz
import numpy as np
from PIL import Image

# ── macOS zbar 路径补丁 ──
import ctypes, ctypes.util, platform

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
except Exception:
    HAS_PYZBAR = False
    print("⚠️  pyzbar not available — barcode decoding disabled", file=sys.stderr)

from ultralytics import YOLO

# ── 路径 ──
SELF_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SELF_DIR, "models", "yolov8s-barcode-detection.pt")

# ── 型号提取 ──
MODEL_RE = re.compile(r'(\d{3,4}[A-Za-z]?(?:[A-Za-z]{2})?)')  # 1972, 1970, OH430, HH760

def extract_model_from_filename(pdf_path):
    """从 PDF 文件名提取型号，如 1972-EN-QS-01 → 1972"""
    name = os.path.splitext(os.path.basename(pdf_path))[0]
    matches = MODEL_RE.findall(name)
    # 排除明显不是型号的数字 (如年份 2025, 2026)
    models = [m for m in matches if not m.startswith('202')]
    if models:
        # 优先取最长的匹配 (如 HH762 > HH)
        return max(models, key=lambda x: (len(x), x))
    return "unknown"

def extract_series(model):
    """从型号推断系列，如 1972 → 19Series"""
    if not model or model == "unknown":
        return "unknown"
    # 取前两位或字母前缀
    prefix = re.match(r'([A-Za-z]*\d{1,2})', model)
    if prefix:
        p = prefix.group(1)
        if p[0].isdigit():
            return f"{p[:2]}Series"
        else:
            return f"{p}系列"
    return f"{model}系列"

# ── 哈希 ──
def img_hash(img_array):
    """图片字节 MD5 前 6 位"""
    return hashlib.md5(img_array.tobytes()).hexdigest()[:6]

def safe_filename(text, maxlen=20):
    """截断文本为安全文件名片段"""
    if not text:
        return "X"
    # 去掉特殊字符
    clean = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '', text)
    clean = clean.strip().replace(' ', '_')
    return clean[:maxlen] if clean else "X"


# ═══════════════════════════════════════════════════════════════════
# PATCH 1: 分栏排版检测
# ═══════════════════════════════════════════════════════════════════

def detect_columns(text_blocks, page_width, page_height):
    if len(text_blocks) < 5:
        return None, 1
    x_centers = sorted((b['x0'] + b['x1']) / 2 for b in text_blocks)
    max_gap, gap_idx = 0, 0
    for i in range(len(x_centers) - 1):
        gap = x_centers[i + 1] - x_centers[i]
        if gap > max_gap:
            max_gap, gap_idx = gap, i
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
    ix0, iy0, ix1, iy1 = inner_bbox
    ox0, oy0, ox1, oy1 = outer_bbox
    ox_w = max(0, min(ix1, ox1) - max(ix0, ox0))
    oy_h = max(0, min(iy1, oy1) - max(iy0, oy0))
    overlap_area = ox_w * oy_h
    inner_area = (ix1 - ix0) * (iy1 - iy0)
    return overlap_area / inner_area if inner_area > 0 else 0


def filter_overlapping_text(text_blocks, barcode_elements):
    kept, removed = [], []
    for tb in text_blocks:
        remove = False
        tb_bbox = (tb['x0'], tb['y0'], tb['x1'], tb['y1'])
        for be_ in barcode_elements:
            cr = containment_ratio(tb_bbox, (be_['x0'], be_['y0'], be_['x1'], be_['y1']))
            if cr > 0.5:
                remove = True; break
            raw = be_.get('raw_value', '')
            if raw and tb['text'].strip() == raw.strip():
                remove = True; break
        (removed if remove else kept).append(tb)
    if removed:
        print(f"  🧹 重叠过滤: 剔除 {len(removed)} 块 → {[r['text'][:30] for r in removed]}")
    return kept


# ═══════════════════════════════════════════════════════════════════
# PATCH 3: 扫描件分流
# ═══════════════════════════════════════════════════════════════════

def is_scanned_page(page):
    blocks = page.get_text("dict")["blocks"]
    text_blocks = [b for b in blocks if b["type"] == 0 and b.get("lines")]
    return len(text_blocks) < 5


# ═══════════════════════════════════════════════════════════════════
# 辅助: PyMuPDF 文本提取
# ═══════════════════════════════════════════════════════════════════

def get_text_blocks(page):
    blocks = page.get_text("dict")["blocks"]
    result = []
    for block in blocks:
        if block["type"] != 0:
            continue
        lines = block.get("lines", [])
        if not lines:
            continue
        full_text = ""
        font_size = None
        for line in lines:
            for span in line.get("spans", []):
                full_text += span["text"]
                if font_size is None:
                    font_size = span.get("size", 0)
        full_text = full_text.strip()
        if not full_text or len(full_text) < 2:
            continue
        bbox = block["bbox"]
        result.append({
            'text': full_text, 'x0': bbox[0], 'y0': bbox[1],
            'x1': bbox[2], 'y1': bbox[3], 'font_size': font_size or 0,
        })
    return result


# ═══════════════════════════════════════════════════════════════════
# 辅助: pyzbar 解码
# ═══════════════════════════════════════════════════════════════════

def decode_with_pyzbar(img_array):
    if not HAS_PYZBAR:
        return None, None
    try:
        pil_img = Image.fromarray(img_array)
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

def extract_v73(pdf_path, out_dir, dpi=300, conf=0.25, skip_pages=None):
    if skip_pages is None:
        skip_pages = {1}

    model_name = extract_model_from_filename(pdf_path)
    series_name = extract_series(model_name)

    os.makedirs(out_dir, exist_ok=True)
    for cat in ['barcode', 'qrcode', 'image']:
        os.makedirs(os.path.join(out_dir, cat), exist_ok=True)

    model = YOLO(MODEL_PATH)
    doc = fitz.open(pdf_path)
    SCALE = dpi / 72.0

    all_pages = []
    image_index = []  # 统一 image_index 格式
    total_elements = 0
    total_barcodes = 0
    image_counter = 0

    print(f"📄 {os.path.basename(pdf_path)}  → 型号={model_name} 系列={series_name}\n")

    for pn in range(1, len(doc) + 1):
        if pn in skip_pages:
            continue

        page = doc[pn - 1]
        pr = page.rect
        page_w, page_h = pr.width, pr.height

        # ── 渲染 ──
        pix = page.get_pixmap(dpi=dpi)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        PAGE_W, PAGE_H = pix.width, pix.height

        # ── PATCH 3: 扫描件 ──
        scanned = is_scanned_page(page)
        if scanned:
            print(f"Page {pn}: 📄 扫描件模式")

        # ── 文字 ──
        text_blocks = get_text_blocks(page) if not scanned else []
        header_margin = page_h * 0.05
        footer_margin = page_h * 0.95
        text_blocks = [tb for tb in text_blocks
                       if not ((tb['y1'] < header_margin or tb['y0'] > footer_margin)
                               and len(tb['text']) < 30)]

        # ── 阶段1: 全页 YOLO ──
        results = model(img, conf=conf, verbose=False)[0]
        detections = []
        for box_data in results.boxes:
            x1, y1, x2, y2 = map(int, box_data.xyxy[0].tolist())
            cls = int(box_data.cls[0].item())
            detections.append({
                'px_box': (x1, y1, x2, y2), 'cls': cls,
                'conf': float(box_data.conf[0].item()), 'source': 'full',
            })

        # ── 阶段2: 嵌入图兜底 ──
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
            if any(max(0, min(d['px_box'][2], embed_px[2]) - max(d['px_box'][0], embed_px[0])) > 10
                   and max(0, min(d['px_box'][3], embed_px[3]) - max(d['px_box'][1], embed_px[1])) > 10
                   for d in detections):
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
                detections.append({
                    'px_box': (int((r.x0 + ex1 * scale_x) * SCALE),
                               int((r.y0 + ey1 * scale_y) * SCALE),
                               int((r.x0 + ex2 * scale_x) * SCALE),
                               int((r.y0 + ey2 * scale_y) * SCALE)),
                    'px_box_full': (int(r.x0 * SCALE), int(r.y0 * SCALE),
                                    int(r.x1 * SCALE), int(r.y1 * SCALE)),
                    'cls': cls, 'conf': emb_conf, 'source': 'embed',
                })

        # ── 全页检测与嵌入图对齐 ──
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

        # ── 去重: 空间聚类 ──
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
                contained = ((dx1 >= kx1 and dy1 >= ky1 and dx2 <= kx2 and dy2 <= ky2)
                           or (kx1 >= dx1 and ky1 >= dy1 and kx2 <= dx2 and ky2 <= dy2))
                if y_overlap_ok or contained:
                    is_dup = True; break
            if not is_dup:
                kept.append(d)
        detections = kept

        # ── 已覆盖区域 (用于 Plan B 排除) ──
        covered_rects = []
        for d in detections:
            x1, y1, x2, y2 = d['px_box']
            covered_rects.append((x1, y1, x2, y2))

        # ── 构建条码元素 ──
        barcode_elements = []
        barcode_idx = 0

        for det in detections:
            x1_px, y1_px, x2_px, y2_px = det['px_box']
            cls = det['cls']
            class_name = model.names[cls]

            if 'px_box_full' in det and det['source'] == 'embed':
                fx1, fy1_, fx2, fy2_ = det['px_box_full']
                x1_px = min(x1_px, fx1)
                y1_px = min(y1_px, fy1_)
                x2_px = max(x2_px, fx2)
                y2_px = max(y2_px, fy2_)

            bx0, by0 = x1_px / SCALE, y1_px / SCALE
            bx1, by1 = x2_px / SCALE, y2_px / SCALE
            bw, bh = bx1 - bx0, by1 - by0
            if bw < 5 or bh < 5:
                continue

            # 误检过滤
            if det['source'] == 'full' and class_name == 'barcode' and bw < 55:
                print(f"  [SKIP] {class_name:7s} too small ({bw:.0f}×{bh:.0f}pt)")
                continue
            if det['source'] in ('embed', 'full+embed') and 'px_box_full' in det:
                efx0, efy0, efx1, efy1 = det['px_box_full']
                ew = (efx1 - efx0) / SCALE
                eh = (efy1 - efy0) / SCALE
                if ew > 120 or eh > 120:
                    cov = (bw * bh) / (ew * eh) * 100 if ew * eh > 0 else 0
                    if cov < 30:
                        print(f"  [SKIP] {class_name:7s} large embed ({ew:.0f}×{eh:.0f}pt, {cov:.0f}%)")
                        continue

            # pyzbar 解码
            px0, py0 = max(0, int(bx0 * SCALE)), max(0, int(by0 * SCALE))
            px1, py1 = min(PAGE_W, int(bx1 * SCALE)), min(PAGE_H, int(by1 * SCALE))
            crop = img[py0:py1, px0:px1]
            raw_value, zbar_type = decode_with_pyzbar(crop) if crop.size > 0 else (None, None)

            # 扩展裁剪 + ocr_label
            PAD = 5
            crop_x0, crop_y0 = max(0, bx0 - PAD), max(0, by0 - PAD)
            crop_x1, crop_y1 = min(page_w, bx1 + PAD), min(page_h, by1 + PAD)
            ocr_label = ""
            for tb in text_blocks:
                tx0, ty0, tx1, ty1 = tb['x0'], tb['y0'], tb['x1'], tb['y1']
                overlap_x = max(0, min(bx1, tx1) - max(bx0, tx0))
                if overlap_x < max(bw * 0.3, 10):
                    continue
                if 0 < ty0 - by1 < 60:
                    crop_y1 = max(crop_y1, ty1 + PAD)
                    ocr_label = tb['text']; break
                if 0 < by0 - ty1 < 30:
                    crop_y0 = min(crop_y0, ty0 - PAD)
                    ocr_label = tb['text']; break

            # 保存图片 — 统一命名
            barcode_idx += 1
            category = class_name  # 'barcode' or 'qrcode'
            value_part = safe_filename(raw_value or ocr_label or f"p{pn}", 25)

            save_px0 = max(0, int(crop_x0 * SCALE))
            save_py0 = max(0, int(crop_y0 * SCALE))
            save_px1 = min(PAGE_W, int(crop_x1 * SCALE))
            save_py1 = min(PAGE_H, int(crop_y1 * SCALE))

            fhash = "000000"
            if save_px1 - save_px0 >= 10 and save_py1 - save_py0 >= 10:
                save_crop = img[save_py0:save_py1, save_px0:save_px1]
                fhash = img_hash(save_crop)

            fname = f"{model_name}_{category}_p{pn:02d}_s{barcode_idx:02d}_{value_part}_{fhash}.png"
            fpath = os.path.join(out_dir, category, fname)

            if save_px1 - save_px0 >= 10 and save_py1 - save_py0 >= 10:
                Image.fromarray(save_crop).save(fpath)

            barcode_elements.append({
                'type': 'barcode',
                'file_name': fname,
                'category': category,
                'class': class_name,
                'confidence': round(det['conf'], 3),
                'raw_value': raw_value or '',
                'zbar_type': zbar_type or '',
                'ocr_label': ocr_label,
                'x0': bx0, 'y0': by0, 'x1': bx1, 'y1': by1,
                'source': det['source'],
            })

            # 加入 image_index
            context = ocr_label or f"Page {pn} barcode"
            image_index.append({
                'image_id': f"{fhash}_{model_name}",
                'file_name': fname,
                'category': category,
                'confidence': round(det['conf'], 3),
                'context_text': context,
                'image_order': barcode_idx,
                'source_doc_rel': pdf_path,
                'applicable_models': [{
                    'category': '手持扫描枪',
                    'series': series_name,
                    'model': model_name,
                    'full_name': model_name,
                }],
                'image_url': f"kb-images/pdf/{category}/{fname}",
                'barcode_raw_value': raw_value or '',
                'zbar_type': zbar_type or '',
                'page': pn,
            })

        # ── 🆕 Plan B: 非条码图片提取 ──
        MIN_IMG_SIZE = 50   # pt, 最小尺寸
        MAX_PAGE_RATIO = 0.8

        for img_info in page.get_images(full=True):
            xref = img_info[0]
            base = doc.extract_image(xref)
            w, h = base['width'], base['height']
            rects = page.get_image_rects(xref)
            if not rects:
                continue
            r = rects[0]
            iw_pt = (r.x1 - r.x0)
            ih_pt = (r.y1 - r.y0)

            # 尺寸过滤
            if iw_pt < MIN_IMG_SIZE or ih_pt < MIN_IMG_SIZE:
                continue
            if iw_pt > page_w * MAX_PAGE_RATIO or ih_pt > page_h * MAX_PAGE_RATIO:
                continue

            # 检查是否被 YOLO 条码覆盖
            ix0_px, iy0_px = int(r.x0 * SCALE), int(r.y0 * SCALE)
            ix1_px, iy1_px = int(r.x1 * SCALE), int(r.y1 * SCALE)
            overlap = False
            for cx1, cy1, cx2, cy2 in covered_rects:
                ox = max(0, min(ix1_px, cx2) - max(ix0_px, cx1))
                oy = max(0, min(iy1_px, cy2) - max(iy0_px, cy1))
                if ox > 20 and oy > 20:
                    overlap = True
                    break
            if overlap:
                continue

            # 保存非条码图片
            image_counter += 1
            try:
                img_bytes = base['image']
                sub_img = Image.open(BytesIO(img_bytes))
                if sub_img.mode == 'RGBA':
                    bg = Image.new('RGB', sub_img.size, (255, 255, 255))
                    bg.paste(sub_img, mask=sub_img.split()[3])
                    sub_img = bg
                elif sub_img.mode != 'RGB':
                    sub_img = sub_img.convert('RGB')
                crop_np = np.array(sub_img)
                fhash = img_hash(crop_np)

                # 找附近文字作标注
                label = ""
                i_mid_y = (r.y0 + r.y1) / 2
                for tb in text_blocks:
                    t_mid_y = (tb['y0'] + tb['y1']) / 2
                    if abs(t_mid_y - i_mid_y) < 60:
                        label = tb['text']
                        break

                label_part = safe_filename(label, 20) if label else f"p{pn}"
                fname = f"{model_name}_image_p{pn:02d}_s{image_counter:02d}_{label_part}_{fhash}.png"
                fpath = os.path.join(out_dir, 'image', fname)
                sub_img.save(fpath)

                image_index.append({
                    'image_id': f"{fhash}_{model_name}",
                    'file_name': fname,
                    'category': 'image',
                    'confidence': 1.0,
                    'context_text': label or f"Page {pn} image",
                    'image_order': image_counter,
                    'source_doc_rel': pdf_path,
                    'applicable_models': [{
                        'category': '手持扫描枪',
                        'series': series_name,
                        'model': model_name,
                        'full_name': model_name,
                    }],
                    'image_url': f"kb-images/pdf/image/{fname}",
                    'page': pn,
                })

                print(f"  [IMAGE] page{pn} → {fname} ({iw_pt:.0f}×{ih_pt:.0f}pt) \"{label[:40]}\"")
            except Exception as e:
                print(f"  [IMAGE] page{pn} ERROR: {e}")

        if not detections and not text_blocks:
            continue

        # ── PATCH 2: 重叠过滤 ──
        text_blocks = filter_overlapping_text(text_blocks, barcode_elements)

        # ── PATCH 1: 分栏检测 ──
        split_x, num_cols = detect_columns(text_blocks, page_w, page_h)

        # ── 空间排序 ──
        all_elements = []
        for tb in text_blocks:
            all_elements.append({
                'type': 'text', 'content': tb['text'],
                'x0': tb['x0'], 'y0': tb['y0'], 'x1': tb['x1'], 'y1': tb['y1'],
            })
        for be_ in barcode_elements:
            all_elements.append({
                'type': 'barcode',
                'file_name': be_['file_name'],
                'category': be_['category'],
                'class': be_['class'],
                'confidence': be_['confidence'],
                'raw_value': be_['raw_value'],
                'zbar_type': be_['zbar_type'],
                'ocr_label': be_['ocr_label'],
                'x0': be_['x0'], 'y0': be_['y0'], 'x1': be_['x1'], 'y1': be_['y1'],
            })

        if num_cols == 2 and split_x is not None:
            left = [e for e in all_elements if (e['x0'] + e['x1']) / 2 < split_x]
            right = [e for e in all_elements if (e['x0'] + e['x1']) / 2 >= split_x]
            left.sort(key=lambda e: e['y0'])
            right.sort(key=lambda e: e['y0'])
            merged, li, ri = [], 0, 0
            while li < len(left) and ri < len(right):
                if right[ri]['y0'] < left[li]['y0'] - 20:
                    merged.append(right[ri]); ri += 1
                else:
                    band_top = left[li]['y0']
                    while li < len(left) and left[li]['y0'] < band_top + 30:
                        merged.append(left[li]); li += 1
                    while ri < len(right) and right[ri]['y0'] < band_top + 30:
                        merged.append(right[ri]); ri += 1
            merged.extend(left[li:])
            merged.extend(right[ri:])
            all_elements = merged
        else:
            all_elements.sort(key=lambda e: e['y0'])

        # ── 结构化输出 ──
        elements_out = []
        for elem in all_elements:
            if elem['type'] == 'text':
                elements_out.append({
                    'type': 'text',
                    'content': elem['content'],
                    'bbox': [round(elem['x0'], 1), round(elem['y0'], 1),
                             round(elem['x1'], 1), round(elem['y1'], 1)],
                })
            else:
                elements_out.append({
                    'type': 'barcode',
                    'file_name': elem['file_name'],
                    'category': elem['category'],
                    'barcode_raw_value': elem['raw_value'],
                    'image_url': f"kb-images/pdf/{elem['category']}/{elem['file_name']}",
                    'ocr_label': elem['ocr_label'],
                    'class': elem['class'],
                    'confidence': elem['confidence'],
                    'zbar_type': elem['zbar_type'],
                    'bbox': [round(elem['x0'], 1), round(elem['y0'], 1),
                             round(elem['x1'], 1), round(elem['y1'], 1)],
                })

        page_data = {
            'page': pn, 'width': round(page_w, 1), 'height': round(page_h, 1),
            'is_scanned': scanned, 'columns': num_cols,
            'element_count': len(elements_out), 'elements': elements_out,
        }
        all_pages.append(page_data)

        n_bc = sum(1 for e in elements_out if e['type'] == 'barcode')
        n_dec = sum(1 for e in elements_out if e['type'] == 'barcode' and e.get('barcode_raw_value'))
        n_txt = len(elements_out) - n_bc
        print(f"  Page {pn}: {len(elements_out)} elements ({n_txt} text + {n_bc} barcode, {n_dec} decoded) | scanned={scanned} | cols={num_cols}")

        total_elements += len(elements_out)
        total_barcodes += n_bc

    doc.close()

    # ── 输出 ──
    output = {
        'pdf': os.path.basename(pdf_path),
        'model': model_name,
        'series': series_name,
        'version': '7.3',
        'total_pages': len(all_pages),
        'total_elements': total_elements,
        'total_barcodes': total_barcodes,
        'total_images': image_counter,
        'pages': all_pages,
    }

    idx_path = os.path.join(out_dir, "index.json")
    with open(idx_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # 统一 image_index (对齐 DOCX 格式)
    img_idx_path = os.path.join(out_dir, "image_index.json")
    with open(img_idx_path, 'w', encoding='utf-8') as f:
        json.dump(image_index, f, ensure_ascii=False, indent=2)

    print(f"\n✅ {os.path.basename(pdf_path)} → {out_dir}")
    print(f"   index.json: {total_elements} elements ({total_barcodes} barcode, {image_counter} image)")
    print(f"   image_index.json: {len(image_index)} entries")
    print(f"   images: kb-images/pdf/{{barcode,qrcode,image}}/")
    return output


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V7.3: YOLO barcode + Plan B images + unified naming")
    parser.add_argument("--pdf", required=True, help="Input PDF path")
    parser.add_argument("--out", required=True, help="Output directory (e.g. kb-images/pdf)")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--skip", type=str, default="1")
    args = parser.parse_args()
    skip = set(int(p.strip()) for p in args.skip.split(",") if p.strip())
    extract_v73(args.pdf, args.out, dpi=args.dpi, conf=args.conf, skip_pages=skip)
