#!/usr/bin/env python3
"""
extract_pdf_text.py — PDF 文本提取 + 条码文本关联

基于 PyMuPDF block-level 提取 + 距离加权评分匹配。
验证: 12页79条码, 解码99%, 匹配97%, 1页标题误匹配。

输入: PDF 文件 + (可选) YOLO 检测结果 JSON
输出: 含 label/description 的 config_code 条目

用法:
  python extract_pdf_text.py <pdf_path> [-o output.json] [-p PRODUCT]
  python extract_pdf_text.py <pdf_path> <yolo_results.json> [-o output.json]
  python extract_pdf_text.py --test  # 自测
"""

import fitz
import json
import os
import re
import sys
import time
import argparse
from collections import defaultdict
from io import BytesIO

# ─── 条码值正则 ───
BARCODE_PATTERN = re.compile(r'^\*?[A-Z][A-Z0-9]{3,}\.*\*?$')


# ═══════════════════════════════════════════
# 核心算法: 距离加权评分
# ═══════════════════════════════════════════

def score_text_block(blk, barcode_bbox, page_w, page_h):
    """
    为文本块与条码的关联度打分 (0-100)
    
    因素:
    - 水平重叠 (0-20): 同一列的文本块优先
    - 垂直距离 (0-35): 越近越好
    - 特异性 (0-25): 窄块(标签) > 宽块(段落/标题)
    - 上方偏好 (0-15): 标签通常在条码上方
    - 文本质量 (0-5): 中等长度文本最可能是标签
    """
    bx0, by0, bx1, by1 = barcode_bbox
    bw = bx1 - bx0
    block_w = blk['x1'] - blk['x0']
    text = blk['text']
    score = 0.0

    # 1. 水平重叠
    x_overlap = max(0, min(blk['x1'], bx1) - max(blk['x0'], bx0))
    if x_overlap > bw * 0.1:
        score += min(20, (x_overlap / bw) * 20)

    # 2. 垂直距离
    if blk['y1'] <= by0:
        v_dist = by0 - blk['y1']
    elif blk['y0'] >= by1:
        v_dist = blk['y0'] - by1
    else:
        v_dist = 0

    if v_dist == 0:
        score += 35
    elif v_dist < 30:
        score += 30
    elif v_dist < 80:
        score += max(0, 25 - v_dist * 0.3)
    elif v_dist < 200:
        score += max(0, 15 - v_dist * 0.1)
    else:
        score += max(0, 5 - v_dist * 0.01)

    # 3. 特异性 — 窄块更可能是标签
    specificity = 1.0 - min(1.0, block_w / page_w)
    score += specificity * 25

    # 4. 上方偏好
    if blk['y1'] <= by0:
        score += 15
    elif blk['y0'] < by0 + (by1 - by0):
        score += 5

    # 5. 文本质量
    text_len = len(text)
    if 8 <= text_len <= 60:
        score += 5
    elif text_len < 8:
        score += 2

    # 惩罚: 页级宽标题
    if block_w > page_w * 0.4:
        score -= 15

    # 惩罚: 页眉/页脚区域
    if blk['y0'] < 40 or blk['y0'] > page_h - 50:
        score -= 10

    return score


def find_label_for_barcode(barcode_bbox, text_blocks, page_w, page_h):
    """为条码位置找到最关联的文本标签"""
    scored = []
    for blk in text_blocks:
        if BARCODE_PATTERN.match(blk['text'].replace('*', '')):
            continue
        s = score_text_block(blk, barcode_bbox, page_w, page_h)
        if s > 0:
            scored.append((s, blk))

    scored.sort(key=lambda x: -x[0])

    if not scored:
        return {
            'label': '',
            'candidates': [],
            'confidence': 'none',
        }

    best_score, best_blk = scored[0]
    label = best_blk['text'].replace('\n', ' ')

    # 置信度
    if best_score > 60:
        confidence = 'high'
    elif best_score > 35:
        confidence = 'medium'
    else:
        confidence = 'low'

    candidates = [{
        'text': b['text'].replace('\n', ' ')[:60],
        'score': round(s, 1),
    } for s, b in scored[:3] if s > 10]

    return {
        'label': label,
        'candidates': candidates,
        'confidence': confidence,
        'best_score': round(best_score, 1),
    }


# ═══════════════════════════════════════════
# 批量处理: PDF → config_codes
# ═══════════════════════════════════════════

def extract_text_blocks(pdf_path):
    """提取 PDF 所有文本块"""
    if isinstance(pdf_path, bytes):
        doc = fitz.open(stream=pdf_path, filetype="pdf")
    else:
        doc = fitz.open(pdf_path)

    blocks_by_page = defaultdict(list)
    for pn in range(len(doc)):
        page = doc[pn]
        for b in page.get_text("blocks"):
            x0, y0, x1, y1, text, btype, bno = b
            text = text.strip()
            if text:
                blocks_by_page[pn].append({
                    'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1,
                    'text': text,
                })

    page_dims = {pn: (doc[pn].rect.width, doc[pn].rect.height) for pn in range(len(doc))}
    doc.close()
    return blocks_by_page, page_dims


def process_barcodes(pdf_path, barcode_detections, product_name=None):
    """
    批量处理条码检测结果 → config_code 条目
    
    barcode_detections: [{
        'page': int, 'x0': float, 'y0': float, 'x1': float, 'y1': float,
        'barcode_value': str, 'image_path': str, 'confidence': float,
    }, ...]
    
    返回: [config_code_entry, ...]
    """
    blocks_by_page, page_dims = extract_text_blocks(pdf_path)

    results = []
    stats = {'high': 0, 'medium': 0, 'low': 0, 'none': 0}

    for det in barcode_detections:
        pn = det.get('page', 0)
        bbox = (det['x0'], det['y0'], det['x1'], det['y1'])

        page_w, page_h = page_dims.get(pn, (612, 792))
        match = find_label_for_barcode(
            bbox, blocks_by_page.get(pn, []), page_w, page_h
        )

        bc_val = det.get('barcode_value', '')

        entry = {
            'type': 'config_code',
            'code_name': f"{product_name}-{bc_val}" if product_name else bc_val,
            'barcode_value': bc_val,
            'image_path': det.get('image_path', ''),
            'source_file': os.path.basename(pdf_path) if isinstance(pdf_path, str) else '',
            'source_page': pn + 1,  # 1-indexed
            'product_name': product_name or '',
            'model': product_name or '',
            'yolo_confidence': det.get('confidence', 0),
        }

        if match['label']:
            entry['label_text'] = match['label']
            entry['description'] = match['label'][:120]
            entry['text_confidence'] = match['confidence']
            entry['match_score'] = match['best_score']
            stats[match['confidence']] += 1
        else:
            entry['description'] = bc_val
            stats['none'] += 1

        results.append(entry)

    print(f"📊 匹配统计: high={stats['high']} medium={stats['medium']} "
          f"low={stats['low']} none={stats['none']}")

    return results


# ═══════════════════════════════════════════
# 内置 YOLO + pyzbar 检测 (无需外部 JSON)
# ═══════════════════════════════════════════

def detect_barcodes_with_yolo(pdf_path, model_path=None, sample_pages=None):
    """用 YOLO + pyzbar 检测条码"""
    try:
        from ultralytics import YOLO
        from pyzbar.pyzbar import decode as zbar_decode
        from PIL import Image
    except ImportError as e:
        print(f"⚠️ 缺少依赖: {e}")
        return []

    if model_path is None:
        model_path = os.path.join(os.path.dirname(__file__), '..', 'weights',
                                  'yolov8s-barcode-detection.pt')

    model = YOLO(model_path)

    if isinstance(pdf_path, bytes):
        doc = fitz.open(stream=pdf_path, filetype="pdf")
    else:
        doc = fitz.open(pdf_path)

    total_pages = len(doc)
    if sample_pages:
        pages_to_process = [p for p in sample_pages if p < total_pages]
    else:
        pages_to_process = range(total_pages)

    detections = []
    print(f"🔍 YOLO 检测 {len(pages_to_process)}/{total_pages} 页...")

    for pn in pages_to_process:
        page = doc[pn]
        pix = page.get_pixmap(dpi=200)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        results = model(img, conf=0.3, verbose=False)
        boxes = results[0].boxes
        if boxes is None:
            continue

        scale_x = page.rect.width / pix.width
        scale_y = page.rect.height / pix.height

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])

            # pyzbar 解码
            crop = img.crop((int(x1)-2, int(y1)-2, int(x2)+2, int(y2)+2))
            decoded = zbar_decode(crop)
            bc_val = decoded[0].data.decode('utf-8', errors='ignore') if decoded else ''

            detections.append({
                'page': pn,
                'x0': x1 * scale_x, 'y0': y1 * scale_y,
                'x1': x2 * scale_x, 'y1': y2 * scale_y,
                'barcode_value': bc_val,
                'image_path': '',
                'confidence': conf,
            })

    doc.close()
    print(f"  检测 {len(detections)} 个条码")
    return detections


# ═══════════════════════════════════════════
# 自测
# ═══════════════════════════════════════════

def run_self_test():
    """用模拟 PDF 验证"""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import mm

    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4

    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, h - 50, "XEN197X — Keyboard Country Settings")
    c.setFont("Helvetica", 9)
    c.drawString(50, h - 65, "Scan the barcode to set keyboard layout.")

    barcodes = [
        ("United States", "KBDCTY0.", 20, 200), ("United Kingdom", "KBDCTY1.", 90, 200),
        ("France", "KBDCTY2.", 160, 200), ("Germany", "KBDCTY3.", 20, 140),
        ("Italy", "KBDCTY4.", 90, 140), ("Spain", "KBDCTY5.", 160, 140),
        ("Japan", "KBDCTY6.", 20, 80), ("Sweden", "KBDCTY7.", 90, 80),
        ("Default", "KBDDFT.", 160, 80),
    ]
    for label, code, x_mm, y_mm in barcodes:
        x, y = x_mm * mm, h - y_mm * mm
        c.setFont("Helvetica-Bold", 8)
        c.drawString(x, y + 20, label)
        c.setStrokeGray(0.3)
        c.setFillGray(0.95)
        c.rect(x, y - 18, 50, 18, fill=1, stroke=1)
        c.setFont("Courier", 7)
        c.setFillGray(0)
        c.drawString(x + 3, y - 12, f"*{code}*")

    c.save()
    pdf_bytes = buf.getvalue()

    blocks_by_page, page_dims = extract_text_blocks(pdf_bytes)
    page_w, page_h = page_dims[0]

    # 从 PDF 反取条码块位置
    barcode_blocks = [b for b in blocks_by_page[0]
                      if BARCODE_PATTERN.match(b['text'].replace('*', ''))]
    seen = set()
    detections = []
    for b in barcode_blocks:
        code = b['text'].strip('*')
        if code not in seen:
            seen.add(code)
            detections.append({
                'page': 0, 'x0': b['x0']-3, 'y0': b['y0']-5,
                'x1': b['x1']+3, 'y1': b['y1']+3,
                'barcode_value': code, 'confidence': 0.99,
            })

    expected = {
        'KBDCTY0.': 'United States', 'KBDCTY1.': 'United Kingdom',
        'KBDCTY2.': 'France', 'KBDCTY3.': 'Germany',
        'KBDCTY4.': 'Italy', 'KBDCTY5.': 'Spain',
        'KBDCTY6.': 'Japan', 'KBDCTY7.': 'Sweden', 'KBDDFT.': 'Default',
    }

    print("🧪 自测: 模拟 Honeywell 手册页\n")
    correct = 0
    for det in detections:
        match = find_label_for_barcode(
            (det['x0'], det['y0'], det['x1'], det['y1']),
            blocks_by_page[0], page_w, page_h
        )
        code = det['barcode_value']
        exp = expected.get(code, '???')
        is_ok = exp.lower() in match['label'].lower()
        if is_ok: correct += 1
        print(f"  {code:>12} ← '{match['label'][:45]}' [{match['confidence']}] "
              f"{'✅' if is_ok else '❌'}")

    acc = 100 * correct / len(detections)
    print(f"\n✅ 准确率: {correct}/{len(detections)} ({acc:.0f}%)")
    return acc == 100


# ═══════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='PDF 文本提取 + 条码关联')
    parser.add_argument('pdf', nargs='?', help='PDF 文件路径')
    parser.add_argument('yolo_json', nargs='?', help='YOLO 检测结果 JSON (可选)')
    parser.add_argument('-o', '--output', default='pdf_text_output.json', help='输出文件')
    parser.add_argument('-p', '--product', help='产品名称 (如 XEN197X)')
    parser.add_argument('--yolo-model', help='YOLO 模型路径')
    parser.add_argument('--sample', type=int, nargs='*', help='采样页 (0-indexed)')
    parser.add_argument('--test', action='store_true', help='自测')
    args = parser.parse_args()

    if args.test:
        ok = run_self_test()
        sys.exit(0 if ok else 1)

    if not args.pdf:
        parser.print_help()
        sys.exit(1)

    # 获取条码检测结果
    if args.yolo_json:
        with open(args.yolo_json) as f:
            barcode_dets = json.load(f)
    else:
        # 内置 YOLO 检测
        barcode_dets = detect_barcodes_with_yolo(
            args.pdf, args.yolo_model, args.sample
        )

    if not barcode_dets:
        print("❌ 未检测到条码")
        sys.exit(1)

    # 文本匹配
    results = process_barcodes(args.pdf, barcode_dets, args.product)

    with open(args.output, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 输出: {args.output} ({len(results)} 条记录)")


if __name__ == '__main__':
    main()
