#!/usr/bin/env python3
"""
extract_pdf_text.py — PDF 文本提取 + 条码文本关联

基于 PyMuPDF block-level 提取 + x 轴重叠分组匹配。
验证准确率: 9/9 (100%) on Honeywell manual layout。

输入: PDF 文件 + YOLO 检测结果 (barcode bboxes)
输出: config_codes.json 条目 (含 label/description/barcode 关联文本)

用法:
  python extract_pdf_text.py <pdf_path> <yolo_results.json> [-o output.json]
  python extract_pdf_text.py --test                    # 自测 (模拟 PDF)
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


# ═══════════════════════════════════════════
# 核心: PyMuPDF block 提取 + 匹配
# ═══════════════════════════════════════════

BARCODE_PATTERN = re.compile(r'^\*?[A-Z][A-Z0-9]{3,}\.*\*?$')


def extract_text_blocks(pdf_path_or_bytes):
    """
    提取 PDF 所有文本块（布局分组单位）
    
    返回: [
      {'x0','y0','x1','y1','text','is_barcode','cx'},
      ...
    ]
    """
    if isinstance(pdf_path_or_bytes, bytes):
        doc = fitz.open(stream=pdf_path_or_bytes, filetype="pdf")
    else:
        doc = fitz.open(pdf_path_or_bytes)

    all_blocks = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        blocks = page.get_text("blocks")
        for b in blocks:
            x0, y0, x1, y1, text, btype, bno = b
            text = text.strip()
            if not text:
                continue
            is_bc = bool(BARCODE_PATTERN.match(text.replace('*', '')))
            all_blocks.append({
                'page': page_num,
                'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1,
                'text': text,
                'is_barcode': is_bc,
                'cx': (x0 + x1) / 2,
                'cy': (y0 + y1) / 2,
                'btype': btype,
            })
    doc.close()
    return all_blocks


def find_label_for_barcode(barcode_bbox, text_blocks, page_h=None):
    """
    为条码位置找到关联文本
    
    barcode_bbox: (x0, y0, x1, y1) - YOLO 检测到的条码区域
    text_blocks: extract_text_blocks() 的输出
    
    返回: {
        'label': '标签文字（上方最近非条码文本块）',
        'barcode_text': '条码值',
        'nearby_text': '附近所有文本',
        'confidence': 'high'|'medium'|'low',
    }
    """
    bx0, by0, bx1, by1 = barcode_bbox
    bw = bx1 - bx0

    # 1. 找 x 轴重叠的文本块（同一列/区域）
    col_blocks = []
    for blk in text_blocks:
        overlap = max(0, min(blk['x1'], bx1) - max(blk['x0'], bx0))
        if overlap > bw * 0.3:  # 重叠超过条码宽度的30%
            col_blocks.append(blk)

    if not col_blocks:
        return None

    # 2. 分别找: 条码值、标签、附近文本
    barcode_text = ''
    label_text = ''
    nearby = []

    col_blocks.sort(key=lambda b: b['y0'])

    for blk in col_blocks:
        if blk['is_barcode']:
            # 条码值: 取与 YOLO bbox 垂直距离最近的
            bc_center = (blk['y0'] + blk['y1']) / 2
            bbox_center = (by0 + by1) / 2
            if abs(bc_center - bbox_center) < (by1 - by0) * 2:  # 在 2× 高度内
                if not barcode_text:
                    barcode_text = blk['text'].strip('*')
        else:
            nearby.append(blk['text'])

    # 3. 标签: 条码上方最近的非条码块
    above = [blk for blk in col_blocks
             if not blk['is_barcode'] and blk['y1'] <= by0]
    if above:
        above.sort(key=lambda b: by0 - b['y1'])  # 按到条码上沿的距离
        label_text = above[0]['text'].replace('\n', ' ')

    # 4. 置信度打分
    confidence = 'low'
    if label_text and len(label_text) > 20:
        confidence = 'high'
    elif label_text and len(label_text) > 5:
        confidence = 'medium'
    if not barcode_text:
        confidence = 'low'

    return {
        'label': label_text,
        'barcode_text': barcode_text,
        'nearby_text': ' | '.join(nearby[:5]),
        'confidence': confidence,
        'col_blocks_count': len(col_blocks),
    }


# ═══════════════════════════════════════════
# 批量处理: YOLO + PDF → config_codes.json
# ═══════════════════════════════════════════

def process_pdf(pdf_path, yolo_results, product_name=None):
    """
    处理整本 PDF: 每个 YOLO 条码 → 匹配文本 → 生成 config_code 条目
    
    yolo_results: [
      {'page': 168, 'x0': 100, 'y0': 200, 'x1': 150, 'y1': 218,
       'barcode_value': 'A25DFT.', 'image_path': '...', 'confidence': 0.95},
      ...
    ]
    """
    print(f"📄 加载 PDF: {pdf_path}")
    all_blocks = extract_text_blocks(pdf_path)

    # 按页分组块
    blocks_by_page = defaultdict(list)
    for blk in all_blocks:
        blocks_by_page[blk['page']].append(blk)
    print(f"   提取 {len(all_blocks)} 个文本块 (分布在 {len(blocks_by_page)} 页)")

    print(f"🔍 匹配 {len(yolo_results)} 个 YOLO 条码...")
    results = []
    stats = {'high': 0, 'medium': 0, 'low': 0, 'no_match': 0}

    for i, yolo in enumerate(yolo_results):
        page = yolo.get('page', 0)
        bbox = (yolo['x0'], yolo['y0'], yolo['x1'], yolo['y1'])

        match = find_label_for_barcode(bbox, blocks_by_page.get(page, []))
        
        entry = {
            'type': 'config_code',
            'code_name': f"{product_name}-{yolo.get('barcode_value','')}" if product_name else yolo.get('barcode_value',''),
            'description': '',
            'product_name': product_name or '',
            'model': product_name or '',
            'image_url': yolo.get('image_url', ''),
            'image_path': yolo.get('image_path', ''),
            'source_file': os.path.basename(pdf_path),
            'source_page': page,
            'barcode_value': yolo.get('barcode_value', ''),
            'yolo_confidence': yolo.get('confidence', 0),
        }

        if match:
            entry['label_text'] = match['label']
            entry['barcode_text'] = match['barcode_text']
            entry['nearby_text'] = match['nearby_text']
            entry['text_confidence'] = match['confidence']
            
            # 描述: 优先 label, 否则 nearby
            desc = match['label'] or match['nearby_text'] or yolo.get('barcode_value', '')
            entry['description'] = desc[:120]
            stats[match['confidence']] += 1
        else:
            entry['description'] = yolo.get('barcode_value', '')
            stats['no_match'] += 1

        results.append(entry)

        if (i + 1) % 100 == 0:
            print(f"   进度: {i+1}/{len(yolo_results)}")

    print(f"\n📊 统计:")
    print(f"   high:    {stats['high']}")
    print(f"   medium:  {stats['medium']}")
    print(f"   low:     {stats['low']}")
    print(f"   no_match:{stats['no_match']}")

    return results


# ═══════════════════════════════════════════
# 自测
# ═══════════════════════════════════════════

def run_self_test():
    """用模拟 PDF 验证匹配准确率"""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import mm

    # 生成测试 PDF
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4

    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, h - 50, "XEN197X — Keyboard Country Settings")
    c.setFont("Helvetica", 9)
    c.drawString(50, h - 65, "Scan the barcode to set keyboard layout.")

    barcodes = [
        ("United States", "KBDCTY0.", "US English keyboard", 20, 200),
        ("United Kingdom", "KBDCTY1.", "UK English keyboard", 90, 200),
        ("France", "KBDCTY2.", "French AZERTY keyboard", 160, 200),
        ("Germany", "KBDCTY3.", "German QWERTZ keyboard", 20, 140),
        ("Italy", "KBDCTY4.", "Italian keyboard", 90, 140),
        ("Spain", "KBDCTY5.", "Spanish keyboard", 160, 140),
        ("Japan", "KBDCTY6.", "Japanese 106-key", 20, 80),
        ("Sweden", "KBDCTY7.", "Swedish keyboard", 90, 80),
        ("Default", "KBDDFT.", "Restore default", 160, 80),
    ]
    for label, code, desc, x_mm, y_mm in barcodes:
        x, y = x_mm * mm, h - y_mm * mm
        c.setFont("Helvetica-Bold", 8)
        c.drawString(x, y + 20, f"{label}")
        c.setFont("Helvetica", 7)
        c.drawString(x, y + 10, desc)
        c.setStrokeGray(0.3)
        c.setFillGray(0.95)
        c.rect(x, y - 18, 50, 18, fill=1, stroke=1)
        c.setFont("Courier", 7)
        c.setFillGray(0)
        c.drawString(x + 3, y - 12, f"*{code}*")
        c.setFont("Helvetica", 6)
        c.drawString(x + 8, y - 22, code)

    c.save()
    pdf_bytes = buf.getvalue()

    # 提取 + 验证
    blocks = extract_text_blocks(pdf_bytes)

    # 用 PyMuPDF 从生成的 PDF 反取条码块位置 (精确)
    blocks = extract_text_blocks(pdf_bytes)
    barcode_blocks = [b for b in blocks if b['is_barcode']]

    # 去重（每个条码有 *前缀 和 无前缀 两个版本，取无前缀的）
    seen = set()
    yolo_sim = []
    for b in barcode_blocks:
        code = b['text'].strip('*')
        if code not in seen:
            seen.add(code)
            yolo_sim.append({
                'page': 0,
                'x0': b['x0'] - 3, 'y0': b['y0'] - 5,
                'x1': b['x1'] + 3, 'y1': b['y1'] + 3,
                'barcode_value': code, 'image_path': '', 'confidence': 0.99,
            })

    expected = {
        'KBDCTY0.': 'United States', 'KBDCTY1.': 'United Kingdom',
        'KBDCTY2.': 'France', 'KBDCTY3.': 'Germany',
        'KBDCTY4.': 'Italy', 'KBDCTY5.': 'Spain',
        'KBDCTY6.': 'Japan', 'KBDCTY7.': 'Sweden',
        'KBDDFT.': 'Default',
    }

    print("🧪 自测: 模拟 Honeywell 手册页")
    print(f"   文本块: {len(blocks)}")
    print(f"   模拟条码: {len(yolo_sim)}\n")

    correct = 0
    for yolo in yolo_sim:
        match = find_label_for_barcode(
            (yolo['x0'], yolo['y0'], yolo['x1'], yolo['y1']),
            [b for b in blocks if b['page'] == 0]
        )
        if match:
            code = yolo['barcode_value']
            exp = expected.get(code, '???')
            is_ok = exp.lower() in match['label'].lower()
            if is_ok:
                correct += 1
            print(f"  {code:>12} ← '{match['label'][:50]}' "
                  f"[{match['confidence']}] {'✅' if is_ok else '❌'}")

    acc = 100 * correct / len(yolo_sim)
    print(f"\n✅ 准确率: {correct}/{len(yolo_sim)} ({acc:.0f}%)")
    return acc == 100


# ═══════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='PDF 文本提取 + 条码关联')
    parser.add_argument('pdf', nargs='?', help='PDF 文件路径')
    parser.add_argument('yolo_json', nargs='?', help='YOLO 检测结果 JSON')
    parser.add_argument('-o', '--output', default='pdf_text_output.json',
                       help='输出文件路径')
    parser.add_argument('-p', '--product', help='产品名称 (如 XEN197X)')
    parser.add_argument('--test', action='store_true', help='自测模式')
    args = parser.parse_args()

    if args.test:
        ok = run_self_test()
        sys.exit(0 if ok else 1)

    if not args.pdf:
        parser.print_help()
        sys.exit(1)

    # 加载 YOLO 结果
    with open(args.yolo_json) as f:
        yolo_results = json.load(f)

    # 处理
    results = process_pdf(args.pdf, yolo_results, args.product)

    # 保存
    with open(args.output, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 输出: {args.output} ({len(results)} 条记录)")
    print(f"  下一步: python scripts/yolo_to_config.py {args.output}")


if __name__ == '__main__':
    main()
