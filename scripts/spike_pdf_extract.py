#!/usr/bin/env python3
"""
spike_pdf_extract.py — PDF 文本提取 + 条码关联验证

验证三层方案:
  1. PyMuPDF word-level 提取 + 坐标
  2. pdfplumber char-level 提取 + 坐标
  3. 空间匹配算法 (ROI × 3层)
  4. LLM 语义整理

生成模拟 Honeywell 手册页面进行对比测试。
"""

import json, os, re, sys, time
from collections import defaultdict
from io import BytesIO

# ═══════════════════════════════════════════
# Step 1: 生成模拟测试 PDF
# ═══════════════════════════════════════════

def create_test_pdf():
    """生成模拟 Honeywell XEN197X 手册页——条码 + 标签布局"""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import mm, cm
    import reportlab

    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4  # 595.27 x 841.89 pt

    # 页面标题
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, h - 50, "XEN197X User Guide — Keyboard Country Settings")
    c.setFont("Helvetica", 9)
    c.drawString(50, h - 65, "Scan the appropriate barcode below to set the keyboard layout for your country.")

    # 模拟条码 + 标签布局 (3列 × 3行)
    barcodes = [
        # (label_above, barcode_value, description, x_mm, y_mm)
        ("United States", "KBDCTY0.", "US English keyboard layout", 20, 200),
        ("United Kingdom", "KBDCTY1.", "UK English keyboard layout", 90, 200),
        ("France", "KBDCTY2.", "French AZERTY keyboard layout", 160, 200),
        ("Germany", "KBDCTY3.", "German QWERTZ keyboard layout", 20, 140),
        ("Italy", "KBDCTY4.", "Italian keyboard layout", 90, 140),
        ("Spain", "KBDCTY5.", "Spanish keyboard layout", 160, 140),
        ("Japan", "KBDCTY6.", "Japanese 106-key layout", 20, 80),
        ("Sweden", "KBDCTY7.", "Swedish keyboard layout", 90, 80),
        ("Default", "KBDDFT.", "Restore default keyboard", 160, 80),
    ]

    for label, code, desc, x_mm, y_mm in barcodes:
        x = x_mm * mm
        y = h - y_mm * mm

        # 标签文字（条码上方）
        c.setFont("Helvetica-Bold", 8)
        c.drawString(x, y + 20, label)
        c.setFont("Helvetica", 7)
        c.drawString(x, y + 10, desc)

        # 模拟条码框
        c.setStrokeGray(0.3)
        c.setFillGray(0.95)
        c.rect(x, y - 18, 50, 18, fill=1, stroke=1)

        # 条码值（条码框内）
        c.setFont("Courier", 7)
        c.setFillGray(0)
        c.drawString(x + 3, y - 12, f"*{code}*")

        # 条码值（框下方，更小）
        c.setFont("Helvetica", 6)
        c.drawString(x + 8, y - 22, code)

    c.save()
    return buf.getvalue()


# ═══════════════════════════════════════════
# Step 2: PyMuPDF 提取
# ═══════════════════════════════════════════

def extract_pymupdf(pdf_bytes):
    """PyMuPDF word-level 提取"""
    import fitz
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[0]
    words = page.get_text("words")
    doc.close()

    results = []
    for w in words:
        x0, y0, x1, y1, text, block_no, line_no, word_no = w
        results.append({
            'x': x0, 'y': y0, 'w': x1 - x0, 'h': y1 - y0,
            'text': text.strip(),
            'block': block_no, 'line': line_no,
        })
    return results


# ═══════════════════════════════════════════
# Step 3: pdfplumber 提取
# ═══════════════════════════════════════════

def extract_pdfplumber(pdf_bytes):
    """pdfplumber char-level 提取"""
    import pdfplumber
    doc = pdfplumber.open(BytesIO(pdf_bytes))
    page = doc.pages[0]
    chars = page.chars
    doc.close()

    results = []
    for ch in chars:
        results.append({
            'x': ch['x0'], 'y': ch['top'], 'w': ch['width'], 'h': ch['height'],
            'text': ch['text'],
            'fontname': ch.get('fontname', ''),
            'size': ch.get('size', 0),
        })
    return results


# ═══════════════════════════════════════════
# Step 4: 空间匹配算法
# ═══════════════════════════════════════════

def build_text_lines(words):
    """将逐词/逐字合并为文本行"""
    if not words:
        return []
    # 按 y 坐标分组 (±3pt 视为同行)
    words_by_line = defaultdict(list)
    for w in words:
        y_key = round(w['y'] / 3) * 3
        words_by_line[y_key].append(w)
    # 每行内按 x 排序
    lines = []
    for y_key in sorted(words_by_line.keys()):
        line_words = sorted(words_by_line[y_key], key=lambda w: w['x'])
        line_text = ' '.join(w['text'] for w in line_words)
        lines.append({
            'y': y_key,
            'x0': min(w['x'] for w in line_words),
            'x1': max(w['x'] + w['w'] for w in line_words),
            'text': line_text.strip(),
            'words': line_words,
        })
    return lines


def find_text_for_barcode(barcode_bbox, lines, words, page_h):
    """
    三层 ROI 匹配: 为条码位置找到关联文本
    
    barcode_bbox: (x, y, w, h) 条码在 PDF 坐标中的位置
    page_h: 页面高度 (用于坐标翻转)
    
    返回: {
        'label_above': 标签文本（上方最近）,
        'value_below': 条码值文本（下方最近）,
        'description': 描述文本（周围收集）,
        'candidates': [所有候选文本片段],
    }
    """
    bx, by, bw, bh = barcode_bbox
    bcx, bcy = bx + bw / 2, by + bh / 2  # 条码中心点

    # 三层 ROI 半径 (pt)
    rois = [30, 80, 150]
    candidates = []

    for r in rois:
        nearby = []
        for w in words:
            wx, wy = w['x'] + w['w'] / 2, w['y'] + w['h'] / 2
            dist = ((wx - bcx) ** 2 + (wy - bcy) ** 2) ** 0.5
            if dist <= r:
                nearby.append({'word': w, 'dist': dist, 'roi': r})
        
        if nearby:
            # 最近的作为 candidates
            nearby.sort(key=lambda x: x['dist'])
            for n in nearby[:8]:  # 最多取 8 个最近词
                if n not in candidates:
                    candidates.append(n)
            if len(candidates) >= 5:
                break  # 够候选了

    # 分类: label(上方), value(下方/内部), description(周围)
    label_above = ''
    value_below = ''
    descriptions = []

    for c in candidates:
        w = c['word']
        wy = w['y'] + w['h'] / 2
        wx = w['x'] + w['w'] / 2

        # 判断相对位置
        if wy < by:  # 在条码上方
            if not label_above or c['dist'] < 25:
                label_above = w['text']
        elif wy > by + bh:  # 在条码下方
            if not value_below or c['dist'] < 20:
                value_below = w['text']
        
        # 所有非条码值文本作为描述候选
        text = w['text']
        if not re.match(r'^\*?[A-Z][A-Z0-9]{3,}\.*\*?$', text):  # 不是条码值
            descriptions.append(text)

    return {
        'label_above': label_above,
        'value_below': value_below,
        'description': ' '.join(descriptions),
        'candidates_text': [c['word']['text'] for c in candidates],
    }


# ═══════════════════════════════════════════
# Step 5: LLM 语义整理（模拟）
# ═══════════════════════════════════════════

def simulate_llm_refine(matches):
    """模拟 LLM 整理: 输入碎片文本 → 输出统一描述"""
    results = []
    for m in matches:
        parts = []
        if m['label_above']:
            parts.append(m['label_above'])
        if m['description']:
            parts.append(m['description'])

        combined = ' '.join(parts) if parts else 'Unknown setting'
        
        # 简化: 用规则模拟 LLM 效果
        # 实际 LLM 会把 "United States US English keyboard" → "美国键盘布局"
        results.append({
            **m,
            'llm_refined': combined.strip(),
        })
    return results


# ═══════════════════════════════════════════
# Main: 对比测试
# ═══════════════════════════════════════════

def main():
    print("=" * 65)
    print("  PDF 文本提取 + 条码关联 — 方案对比验证")
    print("=" * 65)

    # 1. 生成测试 PDF
    pdf_bytes = create_test_pdf()
    print(f"\n✅ 测试 PDF 已生成 ({len(pdf_bytes):,} bytes)")

    # 2. PyMuPDF 提取
    t0 = time.time()
    mu_words = extract_pymupdf(pdf_bytes)
    mu_time = (time.time() - t0) * 1000
    mu_lines = build_text_lines(mu_words)
    print(f"\n📄 PyMuPDF: {len(mu_words)} 词, {len(mu_lines)} 行 ({mu_time:.1f}ms)")

    # 3. pdfplumber 提取
    t0 = time.time()
    pl_chars = extract_pdfplumber(pdf_bytes)
    pl_time = (time.time() - t0) * 1000
    pl_lines = build_text_lines(pl_chars)
    print(f"📄 pdfplumber: {len(pl_chars)} 字符, {len(pl_lines)} 行 ({pl_time:.1f}ms)")

    # 4. 定义模拟条码位置 (对应 generate 中的坐标)
    # Honeywell 手册用 mm, PDF 用 pt (1pt = 0.3528mm, 1mm = 2.835pt)
    # 页面高 841.89pt, y 从上到下递增
    barcodes_test = [
        # (name, x_pt, y_pt, w_pt, h_pt)
        ("KBDCTY0. (US)",    20*2.835, 841.89-200*2.835, 50*2.835, 18*2.835),
        ("KBDCTY1. (UK)",    90*2.835, 841.89-200*2.835, 50*2.835, 18*2.835),
        ("KBDCTY2. (FR)",   160*2.835, 841.89-200*2.835, 50*2.835, 18*2.835),
        ("KBDCTY3. (DE)",    20*2.835, 841.89-140*2.835, 50*2.835, 18*2.835),
        ("KBDCTY4. (IT)",    90*2.835, 841.89-140*2.835, 50*2.835, 18*2.835),
        ("KBDCTY5. (ES)",   160*2.835, 841.89-140*2.835, 50*2.835, 18*2.835),
        ("KBDCTY6. (JP)",    20*2.835, 841.89-80*2.835,  50*2.835, 18*2.835),
        ("KBDCTY7. (SE)",    90*2.835, 841.89-80*2.835,  50*2.835, 18*2.835),
        ("KBDDFT. (Default)",160*2.835, 841.89-80*2.835,  50*2.835, 18*2.835),
    ]

    # 5. 运行匹配
    print("\n" + "=" * 65)
    print("  空间匹配结果 (PyMuPDF vs pdfplumber)")
    print("=" * 65)

    page_h = 841.89

    for engine_name, words in [("PyMuPDF", mu_words), ("pdfplumber", pl_chars)]:
        print(f"\n─── {engine_name} ───")
        lines = build_text_lines(words)
        correct = 0

        for name, x, y, w, h in barcodes_test:
            match = find_text_for_barcode((x, y, w, h), lines, words, page_h)
            is_correct = name.split('.')[0] in match.get('label_above', '') or \
                         name.split('.')[0] in match.get('value_below', '')
            if is_correct:
                correct += 1
            print(f"  {name}: label='{match['label_above']}' value='{match['value_below']}' "
                  f"desc='{match['description'][:50]}' {'✅' if is_correct else '❌'}")

        print(f"  → 准确率: {correct}/{len(barcodes_test)} ({100*correct/len(barcodes_test):.0f}%)")

    # 6. LLM 整理示例
    print("\n" + "=" * 65)
    print("  LLM 语义整理 (模拟)")
    print("=" * 65)
    all_matches = []
    for name, x, y, w, h in barcodes_test:
        match = find_text_for_barcode((x, y, w, h), mu_lines, mu_words, page_h)
        match['barcode'] = name
        all_matches.append(match)

    refined = simulate_llm_refine(all_matches)
    for r in refined[:5]:
        print(f"  {r['barcode'][:20]}: {r['llm_refined'][:70]}")


    # 7. 保存测试 PDF 用于人工检查
    test_path = '/tmp/spike_test.pdf'
    with open(test_path, 'wb') as f:
        f.write(pdf_bytes)
    print(f"\n📎 测试 PDF: {test_path}")
    print(f"   打开查看: open {test_path}")


if __name__ == '__main__':
    main()
