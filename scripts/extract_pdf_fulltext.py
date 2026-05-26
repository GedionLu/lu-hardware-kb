#!/usr/bin/env python3 -u
"""
extract_pdf_fulltext.py — PDF 全文提取 + 章节分段

从 UG PDF 提取完整正文，按章节/页分段，输出结构化 JSON 供 Qdrant 索引。

策略:
  1. PyMuPDF 逐页提取文本
  2. 字体大小推断标题层级 (≥16pt = 章节, ≥13pt = 子章节)
  3. 过滤页眉/页脚 (顶部10%、底部5%区域的重复文本)
  4. 输出: [{product, chapter, section, page, text_chunk, chars}, ...]

用法: python extract_pdf_fulltext.py -o fulltext.json
"""

import fitz, json, os, re, argparse, time
from collections import defaultdict, Counter
from pathlib import Path

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.expanduser('~/openclaw/19Series')

# 要提取的 PDF (product → path)
PDFS = {
    'XEN197X': '197x/XEN197X-EN-UG-01 Rev B 2026.1.19.pdf',
    '1900': 'Xenon-UG.pdf',
    '195X': '195x/XEN195X-EN-UG.pdf',
    '199X': 'Granit XP/199x-en-ug.pdf',
    '199Xi': 'Granit XP/Granit XP 199xi Series UG(1990iSR 1991iXR 1991iXLR 1991iSR 1991iXR 1991iXLR).pdf',
    '196X': '1960/sps-ppr-xen196x-en-ug.pdf',
    'OCR': 'OCR-UG Rev B pdf.pdf',
}


def extract_page_text(page):
    """提取单页文本，返回 [{'text':..., 'size':..., 'y':...}, ...]"""
    blocks = page.get_text("dict")["blocks"]
    spans = []
    for b in blocks:
        if b.get("type") != 0:
            continue
        for line in b.get("lines", []):
            line_text = ""
            line_size = 0
            line_y = line["bbox"][1]
            for span in line.get("spans", []):
                line_text += span["text"]
                line_size = max(line_size, span["size"])
            if line_text.strip():
                spans.append({
                    "text": line_text.strip(),
                    "size": round(line_size, 1),
                    "y": line_y,
                })
    return spans


def is_header_footer(span, page_h, header_texts, footer_texts):
    """判断是否是页眉/页脚"""
    y_ratio = span["y"] / page_h
    text = span["text"].strip()

    # 顶部15% → 页眉
    if y_ratio < 0.15:
        header_texts.add(text)
        return True
    # 底部10% → 页脚
    if y_ratio > 0.90:
        footer_texts.add(text)
        return True
    # 已见过的页眉/页脚文本（出现在多页的同一位置）
    if text in header_texts or text in footer_texts:
        return True

    return False


def extract_sections(spans, page_h, header_texts, footer_texts):
    """将 spans 分组为段落/章节"""
    if not spans:
        return []

    sections = []
    current = {"title": "", "lines": [], "level": "body"}

    for span in spans:
        if is_header_footer(span, page_h, header_texts, footer_texts):
            continue

        text = span["text"]
        size = span["size"]

        # 标题检测
        if size >= 16:
            # 保存前一个段落
            if current["lines"]:
                sections.append(dict(current))
            current = {
                "title": text,
                "lines": [],
                "level": "chapter" if size >= 20 else "section",
            }
        elif size >= 13 and len(text) < 80:
            if current["lines"]:
                sections.append(dict(current))
            current = {
                "title": text,
                "lines": [],
                "level": "subsection",
            }
        else:
            current["lines"].append(text)

    if current["lines"]:
        sections.append(dict(current))

    return sections


def process_pdf(pdf_path, product):
    """处理单个 PDF"""
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    print(f"  📄 {product}: {total_pages} 页")

    chunks = []
    header_texts = set()
    footer_texts = set()
    chapter_title = ""
    section_title = ""
    char_count = 0

    for pn in range(total_pages):
        page = doc[pn]
        page_h = page.rect.height
        spans = extract_page_text(page)
        sections = extract_sections(spans, page_h, header_texts, footer_texts)

        for sec in sections:
            text = " ".join(sec["lines"])
            if not text.strip():
                continue

            # 跟踪章节标题
            if sec["level"] == "chapter":
                chapter_title = sec["title"]
                section_title = ""
            elif sec["level"] in ("section", "subsection"):
                section_title = sec["title"]

            # 切分为 ~1000 字符的块
            words = text.split()
            chunk = ""
            for w in words:
                if len(chunk) + len(w) > 1000:
                    if chunk.strip():
                        chunks.append({
                            "product": product,
                            "chapter": chapter_title,
                            "section": section_title or sec.get("title", ""),
                            "page": pn + 1,
                            "text": chunk.strip(),
                            "chars": len(chunk.strip()),
                        })
                        char_count += len(chunk)
                    chunk = w
                else:
                    chunk += " " + w if chunk else w

            if chunk.strip():
                chunks.append({
                    "product": product,
                    "chapter": chapter_title,
                    "section": section_title or sec.get("title", ""),
                    "page": pn + 1,
                    "text": chunk.strip(),
                    "chars": len(chunk.strip()),
                })
                char_count += len(chunk)

        if (pn + 1) % 100 == 0:
            print(f"    page {pn+1}/{total_pages} ({len(chunks)} chunks, {char_count:,} chars)")

    doc.close()
    print(f"    → {len(chunks)} chunks, {char_count:,} chars")
    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', default='data/fulltext.json')
    args = parser.parse_args()

    all_chunks = []
    stats = []

    for product, rel_path in PDFS.items():
        pdf_path = os.path.join(ROOT, rel_path)
        if not os.path.exists(pdf_path):
            print(f"  ⚠️ {product}: PDF not found")
            continue

        t0 = time.time()
        chunks = process_pdf(pdf_path, product)
        elapsed = time.time() - t0

        all_chunks.extend(chunks)
        stats.append({
            "product": product,
            "chunks": len(chunks),
            "chars": sum(c["chars"] for c in chunks),
            "time_s": round(elapsed),
        })

    # Save
    out_path = os.path.join(BASE, args.output)
    with open(out_path, 'w') as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"  全文提取完成: {out_path}")
    print(f"{'='*60}")
    total_chunks = sum(s["chunks"] for s in stats)
    total_chars = sum(s["chars"] for s in stats)
    print(f"  总计: {total_chunks} chunks, {total_chars:,} chars")

    for s in stats:
        print(f"  {s['product']:>8}: {s['chunks']:>5} chunks, {s['chars']:>8,} chars ({s['time_s']}s)")


if __name__ == '__main__':
    main()
