#!/usr/bin/env python3
"""
Stage 1: 批量提取图片 → /tmp/kb-images/raw/
命名: {源文档梗概}_{顺序号}_{SHA8}.{ext}
"""

import zipfile, os, json, re, hashlib, sys
from xml.etree import ElementTree

KB_SRC = "/tmp/KnowledgeBase"
IMG_OUT = "/tmp/kb-images/raw"
META_OUT = os.path.join(os.path.dirname(__file__), "../data/image_metadata.json")

W_NS = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
WP_NS = 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'
A_NS = 'http://schemas.openxmlformats.org/drawingml/2006/main'
R_NS = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'


def extract_images_from_docx(docx_path, doc_stem):
    result = []
    try:
        z = zipfile.ZipFile(docx_path, 'r')
    except:
        return result

    try:
        doc_xml = z.read('word/document.xml')
    except KeyError:
        return result
    root = ElementTree.fromstring(doc_xml)
    body = root.find(f'{{{W_NS}}}body')
    if body is None:
        return result

    # rId → 媒体文件
    try:
        rels = ElementTree.fromstring(z.read('word/_rels/document.xml.rels'))
    except:
        rels = None
    rel_map = {}
    if rels is not None:
        for r in rels:
            rid, target = r.get('Id'), r.get('Target')
            if rid and target and 'media/' in target:
                rel_map[rid] = target

    # 源文档路径哈希（防同名冲突）
    doc_hash = hashlib.md5(docx_path.encode()).hexdigest()[:6]

    # 遍历段落，收集图片 + 带上下文
    paragraphs = list(body)
    image_order = 0

    # 前一段文本（给图片做上下文）
    prev_text = ""

    for para in paragraphs:
        tag = para.tag.split('}')[-1] if '}' in para.tag else para.tag
        if tag != 'p':
            continue

        # 本段所有文本
        texts = [t.text or '' for t in para.iter(f'{{{W_NS}}}t')]
        para_text = ''.join(texts).strip()

        # 找本段所有内联图片
        inlines = list(para.iter(f'{{{WP_NS}}}inline'))
        if not inlines:
            if para_text:
                prev_text = para_text
            continue

        for inline in inlines:
            blips = list(inline.iter(f'{{{A_NS}}}blip'))
            for blip in blips:
                embed = blip.get(f'{{{R_NS}}}embed')
                if not embed or embed not in rel_map:
                    continue

                target = rel_map[embed]
                ext = os.path.splitext(target)[1].lower()
                img_zip_path = f"word/{target}" if not target.startswith('word/') else target
                try:
                    img_data = z.read(img_zip_path)
                except KeyError:
                    continue

                info = z.getinfo(img_zip_path)
                sha8 = hashlib.sha256(img_data).hexdigest()[:8]
                image_order += 1

                # 从源文档名提取有意义的简称
                brief = shorten_doc_name(doc_stem)

                # 文件名 = 简短梗概 + 顺序 + 文档哈希 + 内容hash
                out_name = f"{brief}_s{image_order:03d}_{doc_hash}_{sha8}{ext}"
                out_path = os.path.join(IMG_OUT, out_name)

                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                with open(out_path, 'wb') as f:
                    f.write(img_data)

                result.append({
                    "image_id": f"{sha8}_{doc_hash}",
                    "file_name": out_name,
                    "file_path": out_path,
                    "file_size": info.file_size,
                    "format": ext,
                    "source_doc": docx_path,
                    "source_doc_stem": doc_stem,
                    "image_order": image_order,
                    "context_text": para_text if para_text else prev_text,
                })

                if para_text:
                    prev_text = para_text

    z.close()
    return result


def shorten_doc_name(stem):
    """从完整的 doc_stem 中提取有意义的简短梗概"""
    # 去掉路径分隔符替换出的下划线，保留核心信息
    # 取文档名的后一段（通常是文件名本身）
    parts = stem.split('_')
    # 看能不能找到模型名或功能关键词
    keywords = []
    for kw_key in ['1900', '1902', '1952', '1472', '1470', 'OH430', 'OH450', 
                    'HF680', 'HH490', 'HH760', 'HH762', 'HH492', 'OH462',
                    '7680g', '7580g', '7120', 'SC2800', '3320g', '33x0g',
                    '1991i', '1981i', '1911i', '1990i', '190x', '19xx',
                    '14xx', 'HH4X0', 'OH350', 'OH420', 'OH460',
                    'PM42', 'PM43', 'PX240', 'PX940', 'PC300T',
                    '1202g', '8680i',
                    'USB虚拟串口', '恢复出厂', '配对', '回车', '后缀', '前缀',
                    '串口', '蓝牙', '中文', 'Ezconfig', 'DPM', 'OCR',
                    '截取', '序列扫描', '数据替换', 'DATREP',
                    'Win11', '安卓',
                    'Fiji', '固件', '校准']:
        for p in parts:
            if kw_key.lower() in p.lower():
                keywords.append(kw_key)
                break

    if keywords:
        brief = '_'.join(keywords[:3])
    else:
        # 取最后两个部分
        brief = '_'.join(parts[-2:]) if len(parts) >= 2 else parts[-1]
    
    # 限制长度
    brief = brief[:50]
    return brief


def main():
    os.makedirs(IMG_OUT, exist_ok=True)
    all_images = []
    total_docx = 0

    for root, dirs, files in os.walk(KB_SRC):
        for f in sorted(files):
            if not f.endswith('.docx'):
                continue
            docx_path = os.path.join(root, f)
            doc_rel = os.path.relpath(docx_path, KB_SRC)
            doc_stem = doc_rel.replace('/', '_').replace('\\', '_')
            doc_stem = os.path.splitext(doc_stem)[0]

            total_docx += 1
            sys.stdout.write(f"\r[提取] {total_docx}/154")
            sys.stdout.flush()

            images = extract_images_from_docx(docx_path, doc_stem)
            all_images.extend(images)

    print(f"\n\n===== 完成 =====")
    print(f"docx: {total_docx}, 图片: {len(all_images)}")

    os.makedirs(os.path.dirname(META_OUT), exist_ok=True)
    with open(META_OUT, 'w', encoding='utf-8') as f:
        json.dump(all_images, f, ensure_ascii=False, indent=2)

    print(f"meta: {META_OUT}")
    print(f"imgs: {IMG_OUT}/")


if __name__ == '__main__':
    main()
