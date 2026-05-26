#!/usr/bin/env python3 -u
"""
extract_pdf_images.py — PDF 非条码图片提取 + 分类

从 UG PDF 提取所有嵌入图片，分类为:
  - product_photo: 产品实物图 (>200px, 照片级色彩)
  - diagram: 接线图/示意图 (>300px, 色彩简单)
  - screenshot: 软件截图 (>400px)
  - icon: 小图标 (<80px)
  - barcode: 条码 (YOLO 已覆盖, 跳过)
  - decorative: 装饰元素

策略:
  1. PyMuPDF get_images() 提取所有嵌入图
  2. 启发式分类 (尺寸/长宽比/色彩复杂度)
  3. 可选 VLM 二次确认
  4. 关联到最近文本块获取上下文

用法: python extract_pdf_images.py [-o kb-images] [--vlm]
"""

import fitz, json, os, re, argparse, io, time
from collections import defaultdict, Counter
from pathlib import Path

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.expanduser('~/openclaw/19Series')

PDFS = {
    'XEN197X': '197x/XEN197X-EN-UG-01 Rev B 2026.1.19.pdf',
    '1900': 'Xenon-UG.pdf',
    '195X': '195x/XEN195X-EN-UG.pdf',
    '199X': 'Granit XP/199x-en-ug.pdf',
    '199Xi': 'Granit XP/Granit XP 199xi Series UG(1990iSR 1991iXR 1991iXLR 1991iSR 1991iXR 1991iXLR).pdf',
    '196X': '1960/sps-ppr-xen196x-en-ug.pdf',
    'OCR': 'OCR-UG Rev B pdf.pdf',
}


def classify_image(img_bytes, w, h):
    """启发式图片分类"""
    # 尺寸分类
    if w < 80 and h < 80:
        return 'icon'
    if w > 300 and h > 300 and abs(w - h) < w * 0.3:
        return 'product_photo'  # 近似正方形的大图
    
    # 分析色彩复杂度
    try:
        from PIL import Image
        im = Image.open(io.BytesIO(img_bytes))
        if im.mode == 'RGB':
            # 采样分析色彩数
            im_small = im.resize((50, 50))
            colors = len(set(im_small.getdata()))
            im.close()
            if colors < 50:
                return 'diagram'  # 色彩简单的图
            elif colors < 200 and w > 200:
                return 'screenshot'
            else:
                return 'product_photo' if w > 200 else 'diagram'
        im.close()
    except:
        pass
    
    # 默认
    if w > 200:
        return 'diagram'
    return 'icon'


def extract_images_from_page(page, pn):
    """提取单页所有嵌入图片"""
    imgs = page.get_images(full=True)
    results = []
    
    for img in imgs:
        xref = img[0]
        w, h = img[2], img[3]
        
        # 跳过极小装饰元素
        if w < 20 or h < 20:
            continue
        
        try:
            base = page.parent.extract_image(xref)
            img_bytes = base['image']
            ext = base['ext']
        except:
            continue
        
        # 分类
        category = classify_image(img_bytes, w, h)
        
        results.append({
            'xref': xref,
            'width': w, 'height': h,
            'ext': ext,
            'category': category,
            'bytes': img_bytes,
            'size_bytes': len(img_bytes),
        })
    
    return results


def find_nearby_text(page, pn):
    """提取页面上附近文本作为图片上下文"""
    try:
        text = page.get_text("text")
        # 取前200字符作为页面上下文
        return text[:200].replace('\n', ' ').strip()
    except:
        return ""


def process_pdf(pdf_path, product, output_base):
    """处理单个 PDF"""
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    print(f"  📄 {product}: {total_pages} 页")
    
    out_dir = os.path.join(output_base, product, 'images')
    os.makedirs(out_dir, exist_ok=True)
    
    # 如果已提取过
    existing = len(os.listdir(out_dir)) if os.path.exists(out_dir) else 0
    if existing > 10:
        print(f"    ✅ {existing} images exist, skipping")
        return existing, Counter()
    
    stats = Counter()
    extracted = 0
    
    for pn in range(total_pages):
        page = doc[pn]
        images = extract_images_from_page(page, pn)
        
        if not images:
            continue
        
        # 只取有意义的 (非 icon/barcode/decorative)
        for img in images:
            cat = img['category']
            stats[cat] += 1
            
            if cat in ('icon', 'decorative'):
                continue  # 跳过小图标
            
            # 保存
            fname = f'p{pn+1}_{img["xref"]}_{cat}.{img["ext"]}'
            fpath = os.path.join(out_dir, fname)
            
            if not os.path.exists(fpath):
                with open(fpath, 'wb') as f:
                    f.write(img['bytes'])
                extracted += 1

    doc.close()
    print(f"    → {extracted} images saved ({dict(stats)})")
    return extracted, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', default='kb-images')
    parser.add_argument('--vlm', action='store_true', help='VLM 二次分类')
    args = parser.parse_args()

    output_base = os.path.join(BASE, args.output)
    total_imgs = 0
    all_stats = Counter()

    for product, rel_path in PDFS.items():
        pdf_path = os.path.join(ROOT, rel_path)
        if not os.path.exists(pdf_path):
            print(f"  ⚠️ {product}: not found")
            continue

        t0 = time.time()
        extracted, stats = process_pdf(pdf_path, product, output_base)
        elapsed = time.time() - t0
        
        total_imgs += extracted
        all_stats.update(stats)

    print(f"\n{'='*60}")
    print(f"  非条码图片提取完成")
    print(f"{'='*60}")
    print(f"  总计: {total_imgs} 张")
    print(f"  分类: {dict(all_stats.most_common())}")

    # 更新 image_index (追加)
    idx_path = os.path.join(BASE, 'data', 'image_index.json')
    if os.path.exists(idx_path):
        with open(idx_path) as f:
            index = json.load(f)
    else:
        index = []

    added = 0
    for product in PDFS:
        img_dir = os.path.join(output_base, product, 'images')
        if not os.path.exists(img_dir):
            continue
        for fn in os.listdir(img_dir):
            if fn.endswith(('.png', '.jpg', '.jpeg')):
                # 检查是否已在索引中
                if not any(i.get('file_name') == fn for i in index):
                    index.append({
                        'file_name': fn,
                        'category': 'product_photo' if 'product_photo' in fn else 'diagram',
                        'image_url': f'http://172.24.59.194:8098/kb-images/{product}/images/{fn}',
                        'source': product,
                    })
                    added += 1

    with open(idx_path, 'w') as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    print(f"  image_index: +{added} entries → {len(index)} total")


if __name__ == '__main__':
    main()
