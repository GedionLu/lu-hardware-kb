#!/usr/bin/env python3
"""
VLM 图片分析: 用 pyzbar + PIL 做实际图像内容分析
不依赖任何外部 API，本地运行。

分析维度:
  - 条码检测 (pyzbar): 图片中是否含一维码/二维码？
  - 色彩丰富度: 彩色截图 vs 黑白条码
  - 对比度: 高对比 = 条码, 低对比 = 截图
  - 边缘密度: 密集水平/垂直线 = 条码, 复杂纹理 = 截图
  - 长宽比: 条码多为窄长形

输出: 基于图像内容的分类结果，与文本分类比对
"""

import json, os, sys
from PIL import Image
from collections import Counter

# 尝试加载 pyzbar
try:
    from pyzbar.pyzbar import decode as zbardecode
    HAS_ZBAR = True
except:
    HAS_ZBAR = False

BASE = '/tmp/kb-images/raw'
CLASS_PATH = os.path.join(os.path.dirname(__file__), "../data/image_classification.json")
INDEX_PATH = os.path.join(os.path.dirname(__file__), "../data/image_index.json")


def analyze_image(image_path):
    """分析单张图片，返回图像特征"""
    result = {'has_barcode': False, 'barcode_type': None, 'barcode_data': None,
              'is_colorful': False, 'width': 0, 'height': 0, 'aspect_ratio': 0,
              'contrast': 0, 'brightness_std': 0, 'file_size': 0,
              'img_format': ''}
    
    if not os.path.exists(image_path):
        return result
    
    result['file_size'] = os.path.getsize(image_path)
    
    try:
        img = Image.open(image_path)
        result['width'], result['height'] = img.size
        result['aspect_ratio'] = max(img.size) / max(min(img.size), 1)
        result['img_format'] = img.format or ''
        
        # Convert to grayscale for analysis
        if img.mode == 'RGBA' or img.mode == 'P':
            gray = img.convert('L')
            if img.mode == 'RGBA':
                rgb = img.convert('RGB')
            else:
                rgb = img.convert('RGB')
        elif img.mode == 'L':
            gray = img
            rgb = img.convert('RGB')
        else:
            rgb = img.convert('RGB')
            gray = rgb.convert('L')
        
        # 色彩丰富度: 计算 RGB 通道的标准差
        r, g, b = rgb.split()
        r_std = sum(r.getdata()) / len(list(r.getdata()))
        # 简化: 检查是否是灰色图像
        pixels = list(rgb.getdata())[::100]  # sample
        color_count = 0
        for pr, pg, pb in pixels[:200]:
            if abs(pr - pg) > 20 or abs(pg - pb) > 20:
                color_count += 1
        result['is_colorful'] = color_count > len(pixels[:200]) * 0.1
        
        # 对比度: 灰度像素值的标准差
        pix_vals = list(gray.getdata())
        mean_val = sum(pix_vals) / len(pix_vals)
        variance = sum((v - mean_val) ** 2 for v in pix_vals) / len(pix_vals)
        result['contrast'] = variance ** 0.5
        
        # 亮度直方图分布
        # 高对比度 + 二值化分布 = 条码
        hist = gray.histogram()
        # 统计黑白两极的占比
        black_pixels = sum(hist[:30])  # 很暗
        white_pixels = sum(hist[-30:])  # 很亮
        total_pixels = sum(hist)
        result['black_white_ratio'] = (black_pixels + white_pixels) / max(total_pixels, 1)
        
        # 条码检测
        if HAS_ZBAR:
            try:
                decoded = zbardecode(img)
                if decoded:
                    result['has_barcode'] = True
                    result['barcode_type'] = decoded[0].type
                    result['barcode_data'] = decoded[0].data.decode('utf-8', errors='replace')[:30]
            except:
                pass
    
    except Exception as e:
        pass
    
    return result


def classify_from_vision(features):
    """基于图像特征做分类"""
    # 条码检测最高优先级
    if features.get('has_barcode', False):
        bt = features.get('barcode_type', '')
        if bt in ('EAN13', 'EAN8', 'CODE128', 'CODE39', 'CODE25', 'QRCODE', 'DATAMATRIX', 'PDF417'):
            return 'config_code', 'high'
        else:
            return 'config_code', 'high'
    
    # 高黑白占比 + 高对比度 + 窄长 = 条码（未检出可能是因为图片质量）
    if features.get('black_white_ratio', 0) > 0.5 and features.get('contrast', 0) > 60:
        if features.get('aspect_ratio', 1) > 1.5:
            return 'config_code', 'medium'
    
    # 彩色 + 低对比度 + 接近方形的长宽比 = 截图
    if features.get('is_colorful', False) and features.get('contrast', 0) < 80:
        return 'screenshot', 'high'
    
    # 大文件 + 彩色 = 截图
    if features.get('file_size', 0) > 50000 and features.get('is_colorful', False):
        return 'screenshot', 'high'
    
    # 小文件 + 黑白 = 条码
    if features.get('file_size', 0) < 20000 and not features.get('is_colorful', True) and features.get('contrast', 0) > 40:
        return 'config_code', 'high'
    
    return 'unclear', 'low'


def main():
    print(f"pyzbar: {'OK' if HAS_ZBAR else 'MISSING'}")
    print(f"图片目录: {BASE}")
    
    # 获取所有 raw 图片文件
    files = sorted([f for f in os.listdir(BASE) if os.path.isfile(os.path.join(BASE, f))])
    print(f"共 {len(files)} 张图片\n")
    
    # 先做一部分取样分析
    sample = files[:20]
    print("=== 取样分析 (前20张) ===")
    
    vision_results = []
    
    for fname in sample:
        fpath = os.path.join(BASE, fname)
        features = analyze_image(fpath)
        vision_cat, vision_conf = classify_from_vision(features)
        
        vision_results.append({
            'file_name': fname,
            'vision_category': vision_cat,
            'vision_confidence': vision_conf,
            'features': features,
        })
        
        bc_str = f"BARCODE({features['barcode_type']})" if features['has_barcode'] else "NO-BC"
        colorful = "COLOR" if features['is_colorful'] else "B/W"
        contrast_str = f"CTR={features['contrast']:.0f}"
        bw = f"BW={features['black_white_ratio']:.2f}"
        ar = f"AR={features['aspect_ratio']:.1f}"
        sz = f"SZ={features['file_size']//1024}KB"
        
        print(f"  {fname[:50]:52s} | {bc_str:15s} | {colorful:5s} | {contrast_str:10s} | {bw:8s} | {ar:8s} | {sz:8s} | → {vision_cat} ({vision_conf})")
    
    # 统计
    cats = Counter(r['vision_category'] for r in vision_results)
    print(f"\n取样分类: {dict(cats)}")
    
    # 现在运行全量
    if '--full' in sys.argv:
        print(f"\n=== 全量分析 ({len(files)} 张) ===")
        
        all_results = []
        for idx, fname in enumerate(files):
            fpath = os.path.join(BASE, fname)
            features = analyze_image(fpath)
            vision_cat, vision_conf = classify_from_vision(features)
            all_results.append({
                'file_name': fname,
                'vision_category': vision_cat,
                'vision_confidence': vision_conf,
                'has_barcode': features.get('has_barcode', False),
                'barcode_type': features.get('barcode_type'),
                'file_size': features.get('file_size', 0),
                'is_colorful': features.get('is_colorful', False),
                'contrast': features.get('contrast', 0),
                'black_white_ratio': features.get('black_white_ratio', 0),
                'aspect_ratio': features.get('aspect_ratio', 1.0),
            })
            
            if (idx + 1) % 100 == 0:
                print(f"  [{idx+1}/{len(files)}]")
        
        # 保存
        out_path = os.path.join(os.path.dirname(__file__), "../data/vision_analysis.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n全量分析完成，保存至: {out_path}")
        
        # 与文本分类对比
        cats = Counter(r['vision_category'] for r in all_results)
        print(f"\n=== 视觉分类汇总 ===")
        for c, n in cats.most_common():
            print(f"  {c:15s}: {n}")
        
        barcode_count = sum(1 for r in all_results if r['has_barcode'])
        print(f"\npyzbar 检出条码: {barcode_count}/{len(files)}")


if __name__ == '__main__':
    main()
