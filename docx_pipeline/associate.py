#!/usr/bin/env python3
"""
Stage 3: 图片关联产品树
输入:  image_classification.json  (已分类图片)
       product_tree.csv           (人工维护的产品树 Excel 导出)
输出:  image_index.json           (最终可查询索引)
       doc_model_map.json         (文档→适用型号映射)

核心功能:
  1. 加载产品树 CSV → 生成 JSON 树 + 继承链
  2. 解析文档标题 → 提取系列关键词
  3. 用系列关键词在产品树中展开到叶子节点
  4. 每张图片绑定：适用产品节点 + 功能标签 + 前序/后序步骤
"""

import csv
import json
import os
import re
import sys

CLASS_PATH = os.path.join(os.path.dirname(__file__), "../data/image_classification.json")
TREE_CSV = os.path.join(os.path.dirname(__file__), "../data/product_tree.csv")
INDEX_OUT = os.path.join(os.path.dirname(__file__), "../data/image_index.json")
DOC_MAP_OUT = os.path.join(os.path.dirname(__file__), "../data/doc_model_map.json")

# ----------------- 1. 产品树处理 -----------------

def load_product_tree_csv(csv_path):
    """读取人工维护的产品树 CSV，构建扁平节点列表"""
    nodes = []
    if not os.path.exists(csv_path):
        print(f"[WARN] 产品树 CSV 不存在: {csv_path}")
        print("  将使用空产品树（仅返回文档名匹配的结果）")
        return nodes

    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 清理空值
            cleaned = {}
            for k, v in row.items():
                k = k.strip()
                v = v.strip()
                if v and v != '—' and v != '-':
                    cleaned[k] = v
            if cleaned:
                nodes.append(cleaned)

    print(f"  产品树加载: {len(nodes)} 个节点")
    return nodes


def build_series_map(nodes):
    """建立系列名称 → 该系列下所有型号的映射"""
    series_map = {}

    for node in nodes:
        # 系列名可能同时在 "系列" 列和 "子系列" 列
        series_names = []
        for col in ['系列', '子系列', '子系列(可选)']:
            if col in node:
                series_names.append(node[col])

        # 型号名
        model_name = node.get('型号', '')
        sn_variant = node.get('子型号/SN变体', '')
        full_name = model_name
        if sn_variant:
            full_name = f"{model_name} ({sn_variant})"
        
        # 型号别名（从型号名提取数字部分）
        aliases = [model_name]
        # 提取纯数字/字母组合
        m = re.search(r'([A-Za-z]{2,}\d+)', model_name)
        if m and m.group(1) != model_name:
            aliases.append(m.group(1))

        # 大类
        category = node.get('大类', '')

        # 建立所有层级的关键词索引
        all_tags = []
        for col in ['大类', '系列', '子系列', '子系列(可选)', '型号']:
            if col in node:
                all_tags.append(node[col])
        for alias in aliases:
            all_tags.append(alias)

        for series_name in series_names:
            if series_name not in series_map:
                series_map[series_name] = []
            series_map[series_name].append({
                'category': category,
                'series': series_name,
                'model': model_name,
                'variant': sn_variant,
                'full_name': full_name,
                'aliases': aliases,
                'tags': all_tags,
            })

        # 也按大类索引
        if category:
            if category not in series_map:
                series_map[category] = []
            series_map[category].append({
                'category': category,
                'series': series_name,
                'model': model_name,
                'variant': sn_variant,
                'full_name': full_name,
                'aliases': aliases,
                'tags': all_tags,
            })

    return series_map


# ----------------- 2. 文档标题解析 -----------------

def extract_series_tags_from_title(doc_rel_path):
    """从文档标题/路径中提取系列标签（含目录名）"""
    filename = os.path.basename(doc_rel_path)
    stem = os.path.splitext(filename)[0]
    
    # 也检查目录路径（MS7120/省电说明.docx → 可提取MS7120）
    dir_path = os.path.dirname(doc_rel_path)
    dir_tags = set()
    for part in dir_path.replace('\\', '/').split('/'):
        part = part.strip()
        if part and part != '.':
            dir_tags.add(part)

    tags = []
    
    # 匹配 "19xx" 模式
    series_patterns = [
        r'19xx', r'14xx', r'33x0', r'HH4x0', r'HH4xx', r'HH7xx', r'OH4xx',
        r'OH350', r'OH430', r'OH450', r'19x1i', r'19x2', r'190x',
        r'7680g', r'7580g', r'7120', r'HF680', r'HF600',
        r'1472', r'1470', r'1900', r'1902', r'1952',
        r'HH490', r'HH492', r'HH760', r'HH762',
        r'PM42', r'PM43', r'PX240', r'PX940', r'PC300T',
        r'SC2800', r'OH420', r'OH460', r'OH462',
        r'MS7120', r'3320g',
        r'Fiji',
    ]

    for pattern in series_patterns:
        if re.search(pattern, stem, re.IGNORECASE):
            tags.append(pattern)
        # 也匹配目录名
        for dt in dir_tags:
            if re.search(pattern, dt, re.IGNORECASE):
                tags.append(pattern)

    # 处理等字(等) - "19xx、14xx等" → 拆成多个
    etc_parts = re.split(r'[、,，/\s]+', stem)
    for part in etc_parts:
        part = part.strip()
        if not part or part == '等':
            continue
        for pattern in series_patterns:
            if pattern in part.replace('等', '').replace(' ', ''):
                if pattern not in tags:
                    tags.append(pattern)

    # 特殊: "安卓设备"、"Win11系统" 等跨型号文档
    if '安卓' in stem or 'Android' in stem:
        tags.append('__cross_model__')
    if 'Win11' in stem or 'Windows' in stem:
        tags.append('__cross_model__')

    return list(set(tags))


def build_doc_model_map(series_map, doc_paths):
    """为每份文档计算它适用的所有具体型号"""
    doc_map = {}

    for doc_path in sorted(set(doc_paths)):
        tags = extract_series_tags_from_title(doc_path)
        
        # 跨型号文档 -> 适用所有 (风险较高)
        if '__cross_model__' in tags:
            doc_map[doc_path] = {
                'tags': tags,
                'applies_to': ['__all__'],
                'match_type': 'cross_model',
            }
            continue

        # 找匹配的型号
        matched_models = []
        matched_series = set()

        for tag in tags:
            for series_name, models in series_map.items():
                # 检查 tag 是否匹配系列名或型号别名
                tag_lower = tag.lower()
                series_lower = series_name.lower()
                
                # 模糊匹配
                if tag_lower in series_lower or series_lower in tag_lower:
                    matched_series.add(series_name)
                    for m in models:
                        if m not in matched_models:
                            matched_models.append(m)
                    continue

                # 匹配型号名
                for m in models:
                    if any(tag_lower == a.lower() or tag_lower in a.lower() 
                           for a in m['aliases']):
                        if m not in matched_models:
                            matched_models.append(m)
                        matched_series.add(series_name)

        # 去重
        unique_models = []
        seen = set()
        for m in matched_models:
            key = m['full_name']
            if key not in seen:
                seen.add(key)
                unique_models.append(m)

        match_type = 'exact'
        if not unique_models:
            match_type = 'unknown'

        doc_map[doc_path] = {
            'tags': tags,
            'matched_series': list(matched_series),
            'applies_to': unique_models,
            'match_type': match_type,
        }

    return doc_map


# ----------------- 3. 图片关联 -----------------

def associate_images(classified_images, doc_model_map):
    """将每张图片关联到具体产品型号"""
    indexed = []

    for img in classified_images:
        source_doc = img['source_doc']
        
        # 找文档匹配的型号
        doc_info = doc_model_map.get(source_doc, {})
        applies_to = doc_info.get('applies_to', [])
        match_type = doc_info.get('match_type', 'unknown')

        # 如果没匹配到任何型号，保留文档名级别的信息
        if not applies_to or applies_to == ['__all__']:
            model_refs = [{
                'doc_only': True,
                'source_doc': source_doc,
                'tags': doc_info.get('tags', []),
            }]
        else:
            model_refs = []
            for m in applies_to:
                model_refs.append({
                    'category': m.get('category', ''),
                    'series': m.get('series', ''),
                    'model': m.get('model', ''),
                    'variant': m.get('variant', ''),
                    'full_name': m.get('full_name', ''),
                })

        entry = {
            'image_id': img['image_id'],
            'file_name': img['file_name'],
            'category': img.get('category'),
            'subcategory': img.get('subcategory'),
            'confidence': img.get('confidence'),
            'sub_confidence': img.get('sub_confidence'),
            'context_text': img.get('context_text', ''),
            'image_order': img.get('image_order'),
            'source_doc_rel': source_doc,
            'doc_match_type': match_type,
            'applicable_models': model_refs,
        }
        indexed.append(entry)

    return indexed


# ----------------- 主流程 -----------------

def main():
    print("===== Stage 3: 图片关联产品树 =====\n")

    # 1. 加载已分类图片
    if not os.path.exists(CLASS_PATH):
        print(f"[错误] 找不到 {CLASS_PATH}，请先运行 classify.py")
        sys.exit(1)

    with open(CLASS_PATH, 'r', encoding='utf-8') as f:
        classified_images = json.load(f)
    print(f"加载分类图片: {len(classified_images)} 张")

    # 2. 加载产品树
    print("\n[产品树]")
    nodes = load_product_tree_csv(TREE_CSV)
    series_map = build_series_map(nodes)

    # 打印产品树概览
    if series_map:
        print(f"  系列数量: {len(series_map)}")
        for sname, models in sorted(series_map.items()):
            if len(models) > 1:
                print(f"    {sname}: {len(models)} 个型号")
    else:
        print("  未加载到产品树，将使用文档名关键词匹配")

    # 3. 建立文档→型号映射
    print("\n[文档→型号映射]")
    doc_paths = [img['source_doc'] for img in classified_images]
    doc_model_map = build_doc_model_map(series_map, doc_paths)

    match_counts = {'exact': 0, 'cross_model': 0, 'unknown': 0}
    for doc_info in doc_model_map.values():
        mt = doc_info.get('match_type', 'unknown')
        if mt in match_counts:
            match_counts[mt] += 1

    print(f"  精确匹配: {match_counts['exact']}")
    print(f"  跨型号文档: {match_counts['cross_model']}")
    print(f"  未知匹配: {match_counts['unknown']}")

    # 4. 关联
    print("\n[关联]")
    image_index = associate_images(classified_images, doc_model_map)

    # 统计
    config_codes = [i for i in image_index if i.get('category') == 'config_code']
    screenshots = [i for i in image_index if i.get('category') == 'screenshot']
    unclassified = [i for i in image_index if not i.get('category')]

    print(f"  配置码: {len(config_codes)} 张")
    print(f"  截图:   {len(screenshots)} 张")
    print(f"  未分类: {len(unclassified)} 张")

    # 5. 输出
    os.makedirs(os.path.dirname(INDEX_OUT), exist_ok=True)

    with open(INDEX_OUT, 'w', encoding='utf-8') as f:
        json.dump(image_index, f, ensure_ascii=False, indent=2)

    with open(DOC_MAP_OUT, 'w', encoding='utf-8') as f:
        json.dump(doc_model_map, f, ensure_ascii=False, indent=2)

    print(f"\n输出:")
    print(f"  图片索引: {INDEX_OUT} ({len(image_index)} 条)")
    print(f"  文档映射: {DOC_MAP_OUT} ({len(doc_model_map)} 条)")


if __name__ == '__main__':
    main()
