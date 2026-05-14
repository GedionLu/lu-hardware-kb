#!/usr/bin/env python3
"""
将人工审核结果合并到主分类文件
用法: python3 pipeline/merge_manual.py

人工审核方法:
  1. 编辑 data/class_manual_review.json
  2. 修改 category / subcategory 字段
  3. 运行本脚本合并
"""

import json
import sys
import os

CLASS_PATH = os.path.join(os.path.dirname(__file__), "../data/image_classification.json")
MANUAL_PATH = os.path.join(os.path.dirname(__file__), "../data/class_manual_review.json")


def main():
    if not os.path.exists(CLASS_PATH):
        print(f"[错误] {CLASS_PATH} 不存在")
        sys.exit(1)
    if not os.path.exists(MANUAL_PATH):
        print(f"[信息] {MANUAL_PATH} 不存在，无需合并")
        return

    with open(CLASS_PATH, 'r', encoding='utf-8') as f:
        classified = json.load(f)

    with open(MANUAL_PATH, 'r', encoding='utf-8') as f:
        manual = json.load(f)

    # 按 image_id 建立索引
    classified_map = {c['image_id']: c for c in classified}

    merged_count = 0
    still_needs_review = []

    for m in manual:
        img_id = m['image_id']
        if img_id in classified_map:
            orig = classified_map[img_id]
            # 只有手动填了 category 才覆盖
            if m.get('category') and m['category'] != orig.get('category'):
                print(f"  覆盖 {img_id}: {orig.get('category')} → {m['category']}")
                orig['category'] = m['category']
                orig['confidence'] = 'manual'
                merged_count += 1

            if m.get('subcategory') and m['subcategory'] != orig.get('subcategory'):
                print(f"  覆盖子类 {img_id}: {orig.get('subcategory')} → {m['subcategory']}")
                orig['subcategory'] = m['subcategory']
                orig['sub_confidence'] = 'manual'
                merged_count += 1

            # 如果仍然缺少 category, 保留到下一轮
            if not orig.get('category'):
                still_needs_review.append(m)
        else:
            # 新加的人工条目
            classified.append(m)
            merged_count += 1

    # 写出合并结果
    with open(CLASS_PATH, 'w', encoding='utf-8') as f:
        json.dump(classified, f, ensure_ascii=False, indent=2)

    # 更新待审核列表（只保留未完成的）
    if still_needs_review:
        with open(MANUAL_PATH, 'w', encoding='utf-8') as f:
            json.dump(still_needs_review, f, ensure_ascii=False, indent=2)
        print(f"\n⚠️  仍有 {len(still_needs_review)} 张待审核")
    else:
        if os.path.exists(MANUAL_PATH):
            os.remove(MANUAL_PATH)
        print("\n✅ 所有图片审核完成")

    print(f"合并: {merged_count} 张")


if __name__ == '__main__':
    main()
