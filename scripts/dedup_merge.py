#!/usr/bin/env python3
"""
dedup_merge.py — 文档去重 + 跨文档合并

策略:
  1. 按 model + subcategory 聚合，识别同 model 同意图的重复组
  2. 合并同 model+subcategory 的步骤（去重 image + 保留最优 context_text）
  3. 跨 model 通用文档去重（19xx/14xx 系列等）
  4. 输出合并后的 image_groups.json

用法:
  python dedup_merge.py            # 分析重复，不修改
  python dedup_merge.py --apply    # 执行合并，输出 image_groups_dedup.json
"""

import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, 'data')


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def step_signature(step):
    """步骤唯一标识：优先 file_name > image_id > image_path > ref_barcode > context_text"""
    for key in ['file_name', 'image_id', 'image_path', 'ref_barcode', 'barcode_image']:
        val = step.get(key, '')
        if val:
            return val
    # 最后兜底: 用 context_text 的前 60 字符
    return step.get('context_text', '')[:60]


def step_text_quality(step):
    """评估步骤文本质量（越长 = 越可能有完整说明）"""
    text = step.get('context_text', '')
    # 去掉括号尾注和纯描述
    cleaned = re.sub(r'[（(][^)）]*[)）]', '', text)
    cleaned = re.sub(r'^[举例]*样本码[：:]', '', cleaned)
    cleaned = cleaned.strip()
    return len(cleaned)


def merge_steps(steps_list):
    """合并多个步骤列表，去重保留最优"""
    seen = {}
    for steps in steps_list:
        for s in steps:
            sig = step_signature(s)
            if not sig:
                continue
            if sig not in seen or step_text_quality(s) > step_text_quality(seen[sig]):
                seen[sig] = dict(s)  # copy

    merged = list(seen.values())
    merged.sort(key=lambda s: s.get('step_order', 99))
    # 重新编号
    for i, s in enumerate(merged):
        s['step_order'] = i + 1
    return merged


def analyze_duplicates(groups):
    """分析重复情况"""
    print("=" * 60)
    print("文档去重分析")
    print("=" * 60)

    # 1. 同 model + subcategory 的重复
    model_cat_groups = defaultdict(list)
    for g in groups:
        key = (g.get('model', 'unknown'), g.get('subcategory', 'unknown'))
        model_cat_groups[key].append(g)

    dupes = {k: v for k, v in model_cat_groups.items() if len(v) > 1}

    print(f"\n  同 model+subcategory 重复组: {len(dupes)} 对")
    total_dup_groups = sum(len(v) for v in dupes.values())
    print(f"  涉及 {total_dup_groups} 个原始组 → 可合并为 {len(dupes)} 个")

    # 统计
    dup_detail = []
    for (model, cat), grps in sorted(dupes.items(), key=lambda x: -len(x[1])):
        total_steps = sum(len(g.get('steps', [])) for g in grps)
        unique_images = len(set(
            step_signature(s) for g in grps
            for s in g.get('steps', []) if step_signature(s)
        ))
        dup_detail.append({
            'model': model, 'subcategory': cat,
            'groups': len(grps),
            'total_steps': total_steps,
            'unique_images': unique_images,
            'source_docs': [g.get('source_doc', '') for g in grps],
        })

    for d in dup_detail[:10]:
        print(f"    {d['model']}/{d['subcategory']}: {d['groups']}组 "
              f"→ {d['total_steps']}步({d['unique_images']}唯一步骤)")

    # 2. 跨 model 通用文档
    print(f"\n  跨 model 通用文档 (model 中含多个型号):")
    cross_model = [g for g in groups if ' ' in (g.get('model', '') or '')]
    print(f"    {len(cross_model)} 组")
    for g in cross_model[:5]:
        print(f"    [{g['model']}] {g['group_id'][:50]}")

    return dup_detail


def merge_groups(groups):
    """执行合并"""
    print("\n" + "=" * 60)
    print("执行合并")
    print("=" * 60)

    # 分组
    model_cat_groups = defaultdict(list)
    standalone = []

    for g in groups:
        key = (g.get('model', 'unknown'), g.get('subcategory', 'unknown'))
        model_cat_groups[key].append(g)

    merged = []
    merge_stats = {'merged_groups': 0, 'removed_groups': 0, 'total_steps_before': 0, 'total_steps_after': 0}

    for (model, cat), grps in model_cat_groups.items():
        if len(grps) == 1:
            merged.append(grps[0])
            merge_stats['total_steps_before'] += len(grps[0].get('steps', []))
            merge_stats['total_steps_after'] += len(grps[0].get('steps', []))
        else:
            # 合并
            all_steps = merge_steps([g.get('steps', []) for g in grps])
            
            # 收集 source_docs 和 applicable_models
            all_docs = []
            all_models = []
            for g in grps:
                src = g.get('source_doc', '')
                if src and src not in all_docs:
                    all_docs.append(src)
                for am in g.get('applicable_models', []):
                    if am not in all_models:
                        all_models.append(am)

            # 选最优的 title（最详细的 context 所在组）
            best_group = max(grps, key=lambda g: sum(
                step_text_quality(s) for s in g.get('steps', [])
            ))

            merged_group = {
                'group_id': best_group['group_id'],
                'title': best_group.get('title', ''),
                'subcategory': cat,
                'model': model,
                'source_docs': all_docs,
                'applicable_models': all_models,
                'total_config_codes': sum(g.get('total_config_codes', 0) for g in grps),
                'steps': all_steps,
                'merged_from': len(grps),
            }
            merged.append(merged_group)

            merge_stats['merged_groups'] += 1
            merge_stats['removed_groups'] += len(grps) - 1
            steps_before = sum(len(g.get('steps', [])) for g in grps)
            merge_stats['total_steps_before'] += steps_before
            merge_stats['total_steps_after'] += len(all_steps)

            if steps_before != len(all_steps):
                print(f"  ✓ {model}/{cat}: {len(grps)}组 → 1组 "
                      f"({steps_before}步 → {len(all_steps)}步)")

    print(f"\n  合并统计:")
    print(f"    合并组: {merge_stats['merged_groups']}")
    print(f"    移除重复: {merge_stats['removed_groups']}")
    print(f"    步骤: {merge_stats['total_steps_before']} → {merge_stats['total_steps_after']}")
    print(f"    组总数: {len(groups)} → {len(merged)}")

    return merged


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true', help='执行合并')
    args = parser.parse_args()

    groups = load_json(os.path.join(DATA, 'image_groups.json'))
    print(f"加载 {len(groups)} 个步骤组")

    analyze_duplicates(groups)

    if args.apply:
        merged = merge_groups(groups)
        out_path = os.path.join(DATA, 'image_groups_dedup.json')
        save_json(out_path, merged)
        print(f"\n  ✅ 已保存: {out_path}")
        print(f"  确认后替换: mv {out_path} {os.path.join(DATA, 'image_groups.json')}")


if __name__ == '__main__':
    main()
