#!/usr/bin/env python3
"""
validate_steps.py — 步骤质量校验 (V1-V3 规则 + V4-V6 LLM)

V1: 组级字段完整性扫描 (subcategory, title, model)
V2: 步骤 subcategory 分类修复
V3: 模型关联校验 (applicable_models + doc_model_map)
V4: LLM 组级元数据推断 (subcategory/title/model)
V5: LLM 步骤 subcategory 校验
V6: 修复建议生成 + 输出修复文件

用法:
  python validate_steps.py          # 规则扫描 (V1-V3)
  python validate_steps.py --fix    # 规则扫描 + 自动修复 + 输出修复文件
  python validate_steps.py --llm    # 规则 + LLM 深度校验 (V4-V6)
  python validate_steps.py --all    # 全部 + 自动修复
"""

import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, 'data')

# ─── INTENT KEYWORDS (from intents.yaml) ───
INTENT_KEYWORDS = {
    'usb_connect':       ['USB', '电脑', '连接', '线缆', '键盘口'],
    'serial_connect':    ['串口', 'RS232', 'serial', 'COM', 'com口'],
    'bluetooth_pairing': ['配对', '蓝牙', '无线', '底座'],
    'virtual_com_port':  ['虚拟串口', 'USB虚拟串口', 'USB仿真串口', '转串口'],
    'add_suffix':        ['回车', '换行', '后缀', 'CRLF', '自动换行', '自动回车'],
    'add_prefix':        ['前缀', '自定义前缀', '加前缀'],
    'restore_factory':   ['恢复出厂', '重置', '初始化', '恢复默认', '清空'],
    'test_comm':         ['测试', '通信验证', '没反应', '扫不出', '不好使', '不行', '没数据', '没输出', '扫了没'],
    'chinese_qr':        ['中文', '中文字符', '汉字'],
    'data_format':       ['截取', '格式编辑', '数据替换', '序列扫描', '只输出', '过滤', 'OCR'],
    'no_read':           ['No Read', '无读取', '空扫'],
    'interface_mode':    ['接口码', '接口模式', '接口设置'],
    'pairing':           ['配对码', '重新配对'],
    'product_overview':  ['介绍', '功能', '说明书', '概述', '参数', '规格', '是什么', '有哪些', '能做什么'],
    'general_setup':     ['设置', '配置', '使用', '怎么用', '说明', '操作', '调试'],
}

# ─── MODEL PATTERNS (from query.py) ───
MODEL_PATTERNS = [
    (r'1900[^2]', '1900'), (r'1902', '1902'), (r'1952', '1952'), (r'1910', '1910i'),
    (r'1911', '1911i'), (r'1912', '1912i'), (r'1920', '1920i'), (r'1990', '1990i'),
    (r'1470[^gG2]', '1470g'), (r'1472', '1472'),
    (r'HH[479]60', 'HH760'), (r'HH[479]62', 'HH762'), (r'HH[479]90', 'HH490'),
    (r'HH[479]92', 'HH492'),
    (r'7120|MS7120', 'MS7120'), (r'1202g', '1202g'),
    (r'PM42', 'PM42'), (r'PM43', 'PM43'), (r'PM45', 'PM45'), (r'PX240', 'PX240'),
    (r'PX940', 'PX940'), (r'PC300T', 'PC300T'), (r'8680i', '8680i'),
    (r'XEN197X|197[Xx]|197[02]', 'XEN197X'),
    (r'OH[48][35]0', 'OH430'),
    (r'19[0-9]{2}', '19xx'),
    (r'14[0-9]{2}', '14xx'),
    (r'33[0-9]{2}[gi]?', '33xx'),
    (r'76[0-9]{2}[gi]?', '7680g'),
    (r'71[0-9]{2}[gi]?', '7120'),
]

# ─── STEP CLASSIFICATION RULES ───
STEP_CLASSIFY_RULES = [
    (r'恢复出厂|重置|初始化|恢复默认', 'restore_factory', 10),
    (r'回车|换行|后缀|CR[ ]*LF|后缀码|VSUFCR|SUFCR|DEFALT|DFMBK[23]|DFMCA[23]', 'suffix', 10),
    (r'前缀|前缀码|PRECR', 'prefix', 10),
    (r'配对码|底座配对码|配对|LNKBT|蓝牙|无线配对|重新配对', 'pairing', 10),
    (r'接口码|接口模式|接口设置|USB键盘口|USB虚拟串口|串口模式|PAP  |USB.*COM', 'interface', 9),
    (r'测试|通信验证|确认.*输出|记事本|文本端', 'test_code', 8),
    (r'断开.*连接|断开枪|DELINK|断开.*底座|取消配对', 'pairing', 6),
    (r'功能|条码|码制|开启|关闭|启[用停]|解码|DPM|读取', 'feature', 7),
    (r'设置|配置|模式|参数|波特率|COM.*参数', 'setup', 5),
    (r'USB.*连接|线缆|键盘口|COM.*口|串口|U口|USB', 'interface', 6),
    (r'No Read|空扫|NOREA', 'feature', 6),
]


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def extract_models_from_text(text):
    found = []
    for pattern, model in MODEL_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            if model not in found:
                found.append(model)
    return found


def extract_models_from_group(group):
    texts = [group.get('group_id', ''), group.get('doc_name', '')]
    for s in group.get('steps', []):
        texts.append(s.get('context_text', ''))
    return extract_models_from_text(' '.join(texts))


def infer_subcategory_from_steps(group):
    subcats = Counter()
    for s in group.get('steps', []):
        sc = s.get('subcategory')
        if sc:
            subcats[sc] += 1
    if not subcats:
        return None
    return subcats.most_common(1)[0][0]


def infer_title_from_group(group):
    gid = group.get('group_id', '')
    title = re.sub(r'^g_', '', gid)
    title = re.sub(r'\([^)]*\)', '', title)
    title = title.strip('、，, ')
    if len(title) > 45:
        title = title[:45]
    return title if title else None


def classify_step(text):
    if not text:
        return None
    best = None
    best_score = 0
    for pattern, subcat, priority in STEP_CLASSIFY_RULES:
        if re.search(pattern, text, re.IGNORECASE):
            if priority > best_score:
                best_score = priority
                best = subcat
    return best


# ═══ V1: 组级字段完整性 ═══

def v1_group_integrity(groups):
    print("\n" + "=" * 60)
    print("V1: 组级字段完整性扫描")
    print("=" * 60)

    issues = []
    auto_fixes = []

    for i, g in enumerate(groups):
        gid = g.get('group_id', f'unknown_{i}')
        fields_missing = []
        fixes = {}

        if not g.get('subcategory'):
            fields_missing.append('subcategory')
            inferred = infer_subcategory_from_steps(g)
            if inferred:
                fixes['subcategory'] = inferred

        if not g.get('title'):
            fields_missing.append('title')
            inferred = infer_title_from_group(g)
            if inferred:
                fixes['title'] = inferred

        if not g.get('model'):
            fields_missing.append('model')
            models = extract_models_from_group(g)
            if models:
                fixes['model'] = models[0]
            elif g.get('applicable_models'):
                tags = []
                for am in g['applicable_models']:
                    tags.extend(am.get('tags', []))
                if tags:
                    fixes['model'] = tags[0]

        if fields_missing:
            step_subcats = Counter(
                s.get('subcategory') for s in g.get('steps', [])
                if s.get('subcategory')
            )
            issues.append({
                'group_id': gid,
                'missing_fields': fields_missing,
                'step_subcats': dict(step_subcats.most_common()),
                'auto_fix': fixes if any(fixes.values()) else None,
            })
            if fixes:
                auto_fixes.append((gid, fixes))

    field_stats = Counter()
    for iss in issues:
        for f in iss['missing_fields']:
            field_stats[f] += 1

    print(f"  组总数: {len(groups)}")
    print(f"  有问题组: {len(issues)}")
    for field, count in field_stats.most_common():
        print(f"    {field}: {count}/{len(groups)} 缺失")
    print(f"  可规则修复: {len(auto_fixes)} 组")

    return issues, auto_fixes


# ═══ V2: 步骤 subcategory ═══

def v2_step_subcategories(groups):
    print("\n" + "=" * 60)
    print("V2: 步骤 subcategory 分类校验")
    print("=" * 60)

    null_steps = []
    classified = []
    all_subcats = Counter()

    for g in groups:
        for s in g.get('steps', []):
            subcat = s.get('subcategory')
            all_subcats[subcat] += 1
            if subcat is None:
                text = s.get('context_text', '')
                c = classify_step(text)
                null_steps.append({
                    'group_id': g['group_id'],
                    'step_order': s.get('step_order'),
                    'file_name': s.get('file_name', ''),
                    'context_text': text[:120],
                    'classified': c,
                })
                if c:
                    classified.append(c)

    print(f"  步骤总数: {sum(all_subcats.values())}")
    print(f"  subcategory=None: {all_subcats[None]} 个")
    print(f"  规则可分类: {len(classified)} / {all_subcats[None]}")
    print(f"  还需人工/LLM: {all_subcats[None] - len(classified)} 个")

    unclassified = [ns for ns in null_steps if not ns['classified']]
    if unclassified:
        print(f"\n  无法规则分类 (前5):")
        for ns in unclassified[:5]:
            print(f"    [{ns['group_id'][:45]}] s{ns['step_order']}")
            print(f"      {ns['context_text'][:100]}")

    return null_steps


# ═══ V3: 模型关联 ═══

def v3_model_association(groups, doc_model_map=None):
    print("\n" + "=" * 60)
    print("V3: 模型关联校验")
    print("=" * 60)

    if doc_model_map is None:
        doc_model_map = {}

    no_tag_groups = set()
    missing_tag_groups = set()
    suggestions = {}

    for g in groups:
        gid = g['group_id']
        source = g.get('source_doc', '')
        am = g.get('applicable_models', [])

        all_tags = []
        for entry in am:
            tags = entry.get('tags', [])
            if not tags and source in doc_model_map:
                dmm_tags = doc_model_map[source].get('tags', [])
                if dmm_tags and '__cross_model__' not in dmm_tags:
                    missing_tag_groups.add(gid)
                    if gid not in suggestions:
                        suggestions[gid] = dmm_tags
            all_tags.extend(tags)

        if not all_tags:
            extracted = extract_models_from_group(g)
            if extracted:
                no_tag_groups.add(gid)
                suggestions[gid] = extracted

    print(f"  无任何 tags: {len(no_tag_groups)} 组")
    print(f"  tags 可从 doc_model_map 补充: {len(missing_tag_groups)} 组")
    print(f"  合计有问题: {len(no_tag_groups | missing_tag_groups)} 组")

    for gid in list(no_tag_groups)[:3]:
        print(f"    [{gid[:50]}] 提取到: {suggestions.get(gid, [])}")

    return no_tag_groups, missing_tag_groups, suggestions


# ═══ V4-V5: LLM ═══

def load_api_config():
    api_key = os.environ.get('DEEPSEEK_API_KEY', os.environ.get('OPENAI_API_KEY', ''))
    base_url = os.environ.get('DEEPSEEK_BASE_URL', 'https://api.deepseek.com/v1')

    if not api_key:
        config_paths = [
            os.path.expanduser('~/.openclaw/openclaw.json'),
            '/home/admin/.openclaw/openclaw.json',
        ]
        for cp in config_paths:
            try:
                with open(cp) as f:
                    cfg = json.load(f)
                ds = cfg.get('models', {}).get('providers', {}).get('deepseek', {})
                api_key = ds.get('apiKey', '')
                raw_base = ds.get('baseUrl', '')
                # openclaw config baseUrl 不带 /v1，需要补上
                if raw_base and '/v1' not in raw_base:
                    raw_base += '/v1'
                if raw_base:
                    base_url = raw_base
                if api_key:
                    break
            except:
                continue
    return api_key, base_url


def call_deepseek(prompt, model="deepseek-v4-flash"):
    import urllib.request

    api_key, base_url = load_api_config()
    if not api_key:
        print("  ⚠️ API key not found")
        return None

    data = json.dumps({
        'model': model,
        'messages': [
            {'role': 'user', 'content': prompt},
        ],
        'temperature': 0.1,
        'max_tokens': 8192,
    }).encode('utf-8')

    try:
        url = f'{base_url}/chat/completions'
        req = urllib.request.Request(url, data=data, headers={
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
        })
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read())
            content = result['choices'][0]['message']['content'].strip()
            if content.startswith('```'):
                content = re.sub(r'^```(?:json)?\s*', '', content)
                content = re.sub(r'\s*```$', '', content)
            return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"  ⚠️ JSON parse error: {e}")
        return None
    except Exception as e:
        print(f"  ⚠️ LLM error: {e}")
        return None


def v4_llm_group_metadata(groups, issues_v1, max_batch=5):
    print("\n" + "=" * 60)
    print("V4: LLM 组级元数据推断")
    print("=" * 60)

    problem_ids = {iss['group_id'] for iss in issues_v1}
    problem_groups = [g for g in groups if g['group_id'] in problem_ids]

    if not problem_groups:
        print("  无需处理")
        return {}

    all_fixes = {}
    total_batches = (len(problem_groups) + max_batch - 1) // max_batch

    for batch_idx in range(total_batches):
        start = batch_idx * max_batch
        batch = problem_groups[start:start + max_batch]

        items = []
        for g in batch:
            steps_text = '; '.join(
                f"s{s.get('step_order')}[{s.get('subcategory') or '?'}] "
                f"{s.get('context_text','')[:50]}"
                for s in g.get('steps', [])[:5]
            )
            items.append({'gid': g['group_id'], 'steps': steps_text})

        prompt = f"""分析该操作步骤组，推断 subcategory, title, model。

subcategory: restore_factory/suffix/prefix/pairing/interface/setup/feature/test_code
model举例: 1900 1902 1472 HH760 HH490 OH430 7120 1202g XEN197X

{json.dumps(items, ensure_ascii=False, indent=2)}

返回: {{"r":[{{"gid":"...","subcategory":"...","title":"...","model":"..."}}, ...]}}"""

        result = call_deepseek(prompt, model="deepseek-chat")
        if result and 'r' in result:
            for r in result['r']:
                all_fixes[r['gid']] = {k: v for k, v in r.items() if k != 'gid'}
            print(f"  批次 {batch_idx+1}/{total_batches}: ✓ {len(result['r'])} 组")
        else:
            print(f"  批次 {batch_idx+1}/{total_batches}: ✗")
            break

    print(f"  LLM 推断完成: {len(all_fixes)} 组")
    return all_fixes


def v5_llm_step_subcategories(null_steps, max_batch=16):
    print("\n" + "=" * 60)
    print("V5: LLM 步骤 subcategory 校验")
    print("=" * 60)

    unclassified = [ns for ns in null_steps if not ns['classified']]
    if not unclassified:
        print("  所有步骤已规则分类")
        return {}

    batch = unclassified[:max_batch]
    print(f"  {len(batch)} / {len(unclassified)} 个未分类步骤")

    items = [{'idx': i, 'text': s['context_text']} for i, s in enumerate(batch)]

    prompt = f"""为以下步骤文本分配 subcategory。

可选值及含义:
- restore_factory: 恢复出厂、重置
- suffix: 添加回车/换行后缀
- prefix: 添加前缀
- pairing: 配对、断开配对
- interface: USB/串口连接
- setup: 通用设置
- feature: 功能开关
- test_code: 测试通信

{json.dumps(items, ensure_ascii=False, indent=2)}

返回: {{"r":[{{"idx":0,"subcategory":"setup"}}, ...]}}"""

    result = call_deepseek(prompt, model="deepseek-chat")
    if result and 'r' in result:
        fixes = {r['idx']: r['subcategory'] for r in result['r']}
        print(f"  LLM 分类完成: {len(fixes)} 个")
        return fixes
    return {}


# ═══ V6: 生成修复文件 ═══

def v6_generate_fixes(groups, fixes_v1, fixes_v4, step_llm_fixes, null_steps,
                       no_tag_groups, missing_tag_groups, tag_suggestions):
    print("\n" + "=" * 60)
    print("V6: 修复建议生成")
    print("=" * 60)

    fix_count = 0
    step_fix_count = 0

    fixes_v1_map = {gid: fix for gid, fix in fixes_v1}

    for g in groups:
        gid = g['group_id']

        # V1 规则修复
        if gid in fixes_v1_map:
            for field, value in fixes_v1_map[gid].items():
                if not g.get(field):
                    g[field] = value
                    fix_count += 1

        # V4 LLM 修复（覆盖 V1）
        if gid in fixes_v4:
            for field in ['subcategory', 'title', 'model']:
                if field in fixes_v4[gid] and fixes_v4[gid][field]:
                    old = g.get(field)
                    g[field] = fixes_v4[gid][field]
                    if old != g[field]:
                        fix_count += 1

        # V3 模型 tags 修复
        if gid in tag_suggestions:
            am = g.get('applicable_models', [])
            if not am:
                g['applicable_models'] = [{'tags': tag_suggestions[gid], 'source': 'validate_steps'}]

        # 步骤 subcategory 修复
        for s in g.get('steps', []):
            if s.get('subcategory') is None:
                c = classify_step(s.get('context_text', ''))
                if c:
                    s['subcategory'] = c
                    step_fix_count += 1

    # LLM 步骤修复
    if step_llm_fixes:
        for i, ns in enumerate(null_steps):
            if i in step_llm_fixes and ns.get('classified') is None:
                for g in groups:
                    if g['group_id'] == ns['group_id']:
                        for s in g.get('steps', []):
                            if (s.get('step_order') == ns['step_order'] and
                                s.get('subcategory') is None):
                                s['subcategory'] = step_llm_fixes[i]
                                step_fix_count += 1
                                break
                        break

    print(f"  组级字段修复: {fix_count}")
    print(f"  步骤 subcategory 修复: {step_fix_count}")

    return groups


# ═══════════════════════════════════════════
# Main
# ═══════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description='步骤质量校验')
    parser.add_argument('--fix', action='store_true', help='自动修复并输出文件')
    parser.add_argument('--llm', action='store_true', help='LLM 深度校验')
    parser.add_argument('--all', action='store_true', help='全部 (规则+LLM+修复)')
    parser.add_argument('--dry-run', action='store_true', help='仅扫描，不修复')
    args = parser.parse_args()

    groups = load_json(os.path.join(DATA, 'image_groups.json'))
    doc_model_map = load_json(os.path.join(DATA, 'doc_model_map.json'))

    print(f"加载 {len(groups)} 个步骤组, {len(doc_model_map)} 条文档-型号映射")

    # V1-V3: 规则扫描
    issues_v1, fixes_v1 = v1_group_integrity(groups)
    null_steps = v2_step_subcategories(groups)
    no_tag_groups, missing_tag_groups, tag_suggestions = v3_model_association(groups, doc_model_map)

    # V4-V5: LLM
    fixes_v4 = {}
    step_llm_fixes = {}
    do_llm = args.llm or args.all
    if do_llm:
        fixes_v4 = v4_llm_group_metadata(groups, issues_v1)
        step_llm_fixes = v5_llm_step_subcategories(null_steps)

    # V6: 修复
    if (args.fix or args.all) and not args.dry_run:
        fixed_groups = v6_generate_fixes(
            groups, fixes_v1, fixes_v4,
            step_llm_fixes, null_steps,
            no_tag_groups, missing_tag_groups, tag_suggestions
        )
        out_path = os.path.join(DATA, 'image_groups_fixed.json')
        save_json(out_path, fixed_groups)
        print(f"\n  ✅ 修复文件: {out_path}")
        print(f"  下一步:")
        print(f"    1. diff data/image_groups.json data/image_groups_fixed.json  # 检查")
        print(f"    2. mv data/image_groups_fixed.json data/image_groups.json    # 确认")
        print(f"    3. python data/build_kb.py                                   # 重建 Qdrant")
        print(f"    4. systemctl restart chatbot_server                          # 重启")

    # 总结
    print("\n" + "=" * 60)
    print("验证报告")
    print("=" * 60)
    print(f"  V1 组级字段: {len(issues_v1)} 组有问题, {len(fixes_v1)} 组可规则修复")
    remaining = len(null_steps) - sum(1 for ns in null_steps if ns['classified'])
    print(f"  V2 步骤分类: {sum(1 for ns in null_steps if ns['classified'])}/{len(null_steps)} 规则覆盖, {remaining} 待 LLM/人工")
    print(f"  V3 模型关联: {len(no_tag_groups)} 组无 tags, {len(missing_tag_groups)} 组需补充")
    if do_llm:
        print(f"  V4 LLM 组级: {len(fixes_v4)} 组推断")
        print(f"  V5 LLM 步骤: {len(step_llm_fixes)} 个分类")


if __name__ == '__main__':
    main()
