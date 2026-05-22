#!/usr/bin/env python3
"""
Stage 2: 图片分类 (规则 + 上下文 + DeepSeek 文本推理)
输入:  data/image_metadata.json
输出:  data/image_classification.json

策略:
  - 437 张: 上下文含"扫描如下码"等 → 规则直出 config_code ✅
  - 72 张:  上下文含"如下图" → screenshot ✅
  - 16 张:  上下文含"接线"等 → diagram ✅
  - 321 张: 无明确上下文但 <20KB 或 >=50KB → 大小规则 ✅
  - 82 张:  20-50KB + 模糊上下文 → DeepSeek 文本推理
"""

import json, os, sys, requests, time

META_PATH = os.path.join(os.path.dirname(__file__), "../data/image_metadata.json")
CLASS_OUT = os.path.join(os.path.dirname(__file__), "../data/image_classification.json")
DEEPSEEK_API = "https://api.deepseek.com/chat/completions"

# ---------- 规则关键词 ----------
CTX_CONFIG = ['扫描如下码', '配置码', '功能码', '设置码', '接口码',
              '恢复出厂', '配对码', 'USB键盘口', 'USB虚拟串口',
              '添加CRLF', '进入设置', '退出设置', '开启设置',
              '执行恢复出厂', '设置连续模式', '开启EIO']
CTX_SCREENSHOT = ['如下图', '如下图所示', '如下截图', '界面如下', '串口工具如下']
CTX_DIAGRAM = ['连接示意', '连接方式', '接线图', '线缆连接']

FUNC_KW = {
    'restore_factory': ['恢复出厂', '重置', 'reset'],
    'test_code': ['测试码', '通信验证', '输出测试'],
    'setup': ['进入设置', '退出设置', '开启设置'],
    'suffix': ['后缀', '回车', '换行', 'CRLF', 'CR', 'LF', '逗号'],
    'prefix': ['前缀', '自定义前缀'],
    'interface': ['接口码', 'USB键盘口', 'USB虚拟串口', '串口模式', 'RS232', 'PS2'],
    'pairing': ['配对', '配对码', '蓝牙'],
    'feature': ['连续模式', '自动扫描', '延时', 'No Read', 'EIO',
                 'DPM', '截取', '序列扫描', '数据替换', 'DATREP',
                 'OCR', '码制', '优化', '省电'],
}


def load_deepseek_key():
    try:
        cfg = json.load(open('/home/admin/.openclaw/openclaw.json'))
        return cfg['models']['providers']['deepseek']['apiKey']
    except:
        return None


def classify_by_rules(img):
    ctx = (img.get('context_text') or '').strip()
    fmt = (img.get('format') or '').lower()
    size = img.get('file_size', 0)

    is_config = any(k in ctx for k in CTX_CONFIG)
    is_ss = any(k in ctx for k in CTX_SCREENSHOT)
    is_diag = any(k in ctx for k in CTX_DIAGRAM)

    # EMF → config_code
    if fmt in ('.emf', '.wmf'):
        sub = next((s for s, kws in FUNC_KW.items() if any(k in ctx for k in kws)), None)
        return 'config_code', sub, 'high', 'emf'

    # 上下文关键词判定
    if is_config and not is_ss and not is_diag:
        sub = next((s for s, kws in FUNC_KW.items() if any(k in ctx for k in kws)), None)
        return 'config_code', sub, 'high', 'ctx_config'

    if is_ss:
        return 'screenshot', None, 'high', 'ctx_screenshot'

    if is_diag:
        return 'diagram', None, 'high', 'ctx_diagram'

    # 大小规则
    if size < 20000:
        sub = next((s for s, kws in FUNC_KW.items() if any(k in ctx for k in kws)), None)
        return 'config_code', sub, 'high', 'size_small'
    elif size >= 50000:
        return 'screenshot', None, 'high', 'size_large'
    else:
        # 20-50KB + 模糊 → 需要 DeepSeek
        return 'config_code', None, 'low', 'need_ds'


def deepseek_classify(img):
    """用 DeepSeek 文本模型分析模糊图片"""
    key = load_deepseek_key()
    if not key:
        return 'config_code', None, 'low', 'no_ds_key'

    ctx = (img.get('context_text') or '').strip()[:300]
    doc = os.path.basename(img.get('source_doc', ''))
    size = img.get('file_size', 0)
    fmt = img.get('format', '')

    prompt = f"""你是一个工业扫描器文档的图片类型判断专家。

文档标题: {doc}
图片格式: {fmt}
图片大小: {size} 字节
图片附近文字: {ctx}

根据以上信息，判断这张图片的类型。只返回一行 JSON，不要多余文字：
{{"category":"config_code","subcategory":"suffix","confidence":0.95}}

category可选: config_code(配置条码), screenshot(软件截图), diagram(示意图/接线图), product_photo(产品图)
subcategory(config_code时): restore_factory, test_code, setup, suffix, prefix, interface, pairing, feature, unknown_config
confidence: 0.0-1.0"""

    try:
        resp = requests.post(DEEPSEEK_API, headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        }, json={
            "model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 200,
            "temperature": 0.1,
        }, timeout=15)

        if resp.status_code != 200:
            return 'config_code', None, 'low', 'ds_api_err'

        data = resp.json()
        rtext = data['choices'][0]['message'].get('content', '').strip()
        if not rtext:
            return 'config_code', None, 'low', 'ds_empty'

        # Parse JSON from response
        # Try to extract from markdown code block if present
        if '```json' in rtext:
            rtext = rtext.split('```json')[1].split('```')[0]
        elif '```' in rtext:
            rtext = rtext.split('```')[1].split('```')[0]

        parsed = json.loads(rtext.strip())
        cat = parsed.get('category', 'config_code')
        sub = parsed.get('subcategory')
        conf_val = parsed.get('confidence', 0.5)
        conf = 'high' if conf_val >= 0.8 else 'medium' if conf_val >= 0.5 else 'low'
        return cat, sub, conf, 'deepseek'
    except Exception as e:
        return 'config_code', None, 'low', f'ds_err:{str(e)[:30]}'


def main():
    if not os.path.exists(META_PATH):
        print(f"[错误] {META_PATH} 不存在")
        sys.exit(1)

    with open(META_PATH, 'r', encoding='utf-8') as f:
        images = json.load(f)

    print(f"加载 {len(images)} 张图片\n")

    classified = []
    ds_count = 0
    stats = {}

    for idx, img in enumerate(images):
        cat, sub, conf, method = classify_by_rules(img)

        # 只有模糊才调 DeepSeek
        if method == 'need_ds':
            ds_count += 1
            cat, sub, conf, method = deepseek_classify(img)

        # 检查是否需要调用 DeepSeek 补充子类型(配置码但无子类型)
        if cat == 'config_code' and not sub and img.get('context_text', '').strip():
            # 用关键词再查一遍
            ctx = img.get('context_text', '')
            for s, kws in FUNC_KW.items():
                if any(k in ctx for k in kws):
                    sub = s
                    break

        stats[method] = stats.get(method, 0) + 1

        if idx < 8 or method in ('need_ds', 'deepseek', 'ds_'):
            ctx_short = (img.get('context_text') or '')[:45]
            print(f"  [{idx+1:>3}] {img['file_name'][:40]:42s} → {cat:15s} {sub or '':20s} [{method}] ctx={ctx_short}")

        classified.append({
            "image_id": img["image_id"],
            "file_name": img["file_name"],
            "file_size": img["file_size"],
            "format": img["format"],
            "source_doc": img["source_doc"],
            "image_order": img["image_order"],
            "context_text": img.get("context_text", ""),
            "category": cat,
            "subcategory": sub,
            "confidence": conf,
            "method": method,
        })

    # 统计
    cats = {}
    subcats = {}
    for c in classified:
        cat = c['category'] or 'unclassified'
        cats[cat] = cats.get(cat, 0) + 1
        if c['subcategory']:
            subcats[c['subcategory']] = subcats.get(c['subcategory'], 0) + 1

    print(f"\n===== 分类结果 =====")
    for cat, n in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {cat:20s}: {n:>4}")
    print(f"\n  子类型分布: {subcats}")

    config_count = sum(1 for c in classified if c.get('category') == 'config_code')
    with_sub = sum(1 for c in classified if c.get('category') == 'config_code' and c.get('subcategory'))
    print(f"\n配置码: {config_count} (含子类型: {with_sub})")
    print(f"识别方法: {stats}")
    print(f"DeepSeek 调用: {ds_count} 次")

    os.makedirs(os.path.dirname(CLASS_OUT), exist_ok=True)
    with open(CLASS_OUT, 'w', encoding='utf-8') as f:
        json.dump(classified, f, ensure_ascii=False, indent=2)
    print(f"\n输出: {CLASS_OUT}")


if __name__ == '__main__':
    main()
