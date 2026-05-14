#!/usr/bin/env python3
"""
Stage 2: 图片分类 (规则 + 上下文关键词 + DeepSeek 文本推理)
输入:  data/image_metadata.json
输出:  data/image_classification.json

方法:
  1. EMF/WMF → 强制 config_code (矢量条码)
  2. 上下文含"扫描如下码"等 → config_code + 关键词匹配子类型
  3. 上下文含"如下图"等 → screenshot/diagram
  4. 20-50KB 灰色地带 + 模糊上下文 → 调用 DeepSeek 文本分析
  5. 其余按文件大小规则
"""

import json, os, re, sys, base64, requests, time
from xml.etree import ElementTree

META_PATH = os.path.join(os.path.dirname(__file__), "../data/image_metadata.json")
CLASS_OUT = os.path.join(os.path.dirname(__file__), "../data/image_classification.json")

DEEPSEEK_API = "https://api.deepseek.com/chat/completions"
DEEPSEEK_KEY = None  # loaded from config

# ---------- 关键词规则 ----------
CTX_CONFIG = ['扫描如下码', '扫描如下', '扫描如下接口码', '扫描如下两个码',
              '配置码', '功能码', '设置码', '接口码',
              '执行恢复出厂', '配对码', '配置条码',
              '进入设置', '退出设置', '开启设置',
              'USB键盘口', 'USB虚拟串口', '串口模式',
              '添加CRLF', '删除所有后缀', '添加回车',
              '恢复出厂', '设置连续模式', '开启EIO']

CTX_SCREENSHOT = ['如下图', '如下图所示', '如图所示', '如下截图',
                  '界面如下', '打开如下软件', '串口工具如下',
                  '设备管理器', '下图所示']

CTX_DIAGRAM = ['连接示意', '如上图', '连接方式', '接线图', '线缆连接']

FUNC_KW = {
    'restore_factory': ['恢复出厂', '重置', 'reset', '恢复默认'],
    'test_code': ['测试码', '测试', '通信验证', '输出测试'],
    'setup': ['进入设置', '退出设置', '开启设置'],
    'suffix': ['后缀', '回车', '换行', 'CRLF', 'CR', 'LF', '逗号'],
    'prefix': ['前缀', '自定义前缀'],
    'interface': ['接口码', 'USB键盘口', 'USB虚拟串口', '串口模式', 'RS232', 'PS2'],
    'pairing': ['配对', '配对码', '连接蓝牙'],
    'feature': ['连续模式', '自动扫描', '延时', 'No Read', 'EIO',
                 'DPM', '截取', '序列扫描', '数据替换', 'DATREP',
                 'OCR', '码制', '编码', '优化', '省电'],
}

# ---------- 加载 DeepSeek Key ----------
def load_key():
    global DEEPSEEK_KEY
    if DEEPSEEK_KEY:
        return DEEPSEEK_KEY
    try:
        cfg = json.load(open('/home/admin/.openclaw/openclaw.json'))
        DEEPSEEK_KEY = cfg['models']['providers']['deepseek']['apiKey']
    except:
        DEEPSEEK_KEY = None
    return DEEPSEEK_KEY


def call_deepseek_text(prompt, max_tokens=200):
    """调用 DeepSeek 文本模型分析"""
    key = load_key()
    if not key:
        return None
    try:
        resp = requests.post(DEEPSEEK_API, headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        }, json={
            "model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.1,
        }, timeout=15)
        if resp.status_code == 200:
            return resp.json()['choices'][0]['message']['content']
        else:
            print(f"  [DeepSeek API 错误] {resp.status_code}: {resp.text[:200]}")
            return None
    except Exception as e:
        print(f"  [DeepSeek 调用失败] {e}")
        return None


# ---------- 规则分类 ----------
def classify_by_rules(img):
    ctx = (img.get('context_text') or '').strip()
    fmt = (img.get('format') or '').lower()
    size = img.get('file_size', 0)
    
    result = {
        'category': None,
        'subcategory': None,
        'confidence': 'low',
        'method': 'rule',
    }

    # EMF/WMF → 配置码
    if fmt in ('.emf', '.wmf'):
        result['category'] = 'config_code'
        result['confidence'] = 'high'
        result['method'] = 'emf_format'
        
        for subcat, kws in FUNC_KW.items():
            if any(kw in ctx for kw in kws):
                result['subcategory'] = subcat
                break
        return result

    # 上下文含明确关键词
    is_config = any(kw in ctx for kw in CTX_CONFIG)
    is_ss = any(kw in ctx for kw in CTX_SCREENSHOT)
    is_diag = any(kw in ctx for kw in CTX_DIAGRAM)

    if is_config and not is_ss and not is_diag:
        result['category'] = 'config_code'
        result['confidence'] = 'high'
        result['method'] = 'ctx_config_kw'
        for subcat, kws in FUNC_KW.items():
            if any(kw in ctx for kw in kws):
                result['subcategory'] = subcat
                result['confidence'] = 'high'
                break
        return result

    if is_ss and not is_config:
        result['category'] = 'screenshot'
        result['confidence'] = 'high'
        result['method'] = 'ctx_screenshot_kw'
        return result

    if is_diag and not is_config:
        result['category'] = 'diagram'
        result['confidence'] = 'high'
        result['method'] = 'ctx_diagram_kw'
        return result

    # 大小规则 (没有明确上下文)
    if size < 20000:
        result['category'] = 'config_code'
        result['confidence'] = 'high'
        result['method'] = 'size_small'
    elif size >= 50000:
        result['category'] = 'screenshot'
        result['confidence'] = 'high'
        result['method'] = 'size_large'
    else:
        # 20-50KB 灰色地带，需要 DeepSeek 推理
        result['category'] = 'config_code'
        result['confidence'] = 'low'
        result['method'] = 'size_ambiguous'
        # 但不标记为 low，因为可能有上下文帮助
        if ctx:
            result['method'] = 'size_ambiguous_with_ctx'

    return result


# ---------- DeepSeek 二次判决 ----------
def deepseek_refine(img, rule_result):
    """对模糊图片用 DeepSeek 文本推理"""
    ctx = (img.get('context_text') or '').strip()
    doc_name = os.path.basename(img.get('source_doc', ''))
    fmt = img.get('format', '')
    size = img.get('file_size', 0)

    prompt = f"""你是一个工业扫描器文档的图片分类专家。基于以下信息判断图片类型。

文档名: {doc_name}
图片格式: {fmt}
图片大小: {size} bytes
图片前后文字描述: {ctx[:200]}

请判断这张图片的类型，只返回 JSON:
{{
  "category": "config_code 或 screenshot 或 diagram 或 product_photo",
  "subcategory": "如果 config_code 则从以下选: restore_factory | test_code | setup | suffix | prefix | interface | pairing | feature | unknown_config; 否则填 null",
  "function": "简短一句话描述图片作用",
  "confidence": 0.0-1.0,
  "reason": "判断理由"
}}"""

    resp = call_deepseek_text(prompt)
    if not resp:
        return rule_result  # fallback

    try:
        # 提取 JSON
        j = resp.strip()
        if '```json' in j:
            j = j.split('```json')[1].split('```')[0]
        elif '```' in j:
            j = j.split('```')[1].split('```')[0]
        parsed = json.loads(j.strip())
        
        result = {
            'category': parsed.get('category', rule_result['category']),
            'subcategory': parsed.get('subcategory', rule_result.get('subcategory')),
            'confidence': 'medium' if parsed.get('confidence', 0) >= 0.7 else 'low',
            'method': 'deepseek',
        }
        return result
    except:
        return rule_result


# ---------- 主流程 ----------
def main():
    if not os.path.exists(META_PATH):
        print(f"[错误] {META_PATH} 不存在")
        sys.exit(1)

    with open(META_PATH, 'r', encoding='utf-8') as f:
        images = json.load(f)

    print(f"加载 {len(images)} 张图片\n")

    classified = []
    ds_count = 0  # DeepSeek 调用计数

    for i, img in enumerate(images):
        rule_result = classify_by_rules(img)

        # 模糊图片 + 有上下文 → 调 DeepSeek
        need_ds = (
            rule_result['method'] in ('size_ambiguous', 'size_ambiguous_with_ctx') 
            and bool(img.get('context_text', '').strip())
        )
        
        if need_ds and load_key():
            result = deepseek_refine(img, rule_result)
            ds_count += 1
        else:
            result = rule_result

        if i < 5 or need_ds:
            brief = img['file_name'][:40]
            ctx_short = (img.get('context_text') or '')[:50]
            print(f"  [{i+1:>3}] {brief:45s} ctx={ctx_short:30s} → {result['category']:15s} {result.get('subcategory') or '':20s} [{result['method']}]")

        classified.append({
            "image_id": img["image_id"],
            "file_name": img["file_name"],
            "file_size": img["file_size"],
            "format": img["format"],
            "source_doc": img["source_doc"],
            "image_order": img["image_order"],
            "context_text": img.get("context_text", ""),
            "category": result["category"],
            "subcategory": result["subcategory"],
            "confidence": result["confidence"],
            "method": result["method"],
        })

    # 统计
    cats = {}
    for c in classified:
        cat = c['category'] or 'unclassified'
        cats[cat] = cats.get(cat, 0) + 1

    print(f"\n===== 统计 =====")
    for cat, n in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {cat:20s}: {n:>4} 张")

    config_count = sum(1 for c in classified if c.get('category') == 'config_code')
    with_sub = sum(1 for c in classified if c.get('category') == 'config_code' and c.get('subcategory'))
    print(f"\n配置码: {config_count} (含子类型: {with_sub})")
    print(f"DeepSeek 调用: {ds_count} 次")

    with open(CLASS_OUT, 'w', encoding='utf-8') as f:
        json.dump(classified, f, ensure_ascii=False, indent=2)
    print(f"\n输出: {CLASS_OUT}")


if __name__ == '__main__':
    main()
