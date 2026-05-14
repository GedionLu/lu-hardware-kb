#!/usr/bin/env python3
"""LLM 辅助补齐产品树"""
import os, re, csv, json, requests

KB = "/tmp/KnowledgeBase"

# 1. Extract all filenames
all_names = set()
for root, dirs, files in os.walk(KB):
    for f in files:
        name = os.path.splitext(f)[0]
        all_names.add(name)

# 2. Pattern matching
model_patterns = {
    r'1202g': '1202g', r'1990i': '1990i', r'1981i': '1981i', r'1991i': '1991i',
    r'1911i': '1911i', r'1950': '1950', r'1952': '1952',
    r'1900-C': '1900-C', r'1900': '1900', r'1902-C': '1902-C', r'1902': '1902',
    r'1472': '1472', r'1470': '1470',
    r'OH430': 'OH430', r'OH450[23]': 'OH4502', r'OH450x': 'OH450x',
    r'OH460': 'OH460', r'OH462': 'OH462', r'OH420': 'OH420', r'OH350': 'OH350X',
    r'HH490': 'HH490', r'HH492': 'HH492', r'HH760': 'HH760', r'HH762': 'HH762',
    r'HH4X0': 'HH4X0',
    r'HF680': 'HF680', r'HF600': 'HF600',
    r'SC2800': 'SC2800',
    r'3320g': '3320g', r'33x0g': '33x0g',
    r'7120-2D': '7120-2D', r'7120PLUS': '7120PLUS', r'MS7120': 'MS7120',
    r'7580g': '7580g', r'7680g': '7680g',
    r'PM42': 'PM42', r'PM43': 'PM43', r'PM45': 'PM45',
    r'PX240': 'PX240', r'PX940': 'PX940', r'PC300T': 'PC300T',
    r'8680i': '8680i', r'19x2': '19x2',
}

found = set()
for name in all_names:
    for pattern, model in model_patterns.items():
        if re.search(pattern, name):
            found.add(model)

# 3. Load existing
existing = set()
with open('data/product_tree.csv') as f:
    for row in csv.DictReader(f):
        m = row.get('型号','').strip()
        if m: existing.add(m)

new_models = sorted(found - existing)
print(f"知识库检出: {len(found)} 型号")
print(f"产品树已有: {len(existing)} 型号")
print(f"缺失 {len(new_models)} 个: {', '.join(new_models)}")

if not new_models:
    print("产品树已完整，无需补充")
    exit(0)

# 4. LLM categorization
cfg = json.load(open('/home/admin/.openclaw/openclaw.json'))
key = cfg['models']['providers']['deepseek']['apiKey']

prompt = (
    "你是工业扫描器产品分类专家。请将以下新型号归类。\n\n"
    "现有产品树结构:\n"
    "大类: 手持扫描枪 / 固定式扫码器 / 平台式扫描器 / 工业打印机 / 可穿戴/特种\n"
    "系列举例: 19xx系列, 14xx系列, HH系列, OH430系列, OH450x系列, HF系列, "
    "33x0g系列, MS系列(老款), Genesis系列, 7120系列, Fiji平台, 工业级, 穿戴式\n\n"
    "新型号: " + ", ".join(new_models) + "\n\n"
    "对每个型号判断 大类 和 系列。输出CSV格式: 型号,大类,系列\n"
    "不确定的系列填\"其他\"。"
)

resp = requests.post("https://api.deepseek.com/chat/completions",
    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    json={"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}],
          "max_tokens": 300, "temperature": 0.1}, timeout=15)

result = resp.json()['choices'][0]['message'].get('content', '')
print(f"\nLLM 建议:\n{result}")

# 5. Parse LLM output and generate CSV rows
new_rows = []
for line in result.strip().split('\n'):
    line = line.strip()
    if ',' not in line or line.startswith('型号') or line.startswith('---'):
        continue
    parts = [p.strip() for p in line.split(',')]
    if len(parts) >= 3:
        model, cat, series = parts[0], parts[1], parts[2]
        if model in new_models:
            new_rows.append({"大类": cat, "系列": series, "子系列(可选)": "", "型号": model, "子型号/SN变体": ""})

if new_rows:
    print(f"\n新增 {len(new_rows)} 行到产品树:")
    for r in new_rows:
        print(f"  {r['大类']:12s} | {r['系列']:20s} | {r['型号']}")

    # Append to CSV
    with open('data/product_tree.csv', 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=["大类","系列","子系列(可选)","型号","子型号/SN变体"])
        w.writerows(new_rows)
    print(f"\n已追加到 data/product_tree.csv")
else:
    print("LLM 输出无法解析，请手动检查")
    print(f"原始输出: {result}")
