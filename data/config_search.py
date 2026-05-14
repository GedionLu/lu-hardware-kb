#!/usr/bin/env python3
"""
知识库搜索工具
用法: python3 config_search.py "<查询>" [top_k]
返回: { text: [...], images: {...} }
"""
import json, os, re, math, sys

DATA_DIR = "/home/admin/.openclaw/workspace/lu/data"

with open(os.path.join(DATA_DIR, "config_codes.json")) as f:
    CONFIG_CODES = json.load(f)
with open(os.path.join(DATA_DIR, "step_groups.json")) as f:
    STEP_GROUPS = json.load(f)

def search(query: str, top_k: int = 10):
    q = query.lower().strip()
    if not q:
        return []
    
    results = []
    
    # Search config codes
    for item in CONFIG_CODES:
        score = _score_item(q, item['keywords'].lower(), item['code_name'].lower(),
                          item['description'].lower(), item['product_name'].lower(), item['model'].lower())
        if score > 0:
            results.append({
                "type": "config_code",
                "score": round(1.0 / (1.0 + math.exp(-score / 5)), 4),
                "content": f"配置码: {item['code_name']}\n说明: {item['description']}",
                "product_name": item['product_name'],
                "model": item['model'],
                "image_url": item.get('image_url', ''),
                "source_file": item['source_file']
            })
    
    # Search step groups (complete sets)
    for group in STEP_GROUPS:
        score = _score_item(q, group['keywords'].lower(), group['titles'].lower(),
                          '', group['product_name'].lower(), group['model'].lower())
        if score > 0:
            steps_text = []
            all_images = []
            for step in group['steps']:
                st = f"步骤 {step['step_number']}: {step['title']}"
                if step['description']:
                    st += f"\n{step['description']}"
                steps_text.append(st)
                all_images.extend(step.get('image_urls', []))
            
            results.append({
                "type": "step_group",
                "score": round(1.0 / (1.0 + math.exp(-score / 5)), 4),
                "content": "\n---\n".join(steps_text),
                "step_count": group['step_count'],
                "image_urls": all_images,
                "product_name": group['product_name'],
                "model": group['model'],
                "source_file": group['source_file']
            })
    
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:top_k]

def _score_item(q: str, kw: str, name: str, desc: str, product: str, model: str) -> int:
    score = 0
    # Full query exact match
    if q in kw: score += 10
    if q in name: score += 8
    if q in product or q in model: score += 6
    
    tokens = q.split()
    for t in tokens:
        if t in name: score += 5
        if t in kw: score += 3
        if t in desc: score += 2
        if t in product or t in model: score += 5
        # Character-level matching for Chinese
        chars = re.findall(r'[\u4e00-\u9fff]', t)
        for c in chars:
            if c in name: score += 2
            elif c in kw: score += 1
    
    return score

def get_images(results: list) -> dict:
    images = {}
    for r in results:
        if r['type'] == 'config_code' and r.get('image_url'):
            name = r['content'].split('\n')[0].replace('配置码: ', '')
            images[name] = r['image_url']
        elif r['type'] == 'step_group' and r.get('image_urls'):
            key = r['source_file'][:50]
            images[key] = r['image_urls']
    return images

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('{"error": "Usage: config_search.py <query> [top_k]"}')
        sys.exit(1)
    query = sys.argv[1]
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    results = search(query, top_k)
    images = get_images(results)
    output = {"text": results, "images": images}
    print(json.dumps(output, ensure_ascii=False))
