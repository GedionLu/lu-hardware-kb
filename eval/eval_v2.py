#!/usr/bin/env python3
"""
增强评测引擎 v2
- 支持多维度评测（型号 / 意图 / 文档 / LLM语义评分）
- 支持分类统计（按产品类别、查询类型）
- 输出详细报告 + JSON
"""

import csv, json, sys, os, re, requests, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from query import QueryEngineV3 as QueryEngineV2
from query import load_intents
INTENT_MAP = load_intents()

EVAL_PATH = os.path.join(os.path.dirname(__file__), "data/eval_set_v2.csv")
RESULT_PATH = os.path.join(os.path.dirname(__file__), "data/eval_result_v2.json")
DEEPSEEK_API = "https://api.deepseek.com/chat/completions"

def load_ds_key():
    try:
        cfg = json.load(open('/home/admin/.openclaw/openclaw.json'))
        return cfg['models']['providers']['deepseek']['apiKey']
    except:
        return None

def llm_judge(query, expected_intent, response_text, models):
    """用 LLM 评估回答质量 (1-5分)"""
    key = load_ds_key()
    if not key:
        return None, "no_key"
    
    prompt = f"""你是工业扫描器技术支持系统的评测专家。评估以下问答质量。

用户问题: {query}
期望意图: {expected_intent}
识别型号: {models}
系统回答:
{response_text[:800]}

评分标准:
5 - 完美: 直接命中问题，配置码/步骤正确
4 - 良好: 相关但可能不够精确
3 - 部分相关: 沾边但不完全对口
2 - 弱相关: 只沾一点边
1 - 不相关: 完全答非所问

只返回: {{"score": 数字, "reason": "一句话理由"}}"""

    try:
        resp = requests.post(DEEPSEEK_API, headers={
            "Authorization": f"Bearer {key}", "Content-Type": "application/json"
        }, json={
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 100, "temperature": 0.1,
        }, timeout=15)
        if resp.status_code == 200:
            content = resp.json()['choices'][0]['message'].get('content', '')
            # parse JSON
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0]
            elif '```' in content:
                content = content.split('```')[1].split('```')[0]
            parsed = json.loads(content.strip())
            return parsed.get('score'), parsed.get('reason', '')
    except:
        pass
    return None, "api_error"


def main():
    engine = QueryEngineV2()
    
    with open(EVAL_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        cases = [row for row in reader if not row['query'].startswith('#')]
    
    print(f"📊 增强评测集: {len(cases)} 条\n")
    
    results = []
    metrics = {
        "model_correct": 0, "intent_correct": 0, "doc_correct": 0,
        "total": 0, "total_with_model": 0, "total_with_doc_kw": 0,
        "llm_scores": [],
    }
    
    # 分类统计
    cat_metrics = {}
    
    for i, case in enumerate(cases):
        query = case['query'].strip()
        if not query:
            continue
        
        exp_model = case.get('model', '').strip()
        exp_intent = case.get('expected_intent', '').strip()
        exp_doc_kw_raw = case.get('expected_doc_kw', '').strip()
        exp_kws = [kw.strip() for kw in exp_doc_kw_raw.split(',') if kw.strip()] if exp_doc_kw_raw else []
        cat = case.get('category', '其他').strip()
        note = ''
        
        metrics['total'] += 1
        
        try:
            result = engine.query(query)
        except Exception as e:
            print(f"  ❌ [{i+1}] 查询异常: {query} → {e}")
            results.append({"query": query, "error": str(e)})
            continue
        
        models = result.get('models', [])
        intents = result.get('intents', [])
        resp = result.get('response', '')
        
        # 评估
        model_ok = (not exp_model) or (exp_model in models)  # 无期望型号则跳过
        intent_ok = (not exp_intent) or (exp_intent in intents)
        
        doc_ok = True
        if exp_kws:
            metrics['total_with_doc_kw'] += 1
            doc_ok = any(kw in resp for kw in exp_kws)
        
        if exp_model:
            metrics['total_with_model'] += 1
            if model_ok: metrics['model_correct'] += 1
        if exp_intent and intent_ok:
            metrics['intent_correct'] += 1
        if exp_kws and doc_ok:
            metrics['doc_correct'] += 1
        
        # LLM 语义评分
        llm_score = None
        if exp_intent:
            llm_score, reason = llm_judge(query, exp_intent, resp, models)
            if llm_score:
                metrics['llm_scores'].append(llm_score)
        
        # 分类统计
        if cat not in cat_metrics:
            cat_metrics[cat] = {"total": 0, "model_ok": 0, "intent_ok": 0, "doc_ok": 0, "all_ok": 0}
        cat_metrics[cat]["total"] += 1
        if model_ok: cat_metrics[cat]["model_ok"] += 1
        if intent_ok: cat_metrics[cat]["intent_ok"] += 1
        if doc_ok: cat_metrics[cat]["doc_ok"] += 1
        
        all_ok = (not exp_model or model_ok) and (not exp_intent or intent_ok) and (not exp_kws or doc_ok)
        if all_ok: cat_metrics[cat]["all_ok"] += 1
        
        status = "✅" if all_ok else "❌"
        
        # 打印每行结果
        issues = []
        if exp_model and not model_ok: issues.append(f"型号:{exp_model}→{models}")
        if exp_intent and not intent_ok: issues.append(f"意图:{exp_intent}→{intents}")
        if exp_kws and not doc_ok: issues.append(f"文档缺{exp_kws}")
        llm_str = f" [LLM:{llm_score}]" if llm_score else ""
        
        print(f"  {status} [{i+1:>2}/{len(cases)}] {query[:45]:45s} | 型号={'✓' if model_ok else '✗'} 意图={'✓' if intent_ok else '✗'} 文档={'✓' if doc_ok else '✗'}{llm_str} | {note}")
        if issues:
            print(f"       {' | '.join(issues)}")
        
        results.append({
            "query": query,
            "expected_model": exp_model,
            "expected_intent": exp_intent,
            "expected_doc_kw": exp_doc_kw_raw,
            "category": cat,
            "note": note,
            "got_models": models,
            "got_intents": intents,
            "response": resp[:300],
            "model_ok": model_ok,
            "intent_ok": intent_ok,
            "doc_ok": doc_ok,
            "llm_score": llm_score,
        })
    
    # ── 汇总报告 ──
    print("\n" + "=" * 60)
    print("📊 评测汇总")
    print("=" * 60)
    
    n = metrics['total']
    n_model = metrics['total_with_model']
    n_doc = metrics['total_with_doc_kw'] or 1
    
    print(f"  总用例: {n}")
    print(f"  型号识别: {metrics['model_correct']}/{n_model} = {metrics['model_correct']/max(n_model,1)*100:.1f}%")
    print(f"  意图识别: {metrics['intent_correct']}/{n} = {metrics['intent_correct']/n*100:.1f}%")
    print(f"  文档匹配: {metrics['doc_correct']}/{n_doc} = {metrics['doc_correct']/max(n_doc,1)*100:.1f}%")
    
    all_ok_count = sum(1 for r in results if r.get('model_ok', True) and r.get('intent_ok', True) and r.get('doc_ok', True))
    print(f"  全对率: {all_ok_count}/{n} = {all_ok_count/n*100:.1f}%")
    
    if metrics['llm_scores']:
        avg_llm = sum(metrics['llm_scores']) / len(metrics['llm_scores'])
        print(f"  LLM语义平均分: {avg_llm:.2f}/5.0 ({len(metrics['llm_scores'])}条)")
    
    # 分类统计
    print(f"\n  分类统计:")
    for cat, cm in sorted(cat_metrics.items()):
        t = cm['total']
        rate = cm['all_ok']/t*100 if t else 0
        print(f"    {cat:12s}: {cm['all_ok']}/{t} = {rate:.0f}%")
    
    # 保存
    with open(RESULT_PATH, 'w', encoding='utf-8') as f:
        json.dump({"metrics": {k: (v if k != 'llm_scores' else len(v)) for k, v in metrics.items()},
                   "llm_avg_score": avg_llm if metrics['llm_scores'] else None,
                   "results": results}, f, ensure_ascii=False, indent=2)
    print(f"\n📄 详细结果: {RESULT_PATH}")


if __name__ == '__main__':
    main()
