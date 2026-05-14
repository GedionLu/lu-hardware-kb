#!/usr/bin/env python3
"""评测引擎: 运行评测集 → 报告准确率"""
import csv, json, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from query import QueryEngineV2

EVAL_PATH = os.path.join(os.path.dirname(__file__), "data/eval_set.csv")
RESULT_PATH = os.path.join(os.path.dirname(__file__), "data/eval_result.json")

def main():
    engine = QueryEngineV2()
    
    with open(EVAL_PATH, 'r', encoding='utf-8') as f:
        cases = list(csv.DictReader(f))
    
    print(f"加载评测集: {len(cases)} 条\n")
    
    results = []
    metrics = {"model_correct": 0, "intent_correct": 0, "doc_correct": 0, "total": len(cases)}
    
    for i, case in enumerate(cases):
        query = case['query']
        exp_model = case['model']
        exp_intent = case['intent']
        exp_doc_kw = case['expected_doc_kw']
        
        result = engine.query(query)
        
        # Check model
        models = result.get('models', [])
        model_ok = exp_model in models
        
        # Check intent
        intents = result.get('intents', [])
        intent_ok = exp_intent in intents
        
        # Check doc (only if doc keyword is specified)
        doc_ok = True
        if exp_doc_kw:
            resp = result.get('response', '')
            doc_ok = exp_doc_kw in resp
        
        if model_ok: metrics['model_correct'] += 1
        if intent_ok: metrics['intent_correct'] += 1
        if doc_ok: metrics['doc_correct'] += 1
        
        status = "✅" if (model_ok and intent_ok and doc_ok) else "❌"
        
        if status == "❌":
            issues = []
            if not model_ok: issues.append(f"型号:期望{exp_model}->实际{models}")
            if not intent_ok: issues.append(f"意图:期望{exp_intent}->实际{intents}")
            if not doc_ok: issues.append(f"文档:期望含'{exp_doc_kw}'")
            print(f"  {status} [{i+1}/{len(cases)}] {query}")
            print(f"     {' | '.join(issues)}")
        
        results.append({
            "query": query,
            "expected_model": exp_model,
            "expected_intent": exp_intent,
            "expected_doc_kw": exp_doc_kw,
            "got_models": models,
            "got_intents": intents,
            "model_ok": model_ok,
            "intent_ok": intent_ok,
            "doc_ok": doc_ok,
        })
    
    # Summary
    print("\n" + "=" * 50)
    print("评测结果")
    print("=" * 50)
    print(f"  总计: {metrics['total']} 条")
    print(f"  型号识别: {metrics['model_correct']}/{metrics['total']} = {metrics['model_correct']/metrics['total']*100:.1f}%")
    print(f"  意图识别: {metrics['intent_correct']}/{metrics['total']} = {metrics['intent_correct']/metrics['total']*100:.1f}%")
    print(f"  文档匹配: {metrics['doc_correct']}/{metrics['total']} = {metrics['doc_correct']/metrics['total']*100:.1f}%")
    total_ok = sum(1 for r in results if r['model_ok'] and r['intent_ok'] and r['doc_ok'])
    print(f"  全对: {total_ok}/{metrics['total']} = {total_ok/metrics['total']*100:.1f}%")
    
    # Save
    with open(RESULT_PATH, 'w', encoding='utf-8') as f:
        json.dump({"metrics": metrics, "results": results}, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果: {RESULT_PATH}")


if __name__ == '__main__':
    main()
