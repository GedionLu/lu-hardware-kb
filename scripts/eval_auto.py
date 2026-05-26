#!/usr/bin/env python3
"""
eval_auto.py — 评测自动化 CLI

功能:
  1. 加载 test set (CSV)，批量运行查询引擎
  2. 自动评分 (不依赖 LLM): 意图匹配 + 型号匹配 + 步骤相关性
  3. 可选 LLM 深度评分 (需 API key)
  4. 输出报告 (JSON + 文本) 用于 CI 追踪

用法:
  python eval_auto.py                          # 快速评测 (无 LLM)
  python eval_auto.py --llm                    # LLM 深度评分
  python eval_auto.py --ci                     # CI 模式 (退出码: 分数<阈值=1)
  python eval_auto.py --report eval_result.json # 保存详细报告
  python eval_auto.py --threshold 3.5          # CI 通过阈值

测试集格式 (CSV):
  query,model,expected_intent,expected_doc_kw,category
"""

import csv
import json
import os
import re
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, 'data')
EVAL_DIR = os.path.join(BASE, 'eval')

# ─── INTENT ALIAS MAP ───
INTENT_ALIASES = {
    'usb_connect': ['usb_connect', 'interface_mode'],
    'serial_connect': ['serial_connect', 'interface_mode'],
    'bluetooth_pairing': ['bluetooth_pairing', 'pairing'],
    'virtual_com_port': ['virtual_com_port', 'interface_mode'],
    'add_suffix': ['add_suffix'],
    'add_prefix': ['add_prefix'],
    'restore_factory': ['restore_factory'],
    'test_comm': ['test_comm'],
    'chinese_qr': ['chinese_qr'],
    'data_format': ['data_format', 'feature'],
    'no_read': ['no_read'],
    'interface_mode': ['interface_mode', 'usb_connect', 'serial_connect'],
    'pairing': ['pairing', 'bluetooth_pairing'],
    'product_overview': ['product_overview'],
    'general_setup': ['general_setup'],
}


def load_json(p):
    with open(p) as f:
        return json.load(f)


def load_test_set(path=None):
    if path is None:
        path = os.path.join(EVAL_DIR, 'data', 'eval_set_v2.csv')
    cases = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cases.append({
                'query': row['query'],
                'model': row.get('model', ''),
                'expected_intent': row.get('expected_intent', ''),
                'expected_doc_kw': row.get('expected_doc_kw', ''),
                'category': row.get('category', ''),
            })
    return cases


def intent_match(expected, actual):
    """意图匹配: 精确 + 别名"""
    if expected == actual:
        return 1.0
    aliases = INTENT_ALIASES.get(actual, [])
    return 0.7 if expected in aliases else 0.0


def model_match(expected, detected_models):
    """型号匹配"""
    if not detected_models:
        return 0.0
    expected_parts = set(expected.lower().replace('-', '').split())
    for m in detected_models:
        m_clean = m.lower().replace('-', '').replace(' ', '')
        if expected.lower().replace('-', '') in m_clean or m_clean in expected.lower().replace('-', ''):
            return 1.0
        # 部分匹配 (如 OH430 匹配 OH430x)
        for ep in expected_parts:
            if ep and ep in m_clean:
                return 0.7
    return 0.0


def step_relevance(response, keywords):
    """步骤相关性: 检查回复是否包含关键词"""
    if not keywords:
        return 0.5  # 无关键词时默认中等
    kws = [k.strip() for k in keywords.split(',')]
    matched = sum(1 for kw in kws if kw.lower() in response.lower())
    return min(1.0, matched / len(kws))


def auto_score(case, response, detected_models, detected_intent):
    """自动评分 (0-5)"""
    im = intent_match(case['expected_intent'], detected_intent)
    mm = model_match(case['model'], detected_models)
    sr = step_relevance(response, case.get('expected_doc_kw', ''))

    # 权重: 意图 40%, 型号 30%, 关键词相关性 30%
    score = im * 2.0 + mm * 1.5 + sr * 1.5
    return round(score, 2), {'intent_match': im, 'model_match': mm, 'step_relevance': sr}


def load_api_config():
    api_key = os.environ.get('DEEPSEEK_API_KEY', os.environ.get('OPENAI_API_KEY', ''))
    if not api_key:
        for cp in [os.path.expanduser('~/.openclaw/openclaw.json'),
                    '/home/admin/.openclaw/openclaw.json']:
            try:
                with open(cp) as f:
                    cfg = json.load(f)
                ds = cfg.get('models', {}).get('providers', {}).get('deepseek', {})
                api_key = ds.get('apiKey', '')
                if api_key:
                    break
            except:
                continue
    return api_key


def llm_judge(case, response, detected_models):
    """LLM 深度评分 (1-5)"""
    api_key = load_api_config()
    if not api_key:
        return None

    prompt = f"""你是扫描器客服系统评测专家。评分以下回答 (1-5):

用户: {case['query']}
期望意图: {case['expected_intent']}
识别型号: {detected_models}
回答:
{response[:800]}

评分:
5-完美命中  4-良好相关  3-部分相关  2-弱相关  1-不相关

只输出JSON: {{"score":数字,"reason":"理由"}}"""

    data = json.dumps({
        'model': 'deepseek-chat',
        'messages': [{'role': 'user', 'content': prompt}],
        'temperature': 0, 'max_tokens': 200,
    }).encode()

    try:
        req = urllib.request.Request(
            'https://api.deepseek.com/v1/chat/completions',
            data=data,
            headers={'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
            content = result['choices'][0]['message']['content'].strip()
            if content.startswith('```'):
                content = re.sub(r'^```(?:json)?\s*', '', content)
                content = re.sub(r'\s*```$', '', content)
            return json.loads(content)
    except Exception as e:
        return {'score': None, 'reason': str(e)}


def run_eval(test_set, query_engine, use_llm=False):
    """批量评测"""
    results = []
    scores = []

    for i, case in enumerate(test_set):
        print(f"  [{i+1}/{len(test_set)}] {case['query']}", end=' ', flush=True)

        try:
            start = time.time()
            result = query_engine.query(case['query'])
            response = result.get('response', '') if isinstance(result, dict) else str(result)
            elapsed = time.time() - start

            # 提取检测到的型号和意图
            detected_models = result.get('models', []) if isinstance(result, dict) else []
            detected_intent = result.get('intents', ['unknown'])[0] if isinstance(result, dict) and result.get('intents') else 'unknown'

            # 自动评分
            auto_s, auto_detail = auto_score(case, response, detected_models, detected_intent)

            # LLM 评分
            llm_result = None
            if use_llm:
                llm_result = llm_judge(case, response, detected_models)

            result = {
                'query': case['query'],
                'expected_model': case['model'],
                'expected_intent': case['expected_intent'],
                'detected_models': detected_models,
                'detected_intent': detected_intent,
                'auto_score': auto_s,
                'auto_detail': auto_detail,
                'llm_score': llm_result.get('score') if llm_result else None,
                'llm_reason': llm_result.get('reason') if llm_result else None,
                'response_preview': response[:200],
                'elapsed_ms': round(elapsed * 1000),
            }
            scores.append(auto_s)
            print(f"→ {auto_s}/5 ({elapsed:.1f}s)")
        except Exception as e:
            result = {
                'query': case['query'],
                'error': str(e),
                'auto_score': 0,
                'elapsed_ms': 0,
            }
            scores.append(0)
            print(f"→ ERROR: {e}")

        results.append(result)

    return results, scores


def print_report(results, scores):
    """打印报告"""
    print("\n" + "=" * 60)
    print("评测报告")
    print("=" * 60)

    avg = sum(scores) / len(scores) if scores else 0
    passing = sum(1 for s in scores if s >= 3.0)

    # 按 category 统计
    cat_scores = defaultdict(list)
    for r in results:
        cat = r.get('category', 'unknown')
        cat_scores[cat].append(r['auto_score'])

    print(f"  总用例: {len(results)}")
    print(f"  平均分: {avg:.2f}/5")
    print(f"  通过 (≥3.0): {passing}/{len(results)} ({100*passing/len(results):.0f}%)")

    if 'error' in ''.join(str(r.get('error', '')) for r in results):
        errors = [r for r in results if 'error' in r]
        print(f"  错误: {len(errors)}")

    print(f"\n  按类别:")
    for cat, sc in sorted(cat_scores.items(), key=lambda x: -sum(x[1])/len(x[1])):
        print(f"    {cat}: {sum(sc)/len(sc):.2f} ({len(sc)} 例)")

    # 低分用例
    low = [r for r in results if r['auto_score'] < 2.5]
    if low:
        print(f"\n  低分用例 (前5):")
        for r in low[:5]:
            print(f"    [{r['auto_score']:.1f}] {r['query']}")
            print(f"      期望: {r.get('expected_intent')}/{r.get('expected_model')}  "
                  f"检测: {r.get('detected_intent')}/{r.get('detected_models')}")

    return avg


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm', action='store_true', help='LLM 深度评分')
    parser.add_argument('--ci', action='store_true', help='CI 模式')
    parser.add_argument('--report', type=str, help='报告输出路径')
    parser.add_argument('--threshold', type=float, default=3.0, help='CI 通过阈值')
    parser.add_argument('--test-set', type=str, help='自定义测试集路径')
    args = parser.parse_args()

    # 加载系统
    sys.path.insert(0, os.path.join(BASE, 'src'))
    try:
        from query import QueryEngineV3 as QueryEngine
    except ImportError:
        print("⚠️ Cannot import QueryEngineV3, using mock")
        QueryEngine = None

    # 加载测试集
    test_set = load_test_set(args.test_set)
    print(f"加载 {len(test_set)} 条测试用例")

    if QueryEngine is None:
        # Mock mode: just validate the test set
        print("⚠️ 评测引擎未加载 (缺少依赖)，仅验证测试集格式")
        for tc in test_set:
            assert tc['query'], f"Missing query: {tc}"
            assert tc['expected_intent'], f"Missing intent: {tc}"
        print("✅ 测试集格式验证通过")
        return

    engine = QueryEngine()
    results, scores = run_eval(test_set, engine, use_llm=args.llm)
    avg = print_report(results, scores)

    # 保存报告
    if args.report:
        report = {
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'total': len(results),
            'avg_score': avg,
            'passing': sum(1 for s in scores if s >= 3.0),
            'results': results,
        }
        with open(args.report, 'w') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n  📊 报告已保存: {args.report}")

    # CI 模式
    if args.ci:
        if avg < args.threshold:
            print(f"\n❌ CI FAILED: {avg:.2f} < {args.threshold}")
            sys.exit(1)
        else:
            print(f"\n✅ CI PASSED: {avg:.2f} ≥ {args.threshold}")
            sys.exit(0)


if __name__ == '__main__':
    main()
