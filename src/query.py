#!/usr/bin/env python3
"""
查询引擎 v3 — YAML 配置 + 混合检索 (向量 + BM25)
- INTENT_MAP 从 intents.yaml 热加载
- 规则优先 → 混合检索兜底
- 支持 jieba 分词 + bge-small-zh 语义匹配
"""

import json
import os
import re
import sys
import time
import urllib.parse
import requests
import yaml

# ── 路径 ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
INTENTS_YAML = os.path.join(ROOT_DIR, "intents.yaml")
INDEX_PATH = os.path.join(ROOT_DIR, "data/image_index.json")
GROUPS_PATH = os.path.join(ROOT_DIR, "data/image_groups.json")
IMG_BASE_URL = "http://172.24.59.194:8098/kb"

DEEPSEEK_API = "https://api.deepseek.com/chat/completions"

# ── 型号识别规则 ──
MODEL_PATTERNS = [
    (r'OH430', 'OH430'), (r'OH450[23]', 'OH4502'), (r'OH460', 'OH460'),
    (r'SC2800', 'SC2800'), (r'OH420', 'OH420'), (r'HF680', 'HF680'),
    (r'HF600', 'HF600'), (r'OH350', 'OH350X'),
    (r'1900[-\s]?C', '1900-C'), (r'1900', '1900'),
    (r'1902[-\s]?C', '1902-C'), (r'1902', '1902'),
    (r'1952', '1952'), (r'1950', '1950'), (r'1981i', '1981i'), (r'1991i', '1991i'), (r'1990i', '1990i'),
    (r'1911i', '1911i'), (r'1472', '1472'), (r'1470', '1470'),
    (r'HH490', 'HH490'), (r'HH492', 'HH492'), (r'HH760', 'HH760'),
    (r'HH762', 'HH762'), (r'[0O]H462', 'OH462'), (r'[0O]H460', 'OH460'),
    (r'3320g', '3320g'), (r'33x0g', '33x0g'),
    (r'7120', '7120-2D'), (r'7580g', '7580g'), (r'7680g', '7680g'),
    (r'MS7120', 'MS7120'), (r'1202g', '1202g'),
    (r'PM42', 'PM42'), (r'PM43', 'PM43'), (r'PM45', 'PM45'), (r'PX240', 'PX240'),
    (r'PX940', 'PX940'), (r'PC300T', 'PC300T'), (r'8680i', '8680i'),
]


def load_intents(yaml_path=INTENTS_YAML):
    """从 YAML 加载 INTENT_MAP，支持热更新"""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config.get('intents', {})


class QueryEngineV3:
    def __init__(self, yaml_path=INTENTS_YAML):
        self.yaml_path = yaml_path
        self.intent_map = load_intents(yaml_path)
        self.yaml_mtime = os.path.getmtime(yaml_path) if os.path.exists(yaml_path) else 0
        self.index = self._load(INDEX_PATH)
        self.groups = self._load(GROUPS_PATH)
        self.index_by_name = {e['file_name']: e for e in self.index if e.get('file_name')}
        self.ds_key = self._load_ds_key()
        self._retriever = None  # lazy init
    
    def _get_retriever(self):
        if self._retriever is None:
            from retriever import get_retriever
            self._retriever = get_retriever()
        return self._retriever
    
    def _reload_intents(self):
        """检查 YAML 是否更新，自动热加载"""
        try:
            mtime = os.path.getmtime(self.yaml_path)
            if mtime > self.yaml_mtime:
                self.intent_map = load_intents(self.yaml_path)
                self.yaml_mtime = mtime
                return True
        except:
            pass
        return False
    
    def _load(self, path):
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    
    def _load_ds_key(self):
        try:
            cfg = json.load(open('/home/admin/.openclaw/openclaw.json'))
            return cfg['models']['providers']['deepseek']['apiKey']
        except:
            return None
    
    def _call_deepseek_chat(self, prompt, max_tokens=50):
        if not self.ds_key:
            return None
        try:
            resp = requests.post(DEEPSEEK_API, headers={
                "Authorization": f"Bearer {self.ds_key}", "Content-Type": "application/json"
            }, json={
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens, "temperature": 0.1,
            }, timeout=15)
            if resp.status_code == 200:
                return resp.json()['choices'][0]['message'].get('content', '')
        except:
            return None
        return None
    
    def _call_deepseek(self, prompt, max_tokens=300):
        if not self.ds_key:
            return None
        try:
            resp = requests.post(DEEPSEEK_API, headers={
                "Authorization": f"Bearer {self.ds_key}", "Content-Type": "application/json"
            }, json={
                "model": "deepseek-v4-flash",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens, "temperature": 0.1,
            }, timeout=15)
            if resp.status_code == 200:
                return resp.json()['choices'][0]['message'].get('content', '')
        except:
            return None
        return None
    
    # ── 型号识别 ──
    def extract_models(self, query):
        found = []
        for pattern, model in MODEL_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                found.append(model)
        return list(set(found))
    
    # ── 规则意图识别 ──
    def _rule_intent(self, query):
        self._reload_intents()
        scores = {}
        for intent, config in self.intent_map.items():
            score = sum(1 for kw in config.get('keywords', []) if kw in query)
            if score > 0:
                scores[intent] = score
        if not scores:
            return []
        return [si[0] for si in sorted(scores.items(), key=lambda x: -x[1])]
    
    # ── LLM 意图理解 ──
    def _llm_intent(self, query, models):
        self._reload_intents()
        intent_lines = []
        for name, cfg in sorted(self.intent_map.items()):
            ex_str = ''
            if cfg.get('examples'):
                ex_str = ' | 例: ' + ', '.join(cfg['examples'][:2])
            intent_lines.append(f'  "{name}": {cfg.get("desc", "")}{ex_str}')
        intent_text = '\n'.join(intent_lines)
        
        # 构建 few-shot
        shot_lines = []
        for name, cfg in sorted(self.intent_map.items()):
            for ex in cfg.get('examples', [])[:1]:
                shot_lines.append(f'  用户: "{ex}" -> "{name}"')
        shot_text = '\n'.join(shot_lines[:10])
        
        prompt = f'''你是工业扫描器技术支持系统的意图分类器。
根据用户问题判断用户意图，只返回一个意图名称。

可用意图:
{intent_text}

示例:
{shot_text}

用户: "{query}"
型号: {models or '未指定'}

意图:'''
        result = self._call_deepseek(prompt, 100)
        if result:
            result = result.strip().strip('"').strip("'")
            for name in self.intent_map:
                if name in result:
                    return [name]
        return ['general_setup']
    
    def detect_intent(self, query):
        self._reload_intents()
        intents = self._rule_intent(query)
        models = self.extract_models(query)
        
        pov_keywords = self.intent_map.get('product_overview', {}).get('keywords', [])
        has_pov_kw = sum(1 for kw in pov_keywords if kw in query)
        
        if intents:
            if len(intents) == 1 and intents[0] == 'general_setup':
                gs_kws = self.intent_map.get('general_setup', {}).get('keywords', [])
                kw_count = sum(1 for kw in gs_kws if kw in query)
                if kw_count <= 1:
                    llm_result = self._llm_intent(query, models)
                    if llm_result:
                        if 'product_overview' in llm_result and not has_pov_kw:
                            return intents
                        return llm_result
            return intents
        
        llm_result = self._llm_intent(query, models)
        if llm_result:
            if 'product_overview' in llm_result and not has_pov_kw:
                remaining = query
                for m in models:
                    remaining = re.sub(r'\s*' + re.escape(m) + r'\s*', '', remaining, flags=re.IGNORECASE)
                if remaining.strip() and len(remaining.strip()) > 2:
                    return ['general_setup']
            return llm_result
        
        return ['general_setup']
    
    # ── 规则组匹配 ──
    def find_groups(self, models, intents, query=''):
        if not models:
            matched = list(self.groups)
        else:
            matched = []
            for g in self.groups:
                g_models = g.get('applicable_models', [])
                for gm in g_models:
                    gm_name = gm.get('full_name') or gm.get('model') or ''
                    for m in models:
                        if m.lower() in gm_name.lower():
                            matched.append(g)
                            break
                    else:
                        continue
                    break
        
        if not matched:
            return []
        
        intent_subs = set()
        for intent in intents:
            cfg = self.intent_map.get(intent, {})
            for sub in cfg.get('subcategories', []):
                intent_subs.add(sub)
        
        query_lower = query.lower()
        query_keywords = set(re.findall(r'[\w\u4e00-\u9fff]+', query_lower))
        
        scored = []
        
        INTENT_CONTRADICT = {
            'usb_connect': ['串口', 'rs232', 'ps2', 'keyboard wedge'],
            'serial_connect': ['usb', '键盘口'],
            'bluetooth_pairing': ['串口', 'ps2'],
            'virtual_com_port': ['ps2', 'keyboard wedge'],
        }
        
        for g in matched:
            score = 0
            g_subs = set(s.get('subcategory') for s in g.get('steps', []))
            score += len(intent_subs & g_subs) * 2
            doc_name = (g.get('doc_name') or '').lower()
            
            for kw in query_keywords:
                if len(kw) > 1 and kw in doc_name:
                    score += 1
            
            for s in g.get('steps', []):
                ctx = (s.get('context_text') or '').lower()
                for kw in query_keywords:
                    if len(kw) > 1 and kw in ctx:
                        score += 0.5
                        break
            
            if '等' in doc_name and models:
                score -= 2
            
            for intent in intents:
                contradictions = INTENT_CONTRADICT.get(intent, [])
                for c in contradictions:
                    if c in doc_name:
                        score -= 3
                        break
            
            for model in models:
                if model.lower() in doc_name:
                    score += 5
                    break
            
            INTENT_BOOST = {
                'usb_connect': ['usb', '键盘口', 'usb线缆'],
                'serial_connect': ['串口', 'rs232', 'com'],
                'bluetooth_pairing': ['蓝牙', '无线', '配对'],
                'add_suffix': ['回车', '后缀', '换行', 'crlf'],
                'restore_factory': ['恢复出厂'],
                'test_comm': ['测试', '通信验证'],
                'pairing': ['配对', '蓝牙'],
                'interface_mode': ['接口', '接口码'],
            }
            for intent in intents:
                boosts = INTENT_BOOST.get(intent, [])
                for b in boosts:
                    if b in doc_name:
                        score += 2
            
            scored.append((score, g))
        
        scored.sort(key=lambda x: -x[0])
        matched = [g for _, g in scored]
        
        seen = set()
        unique = []
        for g in matched:
            gid = g.get('group_id', '')
            if gid not in seen:
                seen.add(gid)
                unique.append(g)
        return unique[:10]
    
    # ── 🔥 混合检索 / RRF 融合 ──
    def _merge_rule_hybrid(self, rule_groups, hybrid_results, models):
        """RRF 融合规则匹配和混合检索结果"""
        scores = {}
        for rank, g in enumerate(rule_groups[:5]):
            gid = g.get("group_id", "")
            scores[gid] = 1.0 / (60 + rank + 1)
        for rank, h in enumerate(hybrid_results[:5]):
            gid = h.get("group_id", "")
            scores[gid] = scores.get(gid, 0) + 1.0 / (60 + rank + 1)
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        merged = []
        seen = set()
        for gid, _ in ranked:
            for g in rule_groups:
                if g.get("group_id") == gid and gid not in seen:
                    merged.append(g)
                    seen.add(gid)
                    break
            else:
                for h in hybrid_results:
                    if h.get("group_id") == gid and gid not in seen:
                        merged.append(h)
                        seen.add(gid)
                        break
        return merged
    
    # ── 🔥 混合检索兜底 (向量 + BM25) ──
    def _hybrid_search(self, query, models, top_k=5):
        """规则匹配失败时 → 用混合检索 (语义 + 关键词)"""
        try:
            retriever = self._get_retriever()
            raw_results = retriever.search(query, top_k=top_k)
            
            # 型号过滤
            if models:
                filtered = []
                for r in raw_results:
                    for m in r.get('applicable_models', []):
                        mn = m.get('full_name') or m.get('model') or ''
                        for mm in models:
                            if mm.lower() in mn.lower():
                                filtered.append(r)
                                break
                        else:
                            continue
                        break
                
                # 有型号匹配 → 返回
                if filtered:
                    return filtered[:top_k]
            
            return raw_results[:top_k]
        except Exception as e:
            print(f"  [Hybrid] 检索失败: {e}")
            return None
    
    # ── LLM 知识库检索 ──
    def _llm_search_kb(self, query, models):
        if not self.ds_key or not models:
            return None
        
        candidates = []
        for g in self.groups:
            for m in g.get('applicable_models', []):
                mn = m.get('full_name') or m.get('model') or ''
                for model in models:
                    if model.lower() in mn.lower():
                        candidates.append(g)
                        break
                else:
                    continue
                break
        
        candidates = candidates[:10]
        if not candidates:
            return None
        
        kb_lines = []
        for idx, g in enumerate(candidates, 1):
            doc_name = g.get('doc_name', '')
            steps_text = []
            for s in g.get('steps', [])[:6]:
                sub = s.get('subcategory') or '?'
                ctx = (s.get('context_text') or '')[:80]
                steps_text.append(f'    [{s["step_order"]}] {sub}: {ctx}')
            steps_str = '\n'.join(steps_text)
            kb_lines.append(f'[{idx}] {doc_name}\n{steps_str}')
        kb_text = '\n'.join(kb_lines)
        
        prompt = f'''你是扫描器知识库检索助手。从以下候选中选出最相关的。

用户问题: "{query}"
型号: {models}

候选方案:
{kb_text}

选出最相关的一个方案。只返回序号 [1] 或空。'''
        
        result = self._call_deepseek(prompt, 100)
        if not result:
            return None
        
        result = result.strip()
        for idx_str in ['[1]', '[2]', '[3]', '[4]', '[5]', '[6]', '[7]', '[8]', '[9]', '[10]']:
            if idx_str in result:
                idx = int(idx_str.strip('[]')) - 1
                if 0 <= idx < len(candidates):
                    return [candidates[idx]]
        
        for word in result.split():
            try:
                idx = int(word.strip('[](). ')) - 1
                if 0 <= idx < len(candidates):
                    return [candidates[idx]]
            except:
                pass
        return None
    
    def _llm_evaluate_groups(self, query, models, intents, candidates):
        if not self.ds_key or len(candidates) < 2:
            return candidates[0] if candidates else None
        
        top_all = candidates[:10]
        lines = []
        for i, g in enumerate(top_all, 1):
            doc = g.get('doc_name', '')
            lines.append(f"[{i}] {doc}")
            for s in g.get('steps', [])[:4]:
                sub = s.get('subcategory') or '?'
                ctx = (s.get('context_text') or '')[:60]
                if ctx:
                    lines.append(f"    - {sub}: {ctx}")
            lines.append("")
        
        groups_text = chr(10).join(lines)
        
        prompt = (
            "你是扫描器技术文档的评估助手。选最相关的一个。\n\n"
            f"用户: \"{query}\"\n型号: {models}\n意图: {intents}\n\n"
            f"候选文档:\n{groups_text}\n\n"
            "只返回最匹配的序号如 [1]。"
        )
        result = self._call_deepseek_chat(prompt, 30)
        if not result:
            return candidates[0]
        
        for idx_str in ['[1]', '[2]', '[3]', '[4]', '[5]', '[6]', '[7]', '[8]', '[9]', '[10]']:
            if idx_str in result:
                idx = int(idx_str.strip('[]')) - 1
                if 0 <= idx < len(top_all):
                    return top_all[idx]
        return candidates[0]
    
    def _product_overview(self, model):
        entries = []
        for e in self.index:
            for m in e.get('applicable_models', []):
                mn = m.get('full_name') or m.get('model') or ''
                if model.lower() in mn.lower():
                    entries.append(e)
                    break
        
        rel_groups = []
        for g in self.groups:
            for m in g.get('applicable_models', []):
                mn = m.get('full_name') or m.get('model') or ''
                if model.lower() in mn.lower():
                    rel_groups.append(g)
                    break
        
        config_codes = sum(1 for e in entries if e.get('category') == 'config_code')
        screenshots = sum(1 for e in entries if e.get('category') == 'screenshot')
        diagrams = sum(1 for e in entries if e.get('category') == 'diagram')
        
        all_subs = set()
        for g in rel_groups:
            for s in g.get('steps', []):
                if s.get('subcategory'):
                    all_subs.add(s['subcategory'])
        
        import csv
        model_info = {'category': '', 'series': '', 'variants': []}
        try:
            with open(os.path.join(ROOT_DIR, 'data/product_tree.csv')) as f:
                for row in csv.DictReader(f):
                    if row.get('型号', '').strip() == model:
                        model_info['category'] = row.get('大类', '')
                        model_info['series'] = row.get('系列', '')
                        v = row.get('子型号/SN变体', '').strip()
                        if v:
                            model_info['variants'].append(v)
        except:
            pass
        
        lines = []
        lines.append(f"📟 {model} 产品数据概览")
        if model_info['category']:
            lines.append(f"分类: {model_info['category']} | 系列: {model_info['series']}")
        lines.append('')
        lines.append('📊 知识库覆盖:')
        lines.append(f"  图片: {len(entries)} 张（配置码{config_codes} + 截图{screenshots} + 示意图{diagrams}）")
        lines.append(f"  步骤组: {len(rel_groups)} 组")
        lines.append('')
        
        func_labels = {
            'restore_factory': '恢复出厂设置', 'test_code': '通信测试',
            'setup': '设置模式', 'suffix': '后缀设置', 'prefix': '前缀设置',
            'interface': '接口模式切换', 'pairing': '蓝牙/无线配对',
            'feature': '功能配置（截取/替换/连续扫描等）',
        }
        if all_subs:
            lines.append('✅ 支持的功能:')
            for s in sorted(all_subs):
                label = func_labels.get(s, s)
                lines.append(f"  • {label}")
        lines.append('')
        lines.append(f'📄 相关文档 {len(rel_groups)} 份:')
        for g in rel_groups[:6]:
            doc = g.get('doc_name', '')
            steps = g.get('total_config_codes', 0)
            lines.append(f'  • {doc}（{steps}步）')
        if len(rel_groups) > 6:
            lines.append(f'  ... 还有 {len(rel_groups)-6} 份')
        
        return '\n'.join(lines)
    
    def _build_group_response(self, groups, models, intents, llm_used=False):
        if not groups:
            return self._no_match(models, intents)
        
        best = groups[0]
        steps = best.get('steps', [])
        
        lines = []
        doc_name = best.get('doc_name', '')
        models_str = ', '.join(models) if models else '通用'
        
        lines.append(f"📌 适用于: {models_str}")
        if llm_used:
            lines.append("🤖 混合检索增强")
        lines.append("")
        
        intent_subs = set()
        for intent in intents:
            cfg = self.intent_map.get(intent, {})
            for sub in cfg.get('subcategories', []):
                intent_subs.add(sub)
        
        for s in steps:
            sub = s.get('subcategory') or ''
            ctx = (s.get('context_text') or '').strip()
            raw_name = s.get('file_name', '')
            entry = self.index_by_name.get(raw_name, {})
            img_url = entry.get('image_url')
            if img_url:
                prefix, fname = img_url.rsplit('/', 1)
                img_url = prefix + '/' + urllib.parse.quote(fname)
            # 也检查 steps 中的 image_url
            if not img_url and s.get('image_url'):
                img_url = s['image_url']
            
            func_label = {
                'restore_factory': '🔄 恢复出厂', 'test_code': '✅ 测试码',
                'setup': '🔧 进入/退出设置', 'suffix': '📝 后缀设置',
                'prefix': '📝 前缀设置', 'interface': '🔌 接口模式',
                'pairing': '📡 配对', 'feature': '⚙️ 功能设置',
            }.get(sub, f'📋 {sub}' if sub else '📋 配置码')
            
            is_match = sub in intent_subs
            marker = '→ ' if is_match else '  '
            
            step_text = f"{marker}步骤 {s['step_order']}: {func_label}"
            if ctx:
                step_text += f"\n   {ctx[:80]}"
            if img_url:
                step_text += f"\n   [图片: {img_url}]"
            
            lines.append(step_text)
            lines.append("")
        
        lines.append(f"📄 来源: {doc_name}")
        if len(groups) > 1:
            lines.append(f"\n💡 还有 {len(groups)-1} 个相关方案")
        
        return '\n'.join(lines)
    
    def _no_match(self, models, intents):
        intent_names = ', '.join(intents)
        if not models:
            return ("未找到匹配的方案。请提供更具体的产品型号（如 1900-C、OH430、1902）\n"
                    "或描述您想解决的问题（加回车、配对、恢复出厂、USB连接等）。")
        return f"未找到 {', '.join(models)} 关于 {intent_names} 的配置方案。\n请确认型号是否正确。"
    
    # ── 主入口 ──
    def query(self, user_text):
        models = self.extract_models(user_text)
        intents = self.detect_intent(user_text)
        groups = self.find_groups(models, intents, query=user_text)
        llm_used = False
        search_method = 'rule'
        hybrid_merged = False
        
        # 产品介绍类 → 返回编译的产品概览
        if 'product_overview' in intents and models:
            overview = self._product_overview(models[0])
            return {
                'query': user_text,
                'models': models,
                'intents': intents,
                'groups_found': 0,
                'llm_used': False,
                'search_method': 'product_overview',
                'response': overview,
            }
        
        # 🔥 混合检索：始终在后台运行，用于交叉验证和增强
        hybrid_results = None
        if models:
            hybrid_results = self._hybrid_search(user_text, models, top_k=5)
        
        # 规则找不到 → 混合检索补位
        if not groups and hybrid_results:
            groups = hybrid_results
            llm_used = True
            search_method = 'hybrid'
        
        # 规则 + 混合都有结果 → RRF 融合重排
        elif groups and hybrid_results:
            merged = self._merge_rule_hybrid(groups, hybrid_results, models)
            if merged and merged[0] != groups[0]:
                # 检查混合检索的 top1 是否更好
                hr_top = hybrid_results[0]
                rule_top = groups[0]
                # 如果混合结果的 doc 更相关（型号精确匹配 + 意图子类型匹配），替换
                hr_doc = hr_top.get('doc_name', '')
                rule_doc = rule_top.get('doc_name', '')
                # 简单启发式：混合结果 top1 含型号名而规则 top1 不含 → 替换
                hr_has_model = any(m.lower() in hr_doc.lower() for m in models)
                rule_has_model = any(m.lower() in rule_doc.lower() for m in models)
                if hr_has_model and not rule_has_model:
                    groups = [hybrid_results[0]] + groups
                    hybrid_merged = True
                    llm_used = True
                    search_method = 'rule+hybrid'
                elif groups[0] not in hybrid_results[:3]:
                    # 规则 top1 不在混合 top3 → 混合 top1 插入
                    groups = [hybrid_results[0]] + groups
                    hybrid_merged = True
                    llm_used = True
                    search_method = 'rule+hybrid'
        
        # 混合检索也找不到 → LLM 知识库检索兜底
        if not groups and models:
            llm_groups = self._llm_search_kb(user_text, models)
            if llm_groups:
                groups = llm_groups
                llm_used = True
                search_method = 'llm_kb'
        
        # 无型号
        if not models:
            response = ("🔍 请提供您的产品型号，我能给出更精确的配置方案。\n\n"
                       "例如：\n"
                       "  • 手持扫描枪：1900-C、1902、1952、HH490、HH760\n"
                       "  • 固定式扫码器：OH430、OH4502、HF680\n"
                       "  • 平台式扫描器：7580g、7680g\n"
                       "  • 打印机：PM42、PM43、PX240\n\n"
                       "请这样问：\"1900-C USB连接\" \"OH430 加回车\" \"HH760 配对\"")
            return {
                'query': user_text,
                'models': models,
                'intents': intents,
                'groups_found': 0,
                'llm_used': False,
                'search_method': 'no_model',
                'response': response,
            }
        
        # LLM 评估候选组
        if len(groups) >= 2 and models and self.ds_key:
            llm_best = self._llm_evaluate_groups(user_text, models, intents, groups)
            if llm_best and llm_best != groups[0]:
                groups.remove(llm_best)
                groups.insert(0, llm_best)
                llm_used = True
        
        response = self._build_group_response(groups, models, intents, llm_used=llm_used)
        return {
            'query': user_text,
            'models': models,
            'intents': intents,
            'groups_found': len(groups),
            'llm_used': llm_used,
            'search_method': search_method,
            'response': response,
        }


def main():
    if len(sys.argv) < 2:
        print("用法: python3 query.py <问题>")
        sys.exit(1)
    
    query_text = ' '.join(sys.argv[1:])
    engine = QueryEngineV3()
    result = engine.query(query_text)
    
    print("=" * 60)
    print(f"问题: {result['query']}")
    print(f"识别型号: {result['models'] or '(未识别)'}")
    print(f"识别意图: {result['intents']}")
    print(f"匹配方案: {result['groups_found']} 组")
    print(f"检索方式: {result.get('search_method', 'rule')}")
    if result['llm_used']:
        print(f"🤖 LLM辅助检索")
    print("=" * 60)
    print()
    print(result['response'])


if __name__ == '__main__':
    main()
