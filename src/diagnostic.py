#!/usr/bin/env python3
"""
diagnostic.py — 对话式诊断流程

支持的诊断场景:
  - 扫不出来/没反应 → 通信测试 + 恢复出厂
  - 连不上/配对失败 → 蓝牙配对步骤
  - 输出乱码/不对 → 数据格式 + 后缀检查

流程结构:
  start → ask_model → show_steps → (resolved | escalate)

用法:
  from diagnostic import DiagnosticEngine
  de = DiagnosticEngine(engine)  # engine = QueryEngineV3
  result = de.handle(query, session_state)
"""

import re, json, os
from collections import defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GROUPS_PATH = os.path.join(BASE, 'data', 'image_groups.json')


# ═══ 诊断场景定义 ═══

DIAGNOSTIC_SCENARIOS = {
    'test_comm': {
        'keywords': ['扫不出', '没反应', '没输出', '没数据', '不好使', '不行',
                     '扫了没', '不工作', '无法扫描', '读不到', '没响应',
                     '扫不了', '不出', '不识别', '不能扫'],
        'title': '🔧 扫描故障排查',
        'steps': [
            {
                'text': '检查线缆连接是否松动，重新插拔USB/串口线',
                'need_image': False,
            },
            {
                'text': '扫描测试码验证通信是否正常',
                'need_image': True,
                'config_type': 'test_code',
            },
            {
                'text': '恢复出厂设置',
                'need_image': True,
                'config_type': 'restore_factory',
            },
            {
                'text': '重新扫描测试码确认问题是否解决',
                'need_image': True,
                'config_type': 'test_code',
            },
        ],
        'escalate': '如以上步骤无法解决，请回复「人工」转接技术支持。'
    },
    'pairing': {
        'keywords': ['连不上', '配对失败', '蓝牙连不上', '搜索不到', '连接不上',
                     '断开', '掉线', '连不了', '无法连接', '配对不到'],
        'title': '🔗 蓝牙配对故障排查',
        'steps': [
            {
                'text': '确认底座已通电（底座指示灯亮起）',
                'need_image': False,
            },
            {
                'text': '扫描断开配对码，清除旧配对',
                'need_image': True,
                'config_type': 'pairing',
                'config_hint': 'DELINK|断开|取消配对',
            },
            {
                'text': '重新扫描底座配对码',
                'need_image': True,
                'config_type': 'pairing',
                'config_hint': 'LNKBT|配对码|底座',
            },
            {
                'text': '等待5-8秒，听提示音确认配对成功',
                'need_image': False,
            },
        ],
        'escalate': '如仍无法配对，请回复「人工」转接技术支持。'
    },
    'data_issue': {
        'keywords': ['乱码', '输出不对', '多了', '少了', '不对', '错误',
                     '串码', '输不出', '多字符', '少字符', '格式不对',
                     '数据错', '显示不对'],
        'title': '📝 数据输出故障排查',
        'steps': [
            {
                'text': '检查输入法是否为英文状态',
                'need_image': False,
            },
            {
                'text': '清除所有后缀/前缀设置',
                'need_image': True,
                'config_type': 'suffix',
                'config_hint': '清除|删除|移除|SUFCA2|DFMCA3',
            },
            {
                'text': '恢复出厂设置',
                'need_image': True,
                'config_type': 'restore_factory',
            },
            {
                'text': '重新测试扫码输出',
                'need_image': False,
            },
        ],
        'escalate': '如仍无法解决，请回复「人工」转接技术支持。'
    },
}


class DiagnosticEngine:
    def __init__(self, query_engine=None):
        self.query_engine = query_engine
        self._groups = None
    
    def _load_groups(self):
        if self._groups is None:
            with open(GROUPS_PATH) as f:
                self._groups = json.load(f)
        return self._groups
    
    def detect_scenario(self, query):
        """检测查询属于哪个诊断场景"""
        scores = {}
        for scenario_id, scenario in DIAGNOSTIC_SCENARIOS.items():
            score = sum(1 for kw in scenario['keywords'] if kw in query)
            if score > 0:
                scores[scenario_id] = score
        
        if not scores:
            return None
        
        return max(scores, key=scores.get)
    
    def start_diagnosis(self, query):
        """
        开始诊断流程
        
        返回: {
            'type': 'diagnostic_start',
            'scenario': 'test_comm',
            'message': '请告诉我产品型号...',
            'state': {'scenario': 'test_comm', 'step': 'ask_model'},
        }
        """
        scenario_id = self.detect_scenario(query)
        if not scenario_id:
            return None
        
        scenario = DIAGNOSTIC_SCENARIOS[scenario_id]
        
        return {
            'type': 'diagnostic_start',
            'scenario': scenario_id,
            'message': f"{scenario['title']}\n\n请告诉我产品型号，例如 1900、1902、OH430、HH760、7680g 等。\n\n直接回复型号即可。",
            'state': {
                'scenario': scenario_id,
                'step': 'ask_model',
            },
        }
    
    def continue_diagnosis(self, user_input, state):
        """
        继续诊断流程
        
        state: 上一步保存的状态
        
        返回: {
            'type': 'diagnostic_steps' | 'diagnostic_complete',
            'message': '...',
            'segments': [...],  # 文字+图片
            'state': {...},
        }
        """
        scenario_id = state.get('scenario')
        step = state.get('step')
        
        if not scenario_id or scenario_id not in DIAGNOSTIC_SCENARIOS:
            return None
        
        scenario = DIAGNOSTIC_SCENARIOS[scenario_id]
        
        # Step 1: 收集型号
        if step == 'ask_model':
            # 提取型号
            model = self._extract_model(user_input)
            if not model:
                return {
                    'type': 'diagnostic_steps',
                    'message': '请提供产品型号，例如 1900、OH430、HH760。直接回复型号即可。',
                    'state': state,
                }
            
            state['model'] = model
            state['step'] = 'show_steps'
            state['step_index'] = 0
            
            return self._build_step_response(scenario, state, 0)
        
        # Step 2: 逐步推进
        if step == 'show_steps':
            # 检查是否退出
            if user_input.strip() in ['人工', '转人工', '客服', 'help']:
                return {
                    'type': 'diagnostic_complete',
                    'message': '正在为您转接人工技术支持...\n\n请描述您的问题和已尝试的步骤，客服将尽快回复。',
                    'state': None,
                }
            
            # 检查是否继续（用户回复了"好了""可以了"）
            if any(kw in user_input for kw in ['好了', '可以了', '解决了', 'ok', '行了']):
                return {
                    'type': 'diagnostic_complete',
                    'message': '很高兴问题已解决！如还有其他问题，随时提问。',
                    'state': None,
                }
            
            # 推进到下一步
            step_index = state.get('step_index', 0) + 1
            state['step_index'] = step_index
            
            if step_index >= len(scenario['steps']):
                return {
                    'type': 'diagnostic_complete',
                    'message': scenario['escalate'],
                    'state': None,
                }
            
            return self._build_step_response(scenario, state, step_index)
        
        return None
    
    def _extract_model(self, text):
        """从用户输入提取产品型号"""
        text_upper = text.strip().upper()
        
        # 精确型号匹配列表
        models = [
            'XEN197X', '1900', '1902', '1952', '1910I', '1911I', '1920I', '1990I', '1991I',
            '1472', '1470G', '1202G',
            'OH430', 'OH4502', 'OH4503', 'OH350',
            'HH760', 'HH762', 'HH490', 'HH492',
            'HF680', 'HF600', '7680G', '7580G', '3320G', '7120',
            'MS7120', '8680I', 'PC300T',
            'PM42', 'PM43', 'PM45', 'PX240', 'PX940',
        ]
        
        # 优先精确匹配
        for model in models:
            if model in text_upper:
                return model
        
        # 模糊: 1900-C, 1900h 等
        fuzzy = [
            (r'1900', '1900'), (r'1902', '1902'), (r'195[02]', '1952'),
            (r'OH[48][35]0', 'OH430'), (r'HH[479][69]0', 'HH760'),
            (r'76[58]0', '7680g'), (r'33[20]0', '3320g'),
        ]
        for pat, model in fuzzy:
            if re.search(pat, text, re.IGNORECASE):
                return model
        
        return None
    
    def _find_config_image(self, model, config_type, config_hint=None):
        """从 image_groups + image_index 找配置码图片"""
        groups = self._load_groups()
        
        # 1. 找匹配型号 + 类型的步骤组
        best_steps = None
        for g in groups:
            g_model = (g.get('model', '') or '').replace(',', ' ')
            g_subcat = g.get('subcategory', '')
            
            if g_subcat != config_type:
                continue
            if model not in g_model:
                # 检查 applicable_models
                am = g.get('applicable_models', [])
                found_in_am = False
                for entry in am:
                    tags = entry.get('tags', [])
                    if model in tags:
                        found_in_am = True
                        break
                if not found_in_am:
                    continue
            
            steps = g.get('steps', [])
            if steps:
                if config_hint:
                    # 有 hint 时优先匹配
                    for s in steps:
                        ctx = s.get('context_text', '') or ''
                        if re.search(config_hint, ctx, re.IGNORECASE):
                            best_steps = [s]
                            break
                if not best_steps:
                    best_steps = steps[:1]  # 取第一个步骤
                break
        
        if not best_steps:
            pass  # Fall through to config_codes fallback below
        
        # 2. 从 image_index 找 URL
        import json as _json
        idx_path = os.path.join(BASE, 'data', 'image_index.json')
        with open(idx_path) as f:
            index = _json.load(f)
        
        if best_steps:
            for s in best_steps:
                fn = s.get('file_name', '')
                if fn:
                    for img in index:
                        if isinstance(img, dict) and img.get('file_name') == fn:
                            url = img.get('image_url', '')
                            if url:
                                label = s.get('context_text', '') or s.get('subcategory', '')
                                return {'url': url, 'label': label}
        
        # 3. Fallback: 从 config_codes + kb-images 找
        codes_path = os.path.join(BASE, 'data', 'config_codes.json')
        with open(codes_path) as f:
            codes = _json.load(f)
        
        # 类型映射: diagnostic config_type → LLM category keywords
        type_map = {
            'test_code': ['test_code', 'test_comm', '测试', 'test', '通信'],
            'restore_factory': ['restore_factory', '恢复出厂', 'default', 'reset', '出厂'],
            'suffix': ['suffix', '后缀', '回车', '换行'],
            'prefix': ['prefix', '前缀'],
            'pairing': ['pairing', '配对', '蓝牙', 'bluetooth', 'link'],
        }
        
        target_kws = type_map.get(config_type, [config_type])
        
        for c in codes:
            c_model = c.get('model', '')
            if model not in c_model:
                continue
            
            desc = c.get('description', '')
            cat = c.get('category', '')
            combined = f"{cat} {desc}".lower()
            
            # 匹配类型关键词
            if not any(kw.lower() in combined for kw in target_kws):
                continue
            
            url = c.get('image_url', '')
            if url and url != 'N/A':
                return {'url': url, 'label': desc[:60]}
        
        return None
    
    def _build_step_response(self, scenario, state, step_index):
        """构建单步响应"""
        step = scenario['steps'][step_index]
        message_parts = []
        segments = []
        
        # 标题
        if step_index == 0:
            message_parts.append(f"{scenario['title']} — {state.get('model', '')}")
        
        # 步骤编号 + 文字
        message_parts.append(f"\n步骤 {step_index + 1}/{len(scenario['steps'])}：{step['text']}")
        
        # 图片
        if step.get('need_image'):
            img = self._find_config_image(
                state.get('model', ''),
                step.get('config_type', 'test_code'),
                step.get('config_hint'),
            )
            if img and img.get('url'):
                segments.append({
                    'type': 'image',
                    'url': img['url'],
                    'label': img.get('label', ''),
                })
        
        # 导航提示
        if step_index < len(scenario['steps']) - 1:
            message_parts.append('\n回复任意内容进入下一步，或回复「人工」转接。')
        else:
            message_parts.append(f"\n{scenario['escalate']}")
        
        return {
            'type': 'diagnostic_steps',
            'message': '\n'.join(message_parts),
            'segments': segments,
            'state': state,
        }


if __name__ == '__main__':
    # 测试
    de = DiagnosticEngine()
    
    # 测试检测
    for q in ["扫不出来", "蓝牙连不上", "输出乱码", "怎么设置串口"]:
        s = de.detect_scenario(q)
        print(f"'{q}' → {s}")
    
    # 测试流程
    print("\n=== 完整流程模拟 ===")
    result = de.start_diagnosis("扫不出来")
    if result:
        print(f"Bot: {result['message']}")
        
        # 用户回复型号
        r2 = de.continue_diagnosis("1900", result['state'])
        if r2:
            print(f"\nUser: 1900")
            print(f"Bot: {r2['message'][:200]}...")
            print(f"   图片: {len(r2.get('segments', []))} 张")
