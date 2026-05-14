#!/usr/bin/env python3
"""Add LLM group evaluation to query.py"""
import re

with open('query.py') as f:
    content = f.read()

# Add _llm_evaluate_groups method
old_search = """        return None

    def _product_overview"""

new_method = """        return None

    def _llm_evaluate_groups(self, query, models, intents, candidates):
        \"\"\"LLM 评估候选组: 前几名分数接近时，选最相关的\"\"\"
        if not self.ds_key or len(candidates) < 2:
            return candidates[0] if candidates else None

        top3 = candidates[:3]
        lines = []
        for i, g in enumerate(top3, 1):
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
            "你是扫描器技术文档的评估助手。用户问题与多个候选文档匹配，选最相关的一个。\\n\\n"
            f"用户: \\"{query}\\"\\n"
            f"型号: {models}\\n"
            f"意图: {intents}\\n\\n"
            f"候选文档:\\n{groups_text}\\n\\n"
            "只返回最匹配的序号 [1] [2] 或 [3]，不要多余文字。"
        )
        result = self._call_deepseek(prompt, 30)
        if not result:
            return candidates[0]

        for idx_str in ['[1]', '[2]', '[3]']:
            if idx_str in result:
                idx = int(idx_str.strip('[]')) - 1
                if 0 <= idx < len(top3):
                    return top3[idx]
        for w in result.split():
            try:
                idx = int(w.strip('[](). ')) - 1
                if 0 <= idx < len(top3):
                    return top3[idx]
            except:
                pass
        return candidates[0]

    def _product_overview"""

content = content.replace(old_search, new_method)

# Add LLM evaluation step in query method
old_query = """        response = self.build_response(groups, models, intents, llm_used=llm_used)
        return {"""

new_query = """        # LLM 评估候选组（有多个候选时）
        if len(groups) >= 2 and models and self.ds_key:
            llm_best = self._llm_evaluate_groups(user_text, models, intents, groups)
            if llm_best and llm_best != groups[0]:
                groups.remove(llm_best)
                groups.insert(0, llm_best)
                llm_used = True

        response = self.build_response(groups, models, intents, llm_used=llm_used)
        return {"""

content = content.replace(old_query, new_query)

with open('query.py', 'w') as f:
    f.write(content)
print("Done")
