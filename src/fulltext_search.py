#!/usr/bin/env python3
"""
fulltext_search.py — 全文检索引擎 (纯 Python, 无外部依赖)

从 fulltext.json 加载 4,256 个文本块，提供 TF-IDF 关键词搜索。
集成到 query.py 中作为兜底检索源。
"""

import json, os, re, math
from collections import defaultdict, Counter

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FULLTEXT_PATH = os.path.join(BASE, 'data', 'fulltext.json')


class FulltextSearch:
    def __init__(self, fulltext_path=None):
        if fulltext_path is None:
            fulltext_path = FULLTEXT_PATH

        with open(fulltext_path) as f:
            self.chunks = json.load(f)
        self._init()
        self.chunk_count = len(self.chunks)

    def _tokenize(self, text):
        """分词: 英文 word / 中文 char-bigram"""
        has_cjk = bool(re.search(r'[\u4e00-\u9fff]', text))
        if has_cjk:
            # 中文: char bigram
            cleaned = re.sub(r'[^\u4e00-\u9fff]', '', text)
            return [cleaned[i:i+2] for i in range(len(cleaned)-1)]
        # 英文: lowercase words ≥2 chars
        return re.findall(r'[a-z0-9]{2,}', text.lower())

    def _init(self):
        """构建 TF-IDF 索引"""
        self.docs = []
        self.doc_meta = []
        for c in self.chunks:
            tokens = self._tokenize(c['text'])
            self.docs.append(Counter(tokens))
            self.doc_meta.append({
                'product': c['product'],
                'chapter': c['chapter'],
                'page': c['page'],
                'chars': len(c['text']),
            })

        # IDF
        N = len(self.docs)
        self.idf = defaultdict(float)
        for doc in self.docs:
            for word in doc:
                self.idf[word] += 1
        for word in self.idf:
            self.idf[word] = math.log(N / (1 + self.idf[word])) + 1

    def _tfidf_vector(self, tokens):
        """计算查询的 TF-IDF 向量"""
        vec = Counter(tokens)
        for w in vec:
            vec[w] *= self.idf.get(w, 0)
        return vec

    def _cosine(self, vec_q, doc_counter, doc_norm=None):
        """余弦相似度"""
        dot = sum(vec_q.get(w, 0) * doc_counter.get(w, 0) for w in vec_q)
        if dot <= 0:
            return 0
        q_norm = math.sqrt(sum(v * v for v in vec_q.values()))
        if doc_norm is None:
            doc_norm = math.sqrt(sum(v * v for v in doc_counter.values()))
        if q_norm == 0 or doc_norm == 0:
            return 0
        return dot / (q_norm * doc_norm)

    def search(self, query, product=None, top_k=5):
        """
        搜索全文
        
        返回: [{'text':..., 'product':..., 'chapter':..., 'page':..., 'score':...}]
        """
        # Doc norms (precompute)
        doc_norms = [math.sqrt(sum(v*v for v in d.values())) for d in self.docs]

        # 检查如果 query 基本是空白
        query_tokens = self._tokenize(query)
        
        # 从 query 提取产品关键词
        query_product = product
        if not query_product:
            for w in query_tokens:
                if w.upper() in {'XEN197X', '1900', '195X', '199X', '199XI', '196X', 'OCR'}:
                    query_product = w.upper()
                    break
        
        # TF-IDF vector for query
        vec_q = self._tfidf_vector(query_tokens)

        # Score all docs
        scored = []
        for i, doc in enumerate(self.docs):
            score = self._cosine(vec_q, doc, doc_norms[i])
            if score <= 0:
                continue
            
            # 产品名匹配加分
            if query_product and self.doc_meta[i]['product'] == query_product:
                score *= 2.0
            elif query_product:
                score *= 0.3
            
            scored.append((score, i))

        scored.sort(key=lambda x: -x[0])
        scored = scored[:top_k * 3]

        # 去重 + 取最佳
        seen = set()
        results = []
        for score, idx in scored:
            key = (self.doc_meta[idx]['product'], self.doc_meta[idx]['chapter'][:40])
            if key in seen:
                continue
            seen.add(key)

            results.append({
                'text': self.chunks[idx]['text'][:600],
                'product': self.doc_meta[idx]['product'],
                'chapter': self.doc_meta[idx]['chapter'],
                'page': self.doc_meta[idx]['page'],
                'score': round(score, 3),
                'chars': self.doc_meta[idx]['chars'],
            })

            if len(results) >= top_k:
                break

        return results

    # ═══ 中文查询桥接 ═══

    def _load_api_key(self):
        """加载 DeepSeek API key"""
        import urllib.request
        api_key = os.environ.get('DEEPSEEK_API_KEY', '')
        if not api_key:
            for cp in [os.path.expanduser('~/.openclaw/openclaw.json'),
                        '/home/admin/.openclaw/openclaw.json']:
                try:
                    with open(cp) as f:
                        cfg = json.load(f)
                    ds = cfg.get('models', {}).get('providers', {}).get('deepseek', {})
                    api_key = ds.get('apiKey', '')
                    if api_key: break
                except: continue
        return api_key

    def _extract_keywords_cn(self, query):
        """
        用 DeepSeek 从中文查询提取英文搜索关键词
        
        输入: "XEN197X 怎么配置串口通信参数"
        输出: {"product": "XEN197X", "keywords": ["serial port", "configuration", "RS232"]}
        """
        import urllib.request
        api_key = self._load_api_key()
        if not api_key:
            return None

        prompt = f"""Extract English search keywords from this Chinese scanner support query.

Query: {query}

Return ONLY JSON:
{{"product":"model_number_or_empty","keywords":["word1","word2","word3"]}}"""

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
            with urllib.request.urlopen(req, timeout=15) as resp:
                result = json.loads(resp.read())
                content = result['choices'][0]['message']['content'].strip()
                if content.startswith('```'):
                    content = re.sub(r'^```(?:json)?\s*', '', content)
                    content = re.sub(r'\s*```$', '', content)
                return json.loads(content)
        except Exception as e:
            return None

    def search_cn(self, query, top_k=5):
        """
        中文查询入口: 自动翻译关键词 → 英文全文搜索
        
        用法:
          results = ft.search_cn("XEN197X 怎么配置串口")
        """
        # 1. 提取中文查询的英文关键词
        extracted = self._extract_keywords_cn(query)
        
        if not extracted:
            # LLM 不可用, 退化为直接搜索
            return self.search(query, top_k=top_k)
        
        # 2. 构建英文搜索查询
        keywords = extracted.get('keywords', [])
        product = extracted.get('product', '')
        en_query = ' '.join(keywords) if keywords else query
        
        # 3. 搜索
        results = self.search(en_query, product=product if product else None, top_k=top_k)
        
        # 4. 附加元数据
        for r in results:
            r['cn_query'] = query
            r['en_keywords'] = en_query
        
        return results


if __name__ == '__main__':
    ft = FulltextSearch()
    print(f"Loaded {ft.chunk_count} chunks\n")

    for query, product in [
        ("XEN197X serial port configuration", 'XEN197X'),
        ("how to pair bluetooth scanner", None),
        ("restore factory defaults", None),
        ("keyboard country settings", 'XEN197X'),
        ("USB interface setup", '1900'),
    ]:
        print(f"🔍 '{query}':")
        results = ft.search(query, product=product, top_k=3)
        for r in results:
            print(f"  [{r['product']}] Ch {r['chapter'][:30]} p{r['page']} "
                  f"s={r['score']:.3f}: {r['text'][:80]}")
        if not results:
            print(f"  (no results)")
        print()
    
    # 中文查询测试
    print("=== 中文查询桥接测试 ===\n")
    for cn_query in [
        "XEN197X 怎么配置串口通信",
        "1900 蓝牙配对设置",
        "恢复出厂设置 操作步骤",
    ]:
        print(f"🔍 '{cn_query}':")
        results = ft.search_cn(cn_query, top_k=3)
        for r in results:
            kw = r.get('en_keywords', '')
            print(f"  [{r['product']}] Ch {r['chapter'][:30]} p{r['page']} "
                  f"s={r['score']:.3f}")
            print(f"    en={kw}")
            print(f"    {r['text'][:100]}")
        if not results:
            print(f"  (no results)")
        print()
