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
                  f"s={r['score']:.3f}: {r['text'][:100]}")
        if not results:
            print(f"  (no results)")
        print()
