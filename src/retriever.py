#!/usr/bin/env python3
"""
混合检索引擎: Vector(Qdrant+bge-small-zh) + BM25(jieba) → RRF 重排序

用法:
  from retriever import HybridRetriever
  hr = HybridRetriever()
  results = hr.search("HH760 扫了一下电脑上什么都没有")
"""

import json
import os
import re
import sys
import time
import urllib.parse
import uuid
import hashlib
from collections import defaultdict

# HuggingFace 镜像（国内加速）
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

import requests

# ── 配置 ──
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "hardware_kb"
EMBEDDING_DIM = 512  # bge-small-zh-v1.5 → 512 dim
BGE_MODEL = "BAAI/bge-small-zh-v1.5"

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
INDEX_PATH = os.path.join(ROOT_DIR, "data/image_index.json")
GROUPS_PATH = os.path.join(ROOT_DIR, "data/image_groups.json")

# ── BM25 分词 ──
try:
    import jieba
    HAS_JIEBA = True
except ImportError:
    HAS_JIEBA = False

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False


def tokenize(text):
    """jieba 分词 → token 列表"""
    if not text:
        return []
    if HAS_JIEBA:
        tokens = list(jieba.cut(text))
    else:
        # fallback: 按字 + 英文词切分
        tokens = re.findall(r'[\u4e00-\u9fff]|[a-zA-Z0-9]+', text)
    return [t.strip().lower() for t in tokens if t.strip()]


class DocIndex:
    """文档索引: 管理 BM25 语料 + Qdrant 向量"""
    
    def __init__(self, groups_path=GROUPS_PATH):
        self.groups = self._load_groups(groups_path)
        self.docs = []        # [{id, text, group}]
        self.corpus = []      # [token_list, ...] for BM25
        self.bm25 = None
        
        self._build_docs()
    
    def _load_groups(self, path):
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    
    def _build_docs(self):
        """从 groups 构建文档列表"""
        for g in self.groups:
            gid = g.get('group_id', '')
            doc_name = g.get('doc_name', '')
            steps_text = []
            for s in g.get('steps', []):
                ctx = (s.get('context_text') or '').strip()
                sub = s.get('subcategory') or ''
                steps_text.append(f"[{sub}] {ctx}" if sub else ctx)
            
            # 文档文本 = 文档名 + 步骤上下文
            text = doc_name + ' ' + ' '.join(steps_text)
            
            self.docs.append({
                'id': gid,
                'text': text[:2000],  # 截断长文本
                'doc_name': doc_name,
                'group': g,
            })
            self.corpus.append(tokenize(text[:2000]))
    
    def build_bm25(self):
        """构建 BM25 索引"""
        if HAS_BM25 and self.corpus:
            self.bm25 = BM25Okapi(self.corpus)
    
    def bm25_search(self, query_tokens, top_k=20):
        """BM25 关键词搜索"""
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(query_tokens)
        ranked = sorted(enumerate(scores), key=lambda x: -x[1])
        return [(self.docs[i], float(score)) for i, score in ranked if score > 0][:top_k]


class QdrantStore:
    """Qdrant 向量存储 (HTTP REST API)"""
    
    def __init__(self, url=QDRANT_URL, collection=COLLECTION_NAME, dim=EMBEDDING_DIM):
        self.url = url.rstrip('/')
        self.collection = collection
        self.dim = dim
        self._ensure_collection()
    
    def _ensure_collection(self):
        """确保集合存在"""
        resp = requests.get(f"{self.url}/collections/{self.collection}")
        if resp.status_code == 404:
            requests.put(
                f"{self.url}/collections/{self.collection}",
                json={
                    "vectors": {"size": self.dim, "distance": "Cosine"},
                    "optimizers_config": {"default_segment_number": 2},
                }
            )
    
    def upsert(self, points):
        """批量写入向量点
        points: [{"id": str, "vector": list[float], "payload": dict}, ...]
        """
        # Qdrant 1.x requires UUIDs or unsigned ints as point IDs
        uuid_points = []
        for p in points:
            doc_id = p['id']
            uid = str(uuid.UUID(hashlib.md5(doc_id.encode()).hexdigest()))
            uuid_points.append({
                "id": uid,
                "vector": p['vector'],
                "payload": {"doc_id": doc_id, **p.get('payload', {})},
            })
        requests.put(
            f"{self.url}/collections/{self.collection}/points?wait=true",
            json={"points": uuid_points}
        )
    
    def search(self, vector, top_k=20, score_threshold=0.3):
        """向量相似搜索"""
        resp = requests.post(
            f"{self.url}/collections/{self.collection}/points/search",
            json={
                "vector": vector,
                "limit": top_k,
                "score_threshold": score_threshold,
                "with_payload": True,
            }
        )
        if resp.status_code != 200:
            return []
        # Map UUID back to doc_id
        hits = resp.json().get("result", [])
        for h in hits:
            if 'payload' in h and 'doc_id' in h['payload']:
                h['id'] = h['payload']['doc_id']
        return hits
    
    def count(self):
        resp = requests.post(
            f"{self.url}/collections/{self.collection}/points/count",
            json={}
        )
        if resp.status_code == 200:
            return resp.json()["result"].get("count", 0)
        return 0


EMBED_SERVER = "http://localhost:8190"

class EmbeddingService:
    """bge-small-zh-v1.5 嵌入服务 — 通过 HTTP 调 embed_server.py"""
    
    def __init__(self, server_url=EMBED_SERVER):
        self.server_url = server_url.rstrip('/')
        self._ok = False
        self._check()
    
    def _check(self):
        try:
            resp = requests.get(f"{self.server_url}/health", timeout=5)
            if resp.status_code == 200:
                self._ok = True
                print(f"  [Embed] 嵌入服务就绪: {self.server_url}")
                return
        except:
            pass
        print(f"  [Embed] WARNING: 嵌入服务不可用 ({self.server_url})，向量检索将退化为 BM25-only")
    
    def encode(self, texts):
        """文本 → 向量 (HTTP 调用嵌入服务)"""
        if not self._ok:
            print("  [Embed] WARNING: 无嵌入模型，返回零向量")
            return [[0.0] * EMBEDDING_DIM for _ in texts]
        try:
            resp = requests.post(
                f"{self.server_url}/v1/embeddings",
                json={"input": texts},
                timeout=30,
            )
            if resp.status_code == 200:
                data = resp.json()["data"]
                data.sort(key=lambda x: x["index"])
                return [d["embedding"] for d in data]
        except Exception as e:
            print(f"  [Embed] ERROR: {e}")
        print("  [Embed] WARNING: 嵌入服务调用失败，返回零向量")
        return [[0.0] * EMBEDDING_DIM for _ in texts]


class HybridRetriever:
    """混合检索: Vector + BM25 → RRF 融合"""
    
    def __init__(self, index_path=INDEX_PATH, groups_path=GROUPS_PATH):
        self.doc_index = DocIndex(groups_path)
        self.qdrant = QdrantStore()
        self.embedder = EmbeddingService()
        
        # 加载 image_index (图片URL映射)
        self.image_index = {}
        self._load_image_index(index_path)
        
        # 统计
        print(f"  [Hybrid] 文档数: {len(self.doc_index.docs)}")
        print(f"  [Hybrid] Qdrant 点数: {self.qdrant.count()}")
    
    def _load_image_index(self, path):
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                entries = json.load(f)
            for e in entries:
                fn = e.get('file_name', '')
                url = e.get('image_url', '')
                if fn and url:
                    self.image_index[fn] = url
    
    def index_all(self, force=False):
        """全量索引: 对文档做 embedding → Qdrant"""
        if self.qdrant.count() > 0 and not force:
            print(f"  [Index] 已有 {self.qdrant.count()} 个向量，跳过")
            self.doc_index.build_bm25()
            return
        
        print(f"  [Index] 开始向量化 {len(self.doc_index.docs)} 份文档...")
        
        # 批量 embedding
        BATCH_SIZE = 32
        total = 0
        
        for i in range(0, len(self.doc_index.docs), BATCH_SIZE):
            batch = self.doc_index.docs[i:i + BATCH_SIZE]
            texts = [d['text'][:512] for d in batch]  # 截断到 512 chars
            vectors = self.embedder.encode(texts)
            
            points = []
            for d, vec in zip(batch, vectors):
                points.append({
                    "id": d['id'],
                    "vector": vec,
                    "payload": {"doc_name": d['doc_name'], "text": d['text'][:200]},
                })
            
            self.qdrant.upsert(points)
            total += len(points)
            print(f"  [Index] {total}/{len(self.doc_index.docs)}")
        
        # 构建 BM25
        self.doc_index.build_bm25()
        print(f"  [Index] 完成: {total} 向量 + BM25 就绪")
    
    def search(self, query, top_k=5):
        """
        混合搜索:
        1. bge-small-zh 向量检索 (语义相似)
        2. BM25 关键词匹配 (精确匹配)
        3. RRF 融合排序
        """
        query_tokens = tokenize(query)
        
        # ── 向量检索 ──
        vec_start = time.time()
        query_vec = self.embedder.encode([query])[0]
        vec_hits = self.qdrant.search(query_vec, top_k=20)
        vec_time = time.time() - vec_start
        
        # ── BM25 检索 ──
        bm25_start = time.time()
        bm25_hits = self.doc_index.bm25_search(query_tokens, top_k=20)
        bm25_time = time.time() - bm25_start
        
        # ── RRF 融合 ──
        merged = self._rrf_fusion(vec_hits, bm25_hits, k=60)
        
        # 取 top_k，附带原始 hits
        top_results = merged[:top_k]
        
        # 构建返回结果
        results = []
        for doc, score in top_results:
            group = doc['group']
            steps = group.get('steps', [])
            # 添加图片 URL
            steps_with_images = []
            for s in steps:
                step = dict(s)
                fn = s.get('file_name', '')
                if fn in self.image_index:
                    step['image_url'] = self.image_index[fn]
                steps_with_images.append(step)
            
            results.append({
                'group_id': doc['id'],
                'doc_name': doc['doc_name'],
                'score': round(score, 4),
                'steps': steps_with_images,
                'applicable_models': group.get('applicable_models', []),
                'total_config_codes': group.get('total_config_codes', 0),
            })
        
        return results
    
    def _rrf_fusion(self, vec_hits, bm25_hits, k=60):
        """
        Reciprocal Rank Fusion
        score(doc) = sum(1 / (k + rank_i)) across all rankers
        """
        scores = defaultdict(float)
        doc_map = {}  # id → doc
        
        # 向量排名
        for rank, hit in enumerate(vec_hits):
            hit_id = hit.get('id', '')
            scores[hit_id] += 1.0 / (k + rank + 1)
            # 从 doc_index 找对应 doc
            for d in self.doc_index.docs:
                if d['id'] == hit_id:
                    doc_map[hit_id] = d
                    break
        
        # BM25 排名
        for rank, (doc, bm_score) in enumerate(bm25_hits):
            doc_id = doc['id']
            scores[doc_id] += 1.0 / (k + rank + 1)
            if doc_id not in doc_map:
                doc_map[doc_id] = doc
        
        # 排序
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return [(doc_map.get(doc_id), score) for doc_id, score in ranked if doc_id in doc_map]


# ── 便捷工厂 ──
_retriever_instance = None

def get_retriever(force_reindex=False):
    global _retriever_instance
    if _retriever_instance is None:
        print("[Retriever] 初始化混合检索引擎...")
        _retriever_instance = HybridRetriever()
        _retriever_instance.index_all(force=force_reindex)
    return _retriever_instance


if __name__ == '__main__':
    hr = get_retriever(force_reindex=True)
    
    queries = [
        "HH760 扫了一下电脑上什么都没有",
        "1900 怎么加回车换行",
        "OH430 蓝牙配对",
        "7680g 扫描没输出",
    ]
    
    for q in queries:
        print(f"\n{'='*60}")
        print(f"查询: {q}")
        results = hr.search(q, top_k=3)
        for i, r in enumerate(results):
            print(f"  [{i+1}] {r['doc_name'][:60]} (score={r['score']})")
            for s in r['steps'][:3]:
                ctx = (s.get('context_text') or '')[:50]
                print(f"       [{s.get('subcategory','?')}] {ctx}")
