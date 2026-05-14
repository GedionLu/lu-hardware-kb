#!/usr/bin/env python3
"""bge-small-zh-v1.5 嵌入微服务 — 端口 8190

依赖: conda Python + fastembed
启动: ~/miniconda3/bin/python embed_server.py
"""

import json
import os
import sys
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

MODEL_NAME = 'BAAI/bge-small-zh-v1.5'
BATCH_SIZE = 32

# ── 模型加载 ──
print(f'[embed_server] 加载模型 {MODEL_NAME} ...', file=sys.stderr, flush=True)
from fastembed import TextEmbedding
model = TextEmbedding(model_name=MODEL_NAME)
dims = model.embedding_size if hasattr(model, 'embedding_size') else None
print(f'[embed_server] 模型就绪, dims={dims}', file=sys.stderr, flush=True)


def encode(texts):
    """文本列表 → 向量列表"""
    vectors = []
    for v in model.embed(texts, batch_size=BATCH_SIZE):
        vectors.append(v.tolist())
    return vectors


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            self._json({'status': 'ok', 'model': MODEL_NAME})
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path != '/v1/embeddings':
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length).decode('utf-8')
        try:
            req = json.loads(body)
            texts = req.get('input', [])
            if isinstance(texts, str):
                texts = [texts]
        except:
            texts = [body]

        vectors = encode(texts)
        resp = {'data': [{'embedding': v, 'index': i} for i, v in enumerate(vectors)]}
        self._json(resp)

    def _json(self, data):
        body = json.dumps(data, ensure_ascii=False)
        self.send_response(200)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.end_headers()
        self.wfile.write(body.encode('utf-8'))

    def log_message(self, *a):
        pass


if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8190
    print(f'[embed_server] 启动 http://0.0.0.0:{port}', file=sys.stderr, flush=True)
    HTTPServer(('0.0.0.0', port), H).serve_forever()
