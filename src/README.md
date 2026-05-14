# src/ — 核心引擎

## 文件清单

### chatbot_server.py
生产级 Web 服务，端口 10054。

**特性：** ThreadingMixIn 并发 / GZIP 压缩 / IP 限流 30/min / ETag 图片缓存 / 结构化日志 / 健康检查

**用法：** `python3 chatbot_server.py`

**入口点：**
- `GET /` → 返回 `../static/index.html`
- `GET /health` → `{"status":"ok","qdrant":144,"embed":"ok"}`
- `GET /img/<cat>/<file>` → 图片文件，含 ETag 缓存
- `GET /static/<path>` → CSS/JS 静态资源
- `POST /ask {"query":"..."}` → 查询接口（500 字符限制，30s 超时）

---

### query.py
查询引擎 v3，接收用户问题，返回结构化答案。

**职责：**
1. 型号识别 — 正则匹配 30+ 产品线
2. 意图识别 — `intents.yaml` 关键词 → DeepSeek 兜底
3. 步骤匹配 — `data/image_groups.json` 打分排序
4. 混合检索 — `retriever.py`（Qdrant + BM25 → RRF）
5. LLM 排序 — 候选组 ≥2 时 DeepSeek 精排

**核心类：** `QueryEngineV3`

**依赖：**
- `../intents.yaml` — 意图配置
- `../data/image_index.json` — 图片索引
- `../data/image_groups.json` — 步骤组
- `retriever.py` — 混合检索
- `deepseek API` — LLM 兜底

---

### retriever.py
混合检索引擎 —— 向量语义 + 关键词精确 → RRF 融合。

**组件：**
- `EmbeddingService` — HTTP 调 `embed_server:8190`
- `QdrantStore` — Qdrant REST API (collection: hardware_kb)
- `DocIndex` — 文档索引 + BM25 (jieba + rank_bm25)
- `HybridRetriever` — RRF 融合 (k=60)

**依赖：** `jieba`, `rank_bm25`, `requests`, `../data/image_index.json`, `../data/image_groups.json`

---

### embed_server.py
bge-small-zh-v1.5 嵌入微服务，端口 8190。

**推理后端：** fastembed (ONNX Runtime)

**API：**
- `GET /health` → `{"status":"ok","model":"BAAI/bge-small-zh-v1.5"}`
- `POST /v1/embeddings {"input":["text"]}` → 512维向量

**启动：** `~/miniconda3/bin/python embed_server.py`
