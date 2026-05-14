# 硬件客服助手 (AI_KB_ChatBot)

自研工业硬件智能客服系统，面向扫描枪/固定式扫码器/打印机产品线。
支持型号识别、意图分类、混合检索（语义+关键词）、配置码图片输出。

## 项目结构

```
AI_KB_ChatBot/
├── src/                    # 核心引擎
├── static/                 # Web 前端
├── data/                   # 知识库数据
├── pipeline/               # 数据处理管线
├── scripts/                # 辅助工具
├── eval/                   # 质量评估
├── deploy/                 # 部署配置
├── intents.yaml            # 意图定义
├── TODO.md                 # 待办
└── README.md               # 本文件
```

## 架构

```
Browser
  │ GET  /           → static/index.html
  │ POST /ask
  ▼
src/chatbot_server.py   (生产级 · 并发 · GZIP · 限流)
  │ import
  ▼
src/query.py            (查询引擎 v3)
  ├── 型号识别 · 意图识别 · 步骤匹配
  ├── src/retriever.py  (Qdrant + BM25 + RRF)
  │     └── embed_server:8190 (bge-small-zh-v1.5)
  └── 图片输出: data/image_index.json → /img/ URL
```

## 快速开始

```bash
# 依赖
pip install jieba rank_bm25 pyyaml requests

# 嵌入服务 (conda Python 3.13)
~/miniconda3/bin/pip install fastembed
nohup ~/miniconda3/bin/python src/embed_server.py &

# Qdrant
docker run -d -p 6333:6333 qdrant/qdrant:latest

# 启动
cd src && python3 chatbot_server.py

# 访问 → http://localhost:10054
```

## 服务进程

| 进程 | 端口 | 运行时 |
|---|---|---|
| src/chatbot_server.py | 10054 | Python 3.6 |
| src/embed_server.py | 8190 | conda Python 3.13 |
| Qdrant | 6333 | Docker |

## API

| Method | Path | 说明 |
|---|---|---|
| `GET` | `/` | 前端页面 |
| `GET` | `/health` | 健康检查 |
| `POST` | `/ask` | 查询 |
| `GET` | `/img/<category>/<filename>` | 图片 |

## 技术栈

| 层 | 技术 |
|---|---|
| Web | Python BaseHTTPServer + ThreadingMixIn |
| 嵌入 | fastembed (ONNX) + bge-small-zh-v1.5 |
| 向量 | Qdrant (512维 Cosine) |
| 关键词 | jieba + rank_bm25 (BM25) |
| 融合 | Reciprocal Rank Fusion (RRF) |
| LLM | DeepSeek API (v4-flash) |
| 图片 | HTTP URL + ETag + Cache-Control |
