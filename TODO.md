# TODO

## Pending
- [ ] ECS 部署同步 (git pull + 重建 Qdrant 索引) — XEN197X 数据 + image_groups 修复
- [ ] 文档去重 + 跨文档合并
- [ ] 评测自动化 (eval/ CI)
- [ ] PDF 结构化文本提取 (表格 + 步骤)
- [ ] 知识库自学习机制 (LLM 接管 + 回流)

## Completed
- [x] 步骤质量校验 — 504/504 组级字段已填充 (168 组 × 3 字段), 691 步 → 16 步 None (均为样本码/参考图)
- [x] 图片不显示 → image_index.json 补全 image_url
- [x] 向量检索不工作 → embed_server.py + fastembed + 重建 Qdrant
- [x] retriever.py EmbeddingService HTTP 化
- [x] README + git push
