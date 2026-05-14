# TODO

## Pending
- [ ] **步骤质量校验** — scripts/validate_steps.py（规则扫描 V1-V3 + DeepSeek 深度校验 V4-V6），解决 51/145 组 subcategory 缺失等问题。等待确认后实现。

## Completed
- [x] 图片不显示 → image_index.json 补全 image_url
- [x] 向量检索不工作 → embed_server.py + fastembed + 重建 Qdrant
- [x] retriever.py EmbeddingService HTTP 化
- [x] README + git push
