# TODO

## Pending
- [ ] ECS 部署同步 (git pull + 重建 Qdrant 索引 + deploy dedup image_groups)
- [ ] PDF 结构化文本提取 (表格 + 操作步骤 — MinerU/Unstructured)
- [ ] 知识库自学习机制 (LLM 接管 + 回流)
- [ ] config_codes 图片补全 (XEN197X 等 image_url 为空)

## Completed
- [x] 步骤质量校验 — 504/504 组级字段已填充 (168 组 × 3 字段), 691 步 → 16 步 None (均为样本码/参考图)
- [x] 文档去重合并 — dedup_merge.py: 168 组 → 100 组, 移除 68 个同 model+subcategory 重复
- [x] 评测自动化 — eval_auto.py + run_eval.sh: 31 条测试集, 自动评分 + CI 模式
- [x] 图片不显示 → image_index.json 补全 image_url
- [x] 向量检索不工作 → embed_server.py + fastembed + 重建 Qdrant
- [x] retriever.py EmbeddingService HTTP 化
- [x] README + git push
