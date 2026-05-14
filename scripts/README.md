# scripts/ — 辅助工具

## 文件清单

### llm_structurer.py
LLM 结构化处理器 —— 用 DeepSeek API 将 DOCX 原始内容转为知识库条目。

**输出：** qa_pairs / config_codes / product_specs / operation_steps / classifiers

### docx_processor.py
DOCX 文件内容提取器 —— 段落 + 表格 + 图片原始信息。

### llm_fill_tree.py
LLM 产品型号树填充 —— 从文档中识别产品层级关系。

### doc_image_map.py
文档图片映射工具 —— 建立文档 → 图片引用关系。

### add_llm_eval.py
LLM 评估辅助 —— 在评估数据中补充 LLM 标注。

### update_intent_map.py
意图映射更新 —— 从现有数据中提取关键词补全 `intents.yaml`。

## 用法

```bash
# 结构化一个文档
python3 scripts/llm_structurer.py input.json

# 更新意图配置
python3 scripts/update_intent_map.py
```
