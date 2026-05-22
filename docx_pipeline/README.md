# DOCX Pipeline — 🔒 FROZEN (暂不激活)

将原始 DOCX 文件（/tmp/KnowledgeBase/）处理为结构化数据。

## 状态

**当前不启用。** 该管线处理 DOCX 格式的产品手册和说明书。
激活时请将其改名为 `pipeline/` 并恢复 CI/入口配置。

## 管线流程

```
run.py (总入口)
 ├── extract.py       Stage 1: DOCX → 图片提取 + 元信息
 │    输入: /tmp/KnowledgeBase/*.docx
 │    输出: /tmp/kb-images/
 ├── classify.py      Stage 2: 图片分类 (配置码/图表/产品图)
 ├── associate.py     Stage 3: 图片与文字关联
 └── vlm_analyze.py   Stage 4: VLM 分析 (可选)
```

## 产出

对接统一输出格式 `output_schema.md`，输出到 `data/image_groups.json`。

## 激活步骤

1. `mv docx_pipeline pipeline`
2. 恢复 `scripts/` 中的 DOCX 处理脚本引用
3. 运行 `pipeline/run.py`
