# pipeline/ — 数据处理管线

将原始 DOCX 文件（/tmp/KnowledgeBase/）处理为结构化数据。

## 管线流程

```
run.py (总入口)
  │
  ├── extract.py        Stage 1: DOCX → 图片提取 + 元信息
  │    输入: /tmp/KnowledgeBase/*.docx
  │    输出: /tmp/kb-images/raw/ + data/image_metadata.json
  │
  ├── classify.py       Stage 2: 图片分类
  │    输入: data/image_metadata.json
  │    输出: data/image_classification.json
  │    策略: 规则 (529 张) + DeepSeek 推理 (82 张模糊)
  │
  ├── associate.py      Stage 3: 图片 ↔ 步骤关联
  │    输入: image_classification + 文档元信息
  │    输出: data/doc_model_map.json + step_groups
  │
  └── vlm_analyze.py / vlm_classify.py
       可选: VLM (Qwen-VL) 视觉分析
```

## 文件清单

| 文件 | 功能 |
|---|---|
| `run.py` | 管线总入口，`python3 run.py` |
| `extract.py` | DOCX 图片提取（zipfile + XML 解析） |
| `classify.py` | 图片分类（规则 + DeepSeek） |
| `associate.py` | 图片 ↔ 文档 ↔ 型号关联 |
| `merge_manual.py` | 手动合并辅助 |
| `vlm_analyze.py` | VLM 视觉分析 |
| `vlm_classify.py` | VLM 图片分类 |

## 用法

```bash
# 全量运行
python3 pipeline/run.py

# 只运行分类
python3 pipeline/run.py --stage=classify

# 跳过提取
python3 pipeline/run.py --skip-extract
```
