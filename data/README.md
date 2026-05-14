# data/ — 知识库数据

> 由 `pipeline/` 管线生成，`src/query.py` 读取。
> 图片原始文件存储在 `/tmp/kb-images/`（受 `.gitignore` 排除）。

## 文件清单

### image_index.json (928 条)
图片索引，每条包含 `file_name`、`image_url`、`category`、`context_text`。

**被引用：** `retriever.py`（图片 URL 映射）、`query.py`（`[图片: ...]` 标记生成）

### image_groups.json (145 组)
步骤组 = 文档 → 操作步骤 + 关联图片。

**字段：** `group_id`, `doc_name`, `source_doc_rel`, `steps[{step_order, context_text, file_name, subcategory}]`, `applicable_models`, `total_config_codes`

**被引用：** `query.py`（规则匹配）、`retriever.py`（文档索引）

### image_classification.json (928 条)
图片分类结果，由 `pipeline/classify.py` 生成。

**字段：** `image_id`, `file_name`, `category`（config_code/diagram/screenshot/product_photo）

### config_codes.json (152 条)
配置码定义。

### steps.json (185 条)
原子步骤定义，每个步骤独立为一条记录。

### step_groups.json (41 组)
步骤组定义（作为 image_groups.json 的补充）。

### doc_model_map.json
文档 → 型号映射，由管线 `associate.py` 生成。

### build_kb.py
知识库构建工具脚本。

### config_search.py
配置码搜索工具。
