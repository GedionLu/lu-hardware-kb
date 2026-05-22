# Unified Output Schema — DOCX & PDF Pipelines

两条管线（docx_pipeline / pdf_pipeline）最终都输出此格式，供 chatbot 查询引擎消费。

## Element Types

```json
{
  "type": "text | barcode | image | step",
  "content": "string (text 类型时)",
  "image_path": "relative/path/to/image.png",
  "bbox": [x0, y0, x1, y1],
  "confidence": 0.95,
  "metadata": {}
}
```

### text
纯文本块，来自 PDF 原生文本层或 DOCX 段落。
```json
{"type": "text", "content": "2. USB 虚拟串口设置", "bbox": [100, 200, 400, 230]}
```

### barcode
条码/qrcode 图片，含解码值和标注文字。
```json
{
  "type": "barcode",
  "barcode_id": "bc_page8_01",
  "barcode_raw_value": "~DEFALT.",
  "image_path": "images/bc_page8_01.png",
  "ocr_label": "Standard Product Defaults",
  "class": "barcode",
  "zbar_type": "CODE128",
  "confidence": 0.92,
  "bbox": [120, 350, 380, 400]
}
```

### image
非条码的插图（产品图、连接图、表格截图）。
```json
{
  "type": "image",
  "image_path": "images/fig_page5_01.png",
  "caption": "接口连接示意图",
  "bbox": [50, 500, 550, 700]
}
```

### step
操作步骤，含文字和关联配置码。
```json
{
  "type": "step",
  "step_number": 1,
  "content": "扫描以下条码进入设置模式",
  "image_path": "images/bc_page12_01.png",
  "barcode_raw_value": "%%ENTER_SETUP"
}
```

## Page-Level Wrapper

```json
{
  "page": 8,
  "source": "pdf_pipeline",
  "width": 612.0,
  "height": 792.0,
  "is_scanned": false,
  "columns": 1,
  "element_count": 10,
  "elements": [...]
}
```

## Compatibility

| 字段 | pdf_pipeline V7.2 | docx_pipeline v2.2 | 状态 |
|---|---|---|---|
| `type: text` | ✅ | ✅ | 统一 |
| `type: barcode` | ✅ | ❌ (待对接) | pdf 先出 |
| `type: image` | ❌ (待加) | ✅ | docx 先出 |
| `type: step` | ❌ | ✅ | docx 先出 |
| `bbox` | ✅ | ✅ | 统一 |
| `confidence` | ✅ | ✅ | 统一 |
