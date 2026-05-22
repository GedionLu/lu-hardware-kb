# PDF Barcode Extraction Pipeline

YOLOv8s 条码/qrcode 检测 + pyzbar 解码 + 空间排序 — 独立可测试。

## 依赖

```bash
pip install ultralytics PyMuPDF Pillow pyzbar numpy
# Linux: apt install libzbar0 / yum install zbar-devel
# macOS: brew install zbar
```

## 模型

`models/yolov8s-barcode-detection.pt` (Piero2411/YOLOV8s-Barcode-Detection, 21MB)

## 用法

```bash
cd ai-kb-chatbot

# 基础用法
python3 pdf_pipeline/extract_yolo_v7.2.py \
  --pdf /path/to/document.pdf \
  --out /tmp/output/

# 完整参数
python3 pdf_pipeline/extract_yolo_v7.2.py \
  --pdf /path/to/document.pdf \
  --out /tmp/output/ \
  --dpi 300 \
  --conf 0.25 \
  --skip 1
```

## 输出

```
/tmp/output/
├── index.json          # 结构化输出 (符合 output_schema.md)
└── images/             # 条码裁剪图
    ├── bc_page8_01.png
    ├── bc_page8_02.png
    └── ...
```

## 特性

| 功能 | 说明 |
|---|---|
| YOLOv8s 检测 | 全页 + 嵌入图两阶段检测 |
| pyzbar 解码 | 条码原始值写入元数据 |
| 分栏检测 | 自动识别双栏排版并正确排序 |
| 重叠过滤 | 剔除条码自带 HRI 文字 |
| 扫描件分流 | 自动检测无文本层的扫描页 |
| 空间排序 | 文字块+条码块按阅读顺序排列 |

## 版本

| 文件 | 版本 | 说明 |
|---|---|---|
| `extract_yolo_v7.2.py` | V7.2 | 当前主力（空间排序+三补丁） |
| `extract_yolo_v7.py` | V7.1 | 稳定版（YOLO+pyzbar，无排序） |
