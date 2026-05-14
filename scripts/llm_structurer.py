#!/usr/bin/env python3
"""
LLM 结构化处理器
用 DeepSeek API 将 docx_processor 提取的原始内容转成知识库条目。

输入: docx_processor 输出的 .json (paragraphs + tables + images)
输出: 知识库结构化 JSON，包含:
  - qa_pairs       → FAQ 问答对
  - config_codes   → 配置码（含关联图片）
  - product_specs  → 产品规格
  - operation_steps → 操作步骤
  - classifiers    → 分类标签

用法:
  python3.8 scripts/llm_structurer.py <input.json> [--output OUTPUT] [--no-exec]

环境变量:
  DEEPSEEK_API_KEY — 默认读取 OpenClaw 配置
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

# ── DeepSeek API client (lightweight, no extra deps) ──
try:
    import urllib.request as url_request
    import urllib.error
    HAS_REQUESTS = False
except ImportError:
    HAS_REQUESTS = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# ── Config ──
DEEPSEEK_BASE = "https://api.deepseek.com"
MODEL = "deepseek-chat"
TEMPERATURE = 0.1
MAX_TOKENS_PER_CALL = 4096
MAX_INPUT_TOKENS = 32000  # stay well under 128k context window

# ── Prompt templates ──

SYSTEM_PROMPT = """你是一个工业硬件产品的知识库构建专家。你的任务是将产品的技术文档内容转化为结构化的知识库数据。

你需要按以下规则处理：

## 1. QA 问答对 (qa_pairs)
从文档中提取常见问答。每对包含:
- question: 用户可能问的问题（自然语言）
- answer: 精确答案（附文档引用位置）
- category: 问题类别 (usage / config / troubleshoot / spec / install / misc)
- confidence: 确定性 (exact / high / medium)

## 2. 配置码 (config_codes)
如果文档中出现配置码/条码截图或配置参数，提取:
- code_name: 配置码名称/编号
- description: 配置作用描述
- parameters: 配置参数列表 [{key, value, description}]
- image_ref: 关联图片文件名（如果有）
- applicable_models: 适用型号

## 3. 产品规格 (product_specs)
从表格或段落中提取结构化规格:
- category: 规格类别 (physical / electrical / optical / environment / interface)
- specs: [{name, value, unit, note}]

## 4. 操作步骤 (operation_steps)
从操作指南中提取步骤序列，保持顺序:
- step_number: 步骤编号
- title: 步骤标题
- description: 详细说明
- image_refs: 关联图片
- warnings: 注意事项

## 5. 分类标签 (classifiers)
- product_name: 产品名称
- model: 型号
- doc_type: manual / spec / config_guide / install_guide / faq
- keywords: 关键词列表

## 输出格式
必须输出纯 JSON 对象，不要包含 markdown 代码块包裹。如果某类没有内容，用空数组。"""


def _extract_images_text(images: list, base_dir: str = "") -> str:
    """将图片信息转为可读文本"""
    lines = []
    for img in images:
        parts = []
        parts.append(f"  图片: {img.get('saved_path', img.get('media_path', 'unknown'))}")
        if img.get('width'):
            parts.append(f"  尺寸: {img['width']}×{img['height']}")
        if img.get('alt_text'):
            parts.append(f"  描述: {img['alt_text']}")
        lines.append(" | ".join(parts))
    return "\n".join(lines)


def _build_document_section(data: dict) -> str:
    """将 JSON 数据转为 LLM 可读的文本段落"""
    blocks = []
    metadata = data.get("metadata", {})

    # 文件信息
    blocks.append(f"## 文档: {metadata.get('file_name', 'unknown')}")
    if metadata.get("title"):
        blocks.append(f"标题: {metadata['title']}")
    if metadata.get("author"):
        blocks.append(f"作者: {metadata['author']}")
    blocks.append(f"段落数: {metadata.get('paragraphs', 0)}, 表格数: {metadata.get('tables', 0)}, 图片数: {len(data.get('images', []))}")
    blocks.append("")

    # 段落（含标题/列表标记）
    blocks.append("### 文档内容")
    for p in data.get("paragraphs", []):
        text = p["text"]
        style = p.get("style", "Normal")
        prefix = ""
        if "Heading" in style:
            level = style.replace("Heading ", "")
            prefix = "#" * int(level) + " " if level.isdigit() else "## "
        elif p.get("is_list"):
            prefix = "- "
        else:
            prefix = ""
        blocks.append(f"{prefix}{text}")

    # 表格
    for t in data.get("tables", []):
        blocks.append("")
        blocks.append("### 表格")
        # header
        hdr = t.get("header", [])
        if hdr:
            blocks.append("| " + " | ".join(hdr) + " |")
            blocks.append("| " + " | ".join(["---"] * len(hdr)) + " |")
        for row in t.get("data", []):
            blocks.append("| " + " | ".join(row) + " |")
        blocks.append("")

    # 图片索引
    images = data.get("images", [])
    if images:
        blocks.append("### 附件图片")
        blocks.append(_extract_images_text(images))
        blocks.append("")

    return "\n".join(blocks)


def _estimate_tokens(text: str) -> int:
    """粗略估算 token 数（中文约 1.5 字/token，英文约 4 字符/token）"""
    chinese = len(re.findall(r'[\u4e00-\u9fff]', text))
    english = len(re.sub(r'[\u4e00-\u9fff\s]', '', text))
    return int(chinese * 1.5 + english / 4 + text.count('\n') * 0.5)


def _chunk_document(full_text: str, max_tokens: int = MAX_INPUT_TOKENS) -> list:
    """将内容分块，每块不超过 max_tokens"""
    estimated = _estimate_tokens(full_text)
    if estimated <= max_tokens:
        return [full_text]

    # 按段落拆分
    paragraphs = full_text.split("\n")
    chunks = []
    current_chunk = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = _estimate_tokens(para) + 1  # +1 for newline
        if current_tokens + para_tokens > max_tokens and current_chunk:
            chunks.append("\n".join(current_chunk))
            current_chunk = [para]
            current_tokens = para_tokens
        else:
            current_chunk.append(para)
            current_tokens += para_tokens

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks


def _call_deepseek(messages: list, api_key: str, retry: int = 3) -> dict:
    """调用 DeepSeek API"""
    url = f"{DEEPSEEK_BASE}/v1/chat/completions"
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS_PER_CALL,
        "response_format": {"type": "json_object"},
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    for attempt in range(retry):
        try:
            if HAS_REQUESTS:
                resp = requests.post(url, json=payload, headers=headers, timeout=120)
                resp.raise_for_status()
                return resp.json()
            else:
                body = json.dumps(payload).encode("utf-8")
                req = url_request.Request(url, data=body, headers=headers, method="POST")
                with url_request.urlopen(req, timeout=120) as r:
                    return json.loads(r.read().decode())
        except Exception as e:
            print(f"  API 调用失败 (attempt {attempt+1}/{retry}): {e}", file=sys.stderr)
            if attempt < retry - 1:
                wait = 2 ** attempt
                print(f"  {wait}s 后重试...", file=sys.stderr)
                time.sleep(wait)
            else:
                raise


def _parse_llm_response(content: str) -> dict:
    """解析 LLM 返回的 JSON"""
    # 尝试直接解析
    content = content.strip()
    if content.startswith("```"):
        # 移除可能的 markdown 包裹
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
    return json.loads(content)


def _merge_results(results: list) -> dict:
    """合并多个 chunk 的处理结果"""
    merged = {
        "qa_pairs": [],
        "config_codes": [],
        "product_specs": [],
        "operation_steps": [],
        "classifiers": {},
    }

    for r in results:
        for key in ["qa_pairs", "config_codes", "product_specs", "operation_steps"]:
            if key in r and isinstance(r[key], list):
                merged[key].extend(r[key])
        if "classifiers" in r and isinstance(r["classifiers"], dict):
            # 从后往前覆盖（后面的 chunk 信息更完整）
            merged["classifiers"].update(r["classifiers"])

    # 去重 QA pairs（基于 question 去重）
    seen_qs = set()
    unique_qas = []
    for qa in merged["qa_pairs"]:
        q = qa.get("question", "").strip()
        if q and q not in seen_qs:
            seen_qs.add(q)
            unique_qas.append(qa)
    merged["qa_pairs"] = unique_qas

    # 去重 config codes
    seen_codes = set()
    unique_codes = []
    for cc in merged["config_codes"]:
        c = cc.get("code_name", "").strip()
        if c and c not in seen_codes:
            seen_codes.add(c)
            unique_codes.append(cc)
    merged["config_codes"] = unique_codes

    return merged


def process_document(input_path: str, output_path: str = None, dry_run: bool = False) -> dict:
    """处理单个文档"""
    # 读取输入
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "paragraphs" not in data:
        raise ValueError("输入 JSON 缺少 'paragraphs' 字段 — 请先运行 docx_processor.py")

    file_name = data.get("metadata", {}).get("file_name", Path(input_path).stem)
    print(f"\n{'='*60}")
    print(f"结构化: {file_name}")
    print(f"{'='*60}")

    # 获取 API key
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        # 从 OpenClaw 配置读取
        try:
            with open("/home/admin/.openclaw/openclaw.json") as f:
                cfg = json.load(f)
            api_key = cfg.get("models", {}).get("providers", {}).get("deepseek", {}).get("apiKey", "")
        except Exception:
            pass

    if not api_key:
        print("❌ 未找到 DeepSeek API Key", file=sys.stderr)
        print("   请设置环境变量 DEEPSEEK_API_KEY", file=sys.stderr)
        sys.exit(1)

    # 构建文档文本
    full_text = _build_document_section(data)
    tokens_est = _estimate_tokens(full_text)
    print(f"   内容长度: {len(full_text)} 字符 (~{tokens_est} tokens)")

    # 分块
    chunks = _chunk_document(full_text)
    print(f"   分块: {len(chunks)} 块")

    if dry_run:
        print("   [dry-run 模式，不调用 API]")
        # 输出示例结构
        return {
            "metadata": data.get("metadata", {}),
            "qa_pairs": [],
            "config_codes": [],
            "product_specs": [],
            "operation_steps": [],
            "classifiers": {"product_name": "", "model": "", "doc_type": "manual", "keywords": []},
            "source_file": input_path,
            "_dry_run": True,
        }

    # 逐块调用 LLM
    results = []
    for i, chunk in enumerate(chunks):
        chunk_tokens = _estimate_tokens(chunk)
        print(f"   处理分块 {i+1}/{len(chunks)} (~{chunk_tokens} tokens)...")

        user_prompt = f"""请分析以下文档内容，提取所有结构化的知识信息。

目标是生成立即可导入知识库的 JSON 数据。注意：
- 如果有可配置的参数、码值、选项，提取为 config_codes
- 如果包含步骤流程，提取为 operation_steps（保持顺序）
- 如果看起来是常见问题场景，提取为 qa_pairs
- 如果有关联图片，在对应条目的 image_ref 中填入图片文件名（如 "rId5.png"，不要带路径）

文档内容：
{chunk}"""

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = _call_deepseek(messages, api_key)
            content = response["choices"][0]["message"]["content"]

            # 解析
            result = _parse_llm_response(content)
            result.setdefault("qa_pairs", [])
            result.setdefault("config_codes", [])
            result.setdefault("product_specs", [])
            result.setdefault("operation_steps", [])
            result.setdefault("classifiers", {})

            # 统计
            qa_count = len(result["qa_pairs"])
            cc_count = len(result["config_codes"])
            ps_count = len(result["product_specs"])
            os_count = len(result["operation_steps"])
            print(f"     → QA: {qa_count}, 配置码: {cc_count}, 规格: {ps_count}, 步骤: {os_count}")

            results.append(result)

        except Exception as e:
            print(f"   ❌ 分块 {i+1} 处理失败: {e}", file=sys.stderr)
            # 继续处理后续分块

    # 合并
    final = _merge_results(results)
    final["metadata"] = data.get("metadata", {})
    final["source_file"] = input_path

    # 建立图片路径映射（filename -> 相对路径）
    # 图片保存在 {json_dir}/images/ 目录下
    input_path_obj = Path(input_path)
    input_dir = input_path_obj.parent
    images_path_map = {}
    for img in data.get("images", []):
        saved = img.get("saved_path", "")
        media = img.get("media_path", "")
        basename = os.path.basename(saved or media)
        # 如果 saved_path 已是相对路径，直接拼接 images/
        if saved and not saved.startswith("/"):
            images_path_map[basename] = saved
        else:
            images_path_map[basename] = f"images/{basename}"
    final["_image_map"] = images_path_map

    # ── 后处理：修正 image_ref 路径 + 规范化配置码名称 ──
    # 图片目录在输入 JSON 所在目录的 images/ 下
    input_images_dir = input_dir / "images"
    output_path_obj = Path(output_path) if output_path else input_dir
    output_dir = output_path_obj.parent

    def _normalize_image_ref(ref):
        """统一 image_ref 为 images/filename.png 格式（相对于 JSON 自身）"""
        if not ref or not isinstance(ref, str):
            return None
        ref = ref.strip()
        # 去掉各种路径前缀，只保留文件名
        for prefix in ["/output/", "/input/", "/workspace/", "/"]:
            if prefix == "/":
                # 绝对路径 → 只取文件名
                if ref.startswith("/"):
                    ref = os.path.basename(ref)
            else:
                if ref.startswith(prefix):
                    ref = os.path.basename(ref)
        # 如果已在 images_path_map 中，用映射值
        if ref in images_path_map:
            return images_path_map[ref]
        # 兜底：images/filename
        return f"images/{ref}" if ref else None

    def _normalize_cc_name(code_name, model, existing_names):
        """规范化配置码名称：给常见通用名加型号前缀"""
        common_names = {
            "恢复出厂", "恢复出厂码", "恢复出厂设置", "恢复出厂设置码",
            "测试码", "配对码", "手动模式设置码",
            "回车后缀", "回车(CR)后缀", "CR后缀", "回车",
        }
        if code_name in common_names and model:
            # 加型号前缀，如 "1900-C-恢复出厂"
            model_prefix = model.replace(" ", "")[:15]
            new_name = f"{model_prefix}-{code_name}"
            return new_name
        # 检查是否与已有条目重复
        if code_name in existing_names:
            # 如果已在集合中但不同型号，加后缀
            return code_name
        return code_name

    # 收集已用的配置码名称
    used_cc_names = set()
    for cc in final.get("config_codes", []):
        model = ""
        models = cc.get("applicable_models", [])
        if models:
            model = models[0] if isinstance(models, list) else str(models)
        old_name = cc.get("code_name", "")
        new_name = _normalize_cc_name(old_name, model, used_cc_names)
        cc["code_name"] = new_name
        used_cc_names.add(new_name)
        # 修正 image_ref
        cc["image_ref"] = _normalize_image_ref(cc.get("image_ref"))

    for step in final.get("operation_steps", []):
        refs = step.get("image_refs", [])
        if refs:
            step["image_refs"] = [_normalize_image_ref(r) for r in refs if r]

    # 清理 _image_map（调试用，生产可移除）

    # 统计
    print(f"\n{'='*60}")
    print(f"完成: {file_name}")
    print(f"  QA 问答对:    {len(final['qa_pairs'])}")
    print(f"  配置码条目:   {len(final['config_codes'])}")
    print(f"  产品规格:     {len(final['product_specs'])}")
    print(f"  操作步骤:     {len(final['operation_steps'])}")
    classifiers = final.get("classifiers", {})
    if isinstance(classifiers, dict) and classifiers.get("product_name"):
        print(f"  产品: {classifiers.get('product_name')} ({classifiers.get('model', '')})")
    print(f"{'='*60}")

    # 写入
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final, f, ensure_ascii=False, indent=2)
        print(f"  ✅ 输出 -> {output_path}")

    return final


def batch_process(input_dir: str, output_dir: str = None, dry_run: bool = False) -> list:
    """批量处理目录下所有 docx_processor 输出的 JSON"""
    json_files = sorted(Path(input_dir).glob("**/*.json"))
    # 排除 summary 和已处理的
    json_files = [f for f in json_files
                  if "_batch_summary" not in str(f)
                  and "_structured" not in str(f)
                  and "output" not in str(f.parent.name)]

    if not json_files:
        print(f"未在 {input_dir} 中找到可处理的 JSON 文件")
        return []

    print(f"发现 {len(json_files)} 个 JSON 文件")
    results = []
    for jf in json_files:
        try:
            if output_dir:
                rel = jf.relative_to(input_dir)
                out_path = Path(output_dir) / rel.parent / f"{jf.stem}_structured.json"
            else:
                out_path = jf.parent / f"{jf.stem}_structured.json"

            res = process_document(str(jf), str(out_path), dry_run)
            results.append(res)
        except Exception as e:
            print(f"  FAIL: {jf.name} -> {e}", file=sys.stderr)

    # 汇总
    summary = {
        "total": len(json_files),
        "success": len(results),
        "failed": len(json_files) - len(results),
        "qa_pairs_total": sum(len(r.get("qa_pairs", [])) for r in results),
        "config_codes_total": sum(len(r.get("config_codes", [])) for r in results),
        "operation_steps_total": sum(len(r.get("operation_steps", [])) for r in results),
    }
    summary_file = Path(output_dir or input_dir) / "_structured_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n批量汇总 -> {summary_file}")
    return results


# ── CLI ──
def main():
    parser = argparse.ArgumentParser(description="LLM 结构化处理器")
    parser.add_argument("input", help="docx_processor 输出的 JSON 文件或包含 JSON 的目录")
    parser.add_argument("--output", "-o", help="输出路径（默认: 与输入同级 *_structured.json）")
    parser.add_argument("--batch", action="store_true", help="批量模式")
    parser.add_argument("--no-exec", "--dry-run", action="store_true", dest="dry_run",
                        help="不调用 API，只展示处理计划")
    parser.add_argument("--api-key", help="DeepSeek API Key（默认读取配置文件）")
    args = parser.parse_args()

    if args.api_key:
        os.environ["DEEPSEEK_API_KEY"] = args.api_key

    if args.batch:
        batch_process(args.input, args.output, args.dry_run)
    else:
        if not args.output and not args.dry_run:
            inp = Path(args.input)
            args.output = str(inp.parent / f"{inp.stem}_structured.json")
        process_document(args.input, args.output, args.dry_run)


if __name__ == "__main__":
    main()
