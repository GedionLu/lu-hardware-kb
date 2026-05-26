#!/usr/bin/env python3
"""
batch_process_ugs.py — 批量处理 19Series User Guides

对每本 UG:
  1. extract_pdf_text.py (YOLO + 文本匹配) → raw_{name}.json
  2. llm_refine.py (LLM 润色) → refined_{name}.json
  3. 合并到最终 config_codes.json + image_groups.json

用法:
  python batch_process_ugs.py [--skip-yolo] [--skip-llm]
"""

import json, os, subprocess, sys, time, re, tempfile
from pathlib import Path
from collections import defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(BASE, 'scripts')
DATA = os.path.join(BASE, 'data')
OUTPUT = '/tmp/ug_batch'

# 7 本 User Guides (去重后)
UG_FILES = [
    {
        'path': os.path.expanduser('~/openclaw/19Series/Xenon-UG.pdf'),
        'product': '1900',
        'name': 'Xenon-UG',
    },
    {
        'path': os.path.expanduser('~/openclaw/19Series/195x/XEN195X-EN-UG.pdf'),
        'product': '195X',
        'name': 'XEN195X-EN-UG',
    },
    {
        'path': os.path.expanduser('~/openclaw/19Series/Granit XP/199x-en-ug.pdf'),
        'product': '199X',
        'name': '199x-en-ug',
    },
    {
        'path': os.path.expanduser('~/openclaw/19Series/Granit XP/Granit XP 199xi Series UG(1990iSR 1991iXR 1991iXLR 1991iSR 1991iXR 1991iXLR).pdf'),
        'product': '199Xi',
        'name': '199xi-series-ug',
    },
    {
        'path': os.path.expanduser('~/openclaw/19Series/1960/sps-ppr-xen196x-en-ug.pdf'),
        'product': '196X',
        'name': 'xen196x-en-ug',
    },
    {
        'path': os.path.expanduser('~/openclaw/19Series/1960/sps-ppr-196x-acc-en-ug.pdf'),
        'product': '196X-ACC',
        'name': '196x-acc-en-ug',
    },
    {
        'path': os.path.expanduser('~/openclaw/19Series/OCR-UG Rev B pdf.pdf'),
        'product': 'OCR',
        'name': 'OCR-UG',
    },
]


def run_cmd(cmd, desc='', timeout=3600):
    """运行命令并打印进度，用临时文件避免 pipe buffer deadlock"""
    print(f"  {desc}...", flush=True)
    t0 = time.time()
    outfile = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.out')
    errfile = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.err')
    try:
        result = subprocess.run(cmd, shell=True, stdout=outfile, stderr=errfile,
                                timeout=timeout)
        elapsed = time.time() - t0
        outfile.close()
        errfile.close()
        with open(outfile.name) as f:
            stdout_text = f.read()
        with open(errfile.name) as f:
            stderr_text = f.read()
        if result.returncode != 0:
            print(f"  ❌ FAILED ({elapsed:.0f}s)")
            if stderr_text:
                for l in stderr_text.strip().split('\n')[-15:]:
                    print(f"  err: {l[:120]}")
            if stdout_text:
                for l in stdout_text.strip().split('\n')[-10:]:
                    print(f"  out: {l[:120]}")
            return False
        print(f"  ✅ {elapsed:.0f}s")
        if stdout_text.strip():
            lines = stdout_text.strip().split('\n')
            for l in lines[-20:]:
                print(f"     {l[:120]}")
        return True
    finally:
        try:
            os.unlink(outfile.name)
        except:
            pass
        try:
            os.unlink(errfile.name)
        except:
            pass


def main():
    os.makedirs(OUTPUT, exist_ok=True)
    
    all_refined = []
    stats = []

    for i, ug in enumerate(UG_FILES):
        name = ug['name']
        pdf = ug['path']
        product = ug['product']

        if not os.path.exists(pdf):
            print(f"\n⚠️ {name}: PDF not found at {pdf}")
            continue

        print(f"\n{'='*60}")
        print(f"  [{i+1}/7] {name} ({product})")
        print(f"{'='*60}")

        raw_file = os.path.join(OUTPUT, f'raw_{name}.json')
        refined_file = os.path.join(OUTPUT, f'refined_{name}.json')

        # Step 1: YOLO + 文本匹配
        if os.path.exists(raw_file) and os.path.getsize(raw_file) > 100:
            print(f"  📂 Using cached raw: {raw_file}")
        else:
            ok = run_cmd(
                f'python3 -u {SCRIPTS}/extract_pdf_text.py '
                f'"{pdf}" -p {product} -o {raw_file} '
                f'--yolo-model {BASE}/weights/yolov8s-barcode-detection.pt',
                'YOLO + 文本匹配',
                timeout=900
            )
            if not ok:
                stats.append({'name': name, 'status': 'YOLO_FAILED'})
                continue

        # Load stats
        try:
            with open(raw_file) as f:
                raw_data = json.load(f)
            total = len(raw_data)
            decoded = sum(1 for r in raw_data if r.get('barcode_value'))
            has_label = sum(1 for r in raw_data if r.get('label_text'))
            print(f"  📊 {total} barcodes, {decoded} decoded, {has_label} with labels")
        except:
            total = decoded = has_label = '?'

        # Step 2: LLM 润色
        if os.path.exists(refined_file) and os.path.getsize(refined_file) > 100:
            print(f"  📂 Using cached refined: {refined_file}")
        else:
            to_refine = sum(1 for r in raw_data if r.get('barcode_value') and r.get('label_text'))
            if to_refine < 5:
                print(f"  ⏭️ 只有 {to_refine} 条可润色，跳过 LLM")
                refined_data = raw_data
            else:
                ok = run_cmd(
                    f'python3 -u {SCRIPTS}/llm_refine.py {raw_file} -o {refined_file} --batch 15',
                    'LLM 润色',
                    timeout=3600
                )
                if not ok:
                    refined_file = raw_file  # fallback

        # Load refined
        with open(refined_file if os.path.exists(refined_file) else raw_file) as f:
            refined_data = json.load(f)

        llm_count = sum(1 for r in refined_data if r.get('llm_refined'))
        stats.append({
            'name': name, 'product': product, 'status': 'OK',
            'total': total, 'decoded': decoded if type(decoded)==int else 0,
            'llm_refined': llm_count,
        })
        all_refined.extend(refined_data)
        print(f"  ✅ Done: {len(refined_data)} entries, {llm_count} LLM refined")

    # ═══ Final merge ═══
    print(f"\n{'='*60}")
    print(f"  Final: merging {len(all_refined)} entries into config_codes.json")
    print(f"{'='*60}")

    # Load existing config_codes
    with open(os.path.join(DATA, 'config_codes.json')) as f:
        existing = json.load(f)

    # Keep non-19Series products (OH430, HH760, 7120, etc.)
    non_ug = [c for c in existing
              if c.get('model', '') not in
              ['1900', '195X', '199X', '199Xi', '196X', '196X-ACC', 'OCR', 'XEN197X']]

    # Convert new entries
    new_codes = []
    seen = set(c.get('barcode_value', '') for c in non_ug)
    seen.update(c.get('barcode_value', '') for c in existing if c.get('model') == 'XEN197X')

    for r in all_refined:
        bc = r.get('barcode_value', '')
        if not bc or not re.match(r'^[A-Z0-9~.]{3,}$', bc):
            continue
        if bc in seen:
            continue
        seen.add(bc)

        product = r.get('product_name', r.get('model', ''))
        desc = r.get('description', r.get('label_text', ''))
        
        new_codes.append({
            'type': 'config_code',
            'code_name': f"{product}-{bc}",
            'description': desc,
            'product_name': product,
            'model': product,
            'image_url': r.get('image_url', ''),
            'image_path': r.get('image_path', ''),
            'source_file': r.get('source_file', ''),
            'source_page': r.get('source_page', ''),
            'barcode_value': bc,
            'category': r.get('category', ''),
            'llm_refined': r.get('llm_refined', False),
        })

    final = non_ug + [c for c in existing if c.get('model') == 'XEN197X'] + new_codes

    backup = os.path.join(DATA, 'config_codes_backup.json')
    with open(backup, 'w') as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

    with open(os.path.join(DATA, 'config_codes.json'), 'w') as f:
        json.dump(final, f, ensure_ascii=False, indent=2)

    print(f"\n📊 Batch complete:")
    print(f"  Existing (non-UG): {len(non_ug)}")
    print(f"  New from batch:   {len(new_codes)}")
    print(f"  Total:            {len(final)}")
    print(f"  Backup:           {backup}")

    print(f"\n{'='*60}")
    print(f"  Per-PDF stats:")
    for s in stats:
        prod = s.get('product', '')
        print(f"  {s['status']:>12} {prod:>10} {s['name']:>25} "
              f"total={s.get('total','?')} decoded={s.get('decoded','?')} llm={s.get('llm_refined','?')}")


if __name__ == '__main__':
    main()
