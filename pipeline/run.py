#!/usr/bin/env python3
"""
管线总入口: 依次执行 extract → classify → associate
用法:
  python3 pipeline/run.py              # 全量执行
  python3 pipeline/run.py --skip-extract  # 跳过提取
  python3 pipeline/run.py --stage classify  # 只运行分类
"""

import os
import sys
import subprocess
import json
import time

PIPELINE_DIR = os.path.dirname(__file__)
SCRIPTS = {
    'extract':  os.path.join(PIPELINE_DIR, 'extract.py'),
    'classify': os.path.join(PIPELINE_DIR, 'classify.py'),
    'associate': os.path.join(PIPELINE_DIR, 'associate.py'),
}

OUTPUT_DIR = os.path.join(PIPELINE_DIR, '..', 'data')


def print_stage(title):
    print()
    print("=" * 60)
    print(f"  Stage: {title}")
    print("=" * 60)


def run_script(name, skip=False):
    if skip or name not in SCRIPTS:
        return True
    
    script = SCRIPTS[name]
    if not os.path.exists(script):
        print(f"[错误] 脚本不存在: {script}")
        return False

    print_stage(name.upper())
    result = subprocess.run([sys.executable, script])
    return result.returncode == 0


def summary():
    """打印管线完成状态"""
    print()
    print_stage("SUMMARY")
    
    files = {
        'image_metadata.json':   '原始提取元信息',
        'image_classification.json': '图片分类结果',
        'image_index.json':      '最终索引 (图片+型号)',
        'doc_model_map.json':    '文档→型号映射',
    }

    for fname, desc in files.items():
        fpath = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(fpath):
            size = os.path.getsize(fpath)
            print(f"  ✅ {desc:20s} -> {fname} ({size:,} bytes)")
        else:
            print(f"  ❌ {desc:20s} -> {fname} (不存在)")

    # 统计图片目录
    img_dir = '/opt/kb-images/raw'
    if os.path.exists(img_dir):
        count = len([f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))])
        print(f"  📷 图片文件: {img_dir}/ -> {count} 个文件")


def main():
    args = sys.argv[1:]

    skip_extract = '--skip-extract' in args
    only_stage = None
    for a in args:
        if a.startswith('--stage='):
            only_stage = a.split('=', 1)[1]

    print("=" * 60)
    print("  Image Pipeline v1.0")
    print("  KB Source: /tmp/KnowledgeBase")
    print("  Output:    /opt/kb-images/ + data/")
    print("=" * 60)
    print(f"  Args: {' '.join(args) if args else '(full run)'}")

    t_start = time.time()

    if only_stage:
        # 只运行指定阶段
        success = run_script(only_stage)
        if not success:
            print(f"  ❌ 阶段 {only_stage} 失败")
            sys.exit(1)
    else:
        # 全量运行
        success = True
        if not skip_extract:
            success = run_script('extract') and success
        
        if success:
            success = run_script('classify') and success
        
        if success:
            success = run_script('associate') and success

    elapsed = time.time() - t_start

    print()
    if success:
        print(f"✅ 管线完成，耗时 {elapsed:.1f} 秒")
        summary()
    else:
        print(f"❌ 管线中断，耗时 {elapsed:.1f} 秒")
        sys.exit(1)


if __name__ == '__main__':
    main()
