#!/bin/bash
# run_eval.sh — 一键评测 (CI ready)
# 用法: bash run_eval.sh [--llm] [--ci]

cd "$(dirname "$0")/.."
python3 -u scripts/eval_auto.py "$@" --report eval/data/eval_result.json
