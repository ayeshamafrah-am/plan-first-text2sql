#!/bin/bash
# E1/E2: zero-shot Qwen3-1.7B, direct vs plan-first, on all five evaluation sets
set -e; cd ~/t2sql; mkdir -p preds/p3 results
M=Qwen/Qwen3-1.7B
for mode in direct plan; do
  for s in spider_dev bird_dev spider_syn spider_dk spider_realistic; do
    [ -f preds/p3/zs_${mode}_${s}.jsonl ] && continue
    python infer_p3.py $M data/p3/$s.jsonl preds/p3/zs_${mode}_${s}.jsonl none --mode $mode 2>&1 | grep -v Warning
  done
done
python evaluate_p3.py preds/p3/zs_*.jsonl --tag "E1/E2 zero-shot $(date +%F)"
