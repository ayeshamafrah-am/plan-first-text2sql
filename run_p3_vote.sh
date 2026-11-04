#!/bin/bash
# Paper 3, E12: plan voting (5 sampled plans + greedy, majority structure) with the three plan-first models
set -e; cd ~/t2sql
M=Qwen/Qwen3-1.7B; SETS="spider_dev bird_dev spider_syn spider_dk spider_realistic"
for ad in B_plan B_plan_s2 B_plan_s3; do
  for s in $SETS; do
    f=preds/p3/V_${ad}_$s.jsonl
    if [ -f $f ] && [ "$(wc -l < $f)" -eq "$(wc -l < data/p3/$s.jsonl)" ]; then continue; fi
    echo "=== VOTE $ad $s $(date)"; python infer_p3.py $M data/p3/$s.jsonl $f ckpt/p3_$ad --mode plan --plan-vote 5 2>&1 | grep --line-buffered -v Warning
  done
  python evaluate_p3.py preds/p3/V_${ad}_*.jsonl --tag "E12 plan-vote $(date +%F)"
done
echo "ALL DONE $(date)"
