#!/bin/bash
# Paper 3, E9-E10: seeds 2 and 3 for A (direct) and B (plan-first); E11: oracle plan with model B
set -e; cd ~/t2sql; mkdir -p ckpt preds/p3 results
M=Qwen/Qwen3-1.7B; TR="data/p3/spider_train.jsonl data/p3/bird_train.jsonl"; SETS="spider_dev bird_dev spider_syn spider_dk spider_realistic"
train() { # name mode extra-flags
  [ -f ckpt/p3_$1/adapter_config.json ] && { echo "skip train $1"; return; }
  echo "=== TRAIN $1 ($2 $3) $(date)"; python train_sft_p3.py $M ckpt/p3_$1 $TR --mode $2 $3 2>&1 | grep -v Warning | tee results/p3_train_$1.txt
}
evalset() { # name mode adapter extra
  for s in $SETS; do
    f=preds/p3/$1_$s.jsonl
    if [ -f $f ] && [ "$(wc -l < $f)" -eq "$(wc -l < data/p3/$s.jsonl)" ]; then continue; fi
    echo "=== EVAL $1 $s $(date)"; python infer_p3.py $M data/p3/$s.jsonl $f ckpt/p3_$3 --mode $2 $4 2>&1 | grep --line-buffered -v Warning
  done
}
# oracle plan first (short, ~1-2 h): upper bound of plan-first if the plan were perfect
evalset O_oracle_plan plan B_plan --oracle-plan
python evaluate_p3.py preds/p3/O_oracle_plan_*.jsonl --tag "E11 oracle-plan $(date +%F)"
for seed in 2 3; do
  train A_direct_s$seed direct "--seed $seed"
  train B_plan_s$seed plan "--seed $seed"
  evalset A_direct_s$seed direct A_direct_s$seed
  evalset B_plan_s$seed plan B_plan_s$seed
done
python evaluate_p3.py preds/p3/[AB]_*_s[23]_*.jsonl --tag "E9-E10 seeds $(date +%F)"
echo "ALL DONE $(date)"
