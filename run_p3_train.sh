#!/bin/bash
# Paper 3 training: A direct, B plan-first, C plan-first+curriculum, D direct+curriculum; then evaluation on 5 sets
set -e; cd ~/t2sql; mkdir -p ckpt preds/p3 results
M=Qwen/Qwen3-1.7B; TR="data/p3/spider_train.jsonl data/p3/bird_train.jsonl"
train() { # name mode extra-flags
  [ -f ckpt/p3_$1/adapter_config.json ] && { echo "skip train $1"; return; }
  echo "=== TRAIN $1 ($2 $3) $(date)"; python train_sft_p3.py $M ckpt/p3_$1 $TR --mode $2 $3 2>&1 | grep -v Warning | tee results/p3_train_$1.txt
}
train A_direct direct ""
train B_plan plan ""
train C_plan_curr plan "--curriculum strict"
train D_direct_curr direct "--curriculum strict"
train E_plan_comp plan "--curriculum competence"
train F_direct_comp direct "--curriculum competence"
for name in A_direct:direct B_plan:plan C_plan_curr:plan D_direct_curr:direct E_plan_comp:plan F_direct_comp:direct; do
  n=${name%%:*}; mode=${name##*:}
  for s in spider_dev bird_dev spider_syn spider_dk spider_realistic; do
    [ -f preds/p3/${n}_${s}.jsonl ] && continue
    echo "=== EVAL $n $s $(date)"; python infer_p3.py $M data/p3/$s.jsonl preds/p3/${n}_${s}.jsonl ckpt/p3_$n --mode $mode 2>&1 | grep --line-buffered -v Warning
  done
done
python evaluate_p3.py preds/p3/[EF]_*_*.jsonl --tag "E7-E8 competence $(date +%F)"
