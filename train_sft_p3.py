#!/usr/bin/env python
"""train_sft_p3.py  (Paper 3)  LoRA fine-tuning, direct or plan-first, optional curriculum.
Usage: python train_sft_p3.py <model> <out_dir> <train.jsonl>... [--mode direct|plan] [--curriculum] [--epochs 1] [--lr 2e-4]
                              [--max-len 3072] [--accum 16] [--seed 42] [--warmup 30]
  --mode plan   : target = "PLAN:\n<plan>\nSQL:\n<sql>"   (structure-first);  direct: target = sql
  --curriculum  : examples are presented from structurally simple to complex (score from the gold plan),
                  shuffled only within equal-score buckets; otherwise random order.
Prompts are built by infer_p3.build_messages, rendered with the chat template (Qwen3 thinking off), so training and
inference see identical text. Examples longer than --max-len tokens are dropped (count reported) so no target is cut.
"""
import argparse, json, random, time, torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from torch.utils.data import SequentialSampler
from infer_p3 import build_messages, target_text

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("out"); ap.add_argument("files", nargs="+")
ap.add_argument("--mode", choices=["direct", "plan"], default="direct")
ap.add_argument("--curriculum", choices=["none", "strict", "competence"], default="none",
                help="strict: sorted easy-to-hard (shuffled within equal scores); competence: Platanios et al. 2019, pool grows from easy to all")
ap.add_argument("--c0", type=float, default=0.1, help="initial competence (fraction of easiest examples available at step 0)")
ap.add_argument("--epochs", type=float, default=1.0); ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--max-len", type=int, default=3072); ap.add_argument("--accum", type=int, default=16)
ap.add_argument("--seed", type=int, default=42); ap.add_argument("--warmup", type=int, default=30)
ap.add_argument("--limit", type=int, default=0, help="use only N examples (smoke test)")
a = ap.parse_args()
random.seed(a.seed)


def complexity(plan):
    """Structural score of a gold plan: extra tables + yes-flags + set op + aggregates."""
    s = 0
    for line in plan.split("\n"):
        if line.startswith("tables:"): s += max(0, len([t for t in line[7:].split(",") if t.strip()]) - 1)
        elif line.startswith("joins:"): continue
        else:
            for part in line.split("|"):
                k, _, v = part.partition(":"); k, v = k.strip(), v.strip()
                if k == "set op": s += v != "none"
                elif k == "aggregates": s += v != "none"
                elif v == "yes": s += 1
    return s


tok = AutoTokenizer.from_pretrained(a.model)
kw = {"enable_thinking": False} if "qwen3" in a.model.lower() else {}
rows, dropped = [], 0
for fn in a.files:
    for l in open(fn):
        it = json.loads(l)
        if not it.get("plan"): continue
        prompt = tok.apply_chat_template(build_messages(it, a.mode), add_generation_prompt=True, tokenize=False, **kw)
        comp = target_text(it, a.mode)
        n = len(tok(prompt + comp)["input_ids"]) + 1
        if n > a.max_len: dropped += 1; continue
        rows.append({"prompt": prompt, "completion": comp, "score": complexity(it["plan"])})
print(f"{len(rows)} training examples from {len(a.files)} files; dropped {dropped} longer than {a.max_len} tokens")

random.shuffle(rows)
if a.limit: rows = rows[:a.limit]
from collections import Counter
if a.curriculum != "none":
    print("curriculum buckets (score: n):", dict(sorted(Counter(r["score"] for r in rows).items())))
if a.curriculum == "strict":
    rows.sort(key=lambda r: r["score"])          # stable sort: random order within each score bucket
elif a.curriculum == "competence":
    # Platanios et al. (2019): competence c(t) = min(1, sqrt(t^2 + (1-t^2) c0^2)), t = training progress.
    # At each step one example is drawn uniformly (with replacement) from the easiest c(t) fraction of the data,
    # so simple examples keep appearing until the end and complex ones enter gradually. Same number of steps as 1 epoch.
    import math
    srt = sorted(rows, key=lambda r: r["score"]); N = len(srt)
    order = []
    for i in range(N):
        t = i / N; c = min(1.0, math.sqrt(t * t + (1 - t * t) * a.c0 * a.c0))
        order.append(srt[random.randrange(max(1, int(c * N)))])
    seen = len({id(r) for r in order})
    rows = order
    q = [sum(r["score"] for r in rows[i:i + N // 10]) / max(1, len(rows[i:i + N // 10])) for i in range(0, N, N // 10)]
    print(f"competence curriculum: {seen}/{N} distinct examples used; mean score per tenth of training:", [round(x, 2) for x in q])
ds = Dataset.from_list([{"prompt": r["prompt"], "completion": r["completion"]} for r in rows])

model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda")
cfg = SFTConfig(
    output_dir=a.out, num_train_epochs=a.epochs, learning_rate=a.lr, lr_scheduler_type="cosine", warmup_steps=a.warmup,
    per_device_train_batch_size=1, gradient_accumulation_steps=a.accum, gradient_checkpointing=True,
    bf16=True, logging_steps=10, save_strategy="epoch", save_total_limit=1, report_to="none",
    max_length=a.max_len, seed=a.seed, dataloader_num_workers=0, completion_only_loss=True,
)
lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])


class CurriculumTrainer(SFTTrainer):
    """Keeps the dataset order (no random sampler) so the curriculum is respected."""
    def _get_train_sampler(self, *args, **kwargs):
        return SequentialSampler(self.train_dataset)


TrainerCls = CurriculumTrainer if a.curriculum != "none" else SFTTrainer
trainer = TrainerCls(model=model, args=cfg, train_dataset=ds, processing_class=tok, peft_config=lora)
t0 = time.time()
trainer.train()
trainer.save_model(a.out)
json.dump(trainer.state.log_history, open(f"{a.out}/log_history.json", "w"), indent=1)
json.dump(vars(a) | {"n_examples": len(rows), "dropped": dropped}, open(f"{a.out}/train_args.json", "w"), indent=1)
print(f"saved adapter to {a.out}; wall-clock {(time.time() - t0) / 3600:.2f} h; peak VRAM {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
