# plan-first-text2sql

Code, plans, logs and results for the paper *Plan First, Then Write: Structure-First Generation and Curriculum Training for a Small Text-to-SQL Model*.

All experiments ran on one laptop GPU (NVIDIA RTX 5070 Ti, 12 GB) with Qwen/Qwen3-1.7B and LoRA.
LoRA adapters (ten, ~70 MB each): https://huggingface.co/ayeshmaffi/plan-first-adapters

## Pipeline

| Step | Script | Output |
|---|---|---|
| Data: hardness labels + structural plans | `make_p3_data.py` | `data/p3/*.jsonl` |
| Inference (direct / plan-first / oracle plan / plan voting) | `infer_p3.py` | `preds/p3/*.jsonl` |
| Execution accuracy per hardness + plan quality | `evaluate_p3.py` | `*.eval.jsonl`, `*.summary.json` |
| LoRA fine-tuning, three data orders | `train_sft_p3.py` | `ckpt/p3_*` |
| Exact McNemar test | `significance_p3.py` | printed table |
| Error categories | `analyze_errors_p3.py` | `results/p3_errors.json` |
| Runners | `run_p3_zs.sh`, `run_p3_train.sh`, `run_p3_more.sh`, `run_p3_vote.sh` | |

`results/EXPERIMENT_LOG.md` (Paper 3 section) records every run with date, configuration, wall-clock time and peak VRAM.
`results/p3_results.txt` has one line per evaluation; `results/p3_collect.txt` the seeds, significance and error tables.
`plans/plans.jsonl` gives the extracted structural plan, hardness and complexity score for all 21,072 questions (ids only; obtain Spider, BIRD, Spider-Syn, Spider-DK and Spider-Realistic from their official sources).

## Reproduce

```bash
python3 -m venv venv && source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt_tab')"
git clone https://github.com/taoyds/test-suite-sql-eval.git evalsuite
# put Spider in data/spider_data, BIRD in data/bird (unified JSONL as in prepare_data.py of our earlier release),
# Spider-Syn / Spider-DK / spider-realistic.json in data/p3/, then:
python make_p3_data.py
./run_p3_zs.sh          # zero-shot, direct and plan-first
./run_p3_train.sh       # models A-F and their evaluation
./run_p3_more.sh        # oracle plan, seeds 2 and 3
./run_p3_vote.sh        # plan voting
python significance_p3.py preds/p3/A_direct preds/p3/B_plan
python analyze_errors_p3.py preds/p3/A_direct preds/p3/B_plan preds/p3/O_oracle_plan
```

## Main results (execution accuracy %, 3-seed mean)

| Model | Spider dev | BIRD dev | Spider-Syn | Spider-DK | Spider-Realistic |
|---|---|---|---|---|---|
| A direct SFT | 69.7 | 36.7 | 55.6 | 56.1 | 61.3 |
| B plan-first SFT | 71.5 | 38.5 | 58.7 | 55.3 | 60.0 |
| B + gold plan (oracle) | 81.9 | 56.8 | 74.1 | 67.0 | 71.7 |
| C/E plan-first + curriculum (strict / competence) | 43.0 / 65.3 | 30.9 / 34.1 | 33.1 / 51.7 | 32.2 / 50.4 | 38.0 / 54.3 |

## License

Code: MIT. Datasets retain their original licenses.
