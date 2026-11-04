## 2026-10-14 — Zero-shot baseline, Qwen2.5-Coder-3B-Instruct (4-bit NF4), greedy, Spider dev
EX: ALL 62.7 | easy 85.1 | medium 61.0 | hard 54.0 | extra 42.8 | exec errors 157/1034
Log: results/zeroshot_3b_spider_dev.log  Preds: preds/zeroshot_3b_spider_dev.jsonl
## 2026-10-14 — Zero-shot baseline, Qwen2.5-Coder-3B-Instruct (4-bit NF4), greedy, BIRD dev (with evidence)
EX: ALL 36.8 | simple 44.6 | moderate 26.1 | challenging 20.7 | exec errors 476/1534
Log: results/zeroshot_3b_bird_dev.log  Preds: preds/zeroshot_3b_bird_dev.jsonl
## 2026-10-14 — SFT (QLoRA r=16, 2 epochs, Spider train only), Qwen2.5-Coder-3B, greedy
Train: 866 steps, 2h23m, final loss 0.04, peak VRAM 5.12 GB. Log: results/sft_3b_spider_train.log
Spider dev EX: ALL 76.4 | easy 92.7 | medium 77.1 | hard 73.6 | extra 53.0 | errors 45   (zero-shot was 62.7)
BIRD dev  EX: ALL 35.7 | simple 45.1 | moderate 22.6 | challenging 17.2 | errors 389  (zero-shot was 36.8)
Finding: Spider-only SFT does not transfer to BIRD -> motivates multi-dataset + execution-feedback training.
## 2026-10-15 — SFT-mixed (QLoRA r=16, 2 epochs, Spider train + BIRD train, max_len 3072), Qwen2.5-Coder-3B, greedy
Train: 1996 steps, 7h50m, ~460 examples dropped for exceeding 3072 tokens, final loss ~0.06, peak VRAM 6.93 GB
Spider dev EX: ALL 77.4 | easy 94.0 | medium 78.9 | hard 72.4 | extra 53.6 | errors 36
BIRD dev  EX: ALL 47.0 | simple 54.7 | moderate 35.1 | challenging 35.9 | errors 171
Finding: mixed-dataset SFT generalizes (BIRD +11.3 over Spider-only SFT) without hurting Spider. -> official SFT baseline.
## 2026-10-17 — Control: merged SFT-mixed (bf16 merge, requantized 4-bit), no DPO, BIRD dev
EX: ALL 46.3 | simple 54.1 | moderate 36.0 | challenging 29.7 | errors 195  (adapter version was 47.0) -> merge OK, DPO-v1 recipe at fault
## 2026-10-17 — DPO-v2 (beta 0.2, lr 5e-6, sigmoid+sft loss, 1 epoch) on merged SFT-mixed
Train: 3h57m, eval rewards/chosen -0.004 (fixed), rewards/rejected -0.065, acc 0.656, margins 0.06
Spider dev EX: ALL 78.1 | easy 94.0 | medium 79.4 | hard 75.3 | extra 54.2 | errors 36   (SFT-mixed 77.4)
BIRD dev  EX: ALL 46.5 | simple 54.2 | moderate 36.2 | challenging 30.3 | errors 188 (control 46.3)
Result: no significant change (within noise). Update too conservative; hard tiers and errors move in the right direction.
## 2026-10-17 — DPO-v3 (beta 0.1, lr 2e-5, sigmoid+sft loss, 1 epoch) on merged SFT-mixed  [= v1 + SFT term]
Train: 482 steps, 3h57m, peak 6.77 GB. Final eval: rewards/chosen -0.18, rewards/rejected -0.68, acc 0.691, margins 0.50
(v1 had chosen -0.62; v2 had margins only 0.06) -> strongest margin with chosen likelihood mostly preserved.
NOTE: train log overwritten by accidental rerun; metrics transcribed from terminal.
## 2026-10-17 — DPO-v3 eval
Spider dev EX: ALL 74.6 | easy 92.3 | medium 74.4 | hard 70.7 | extra 52.4 | errors 33   (SFT-mixed 77.4)
BIRD dev  EX: ALL 47.7 | simple 55.6 | moderate 36.6 | challenging 32.4 | errors 159 (control 46.3; best BIRD so far)
Result: BIRD +1.4 overall, challenging +2.7, errors -18%; Spider -2.8. Trade-off: strong update shifts toward BIRD (66% of pairs).
## 2026-10-18 — DPO-v4 (= v3 settings, source-balanced pairs, Spider x2 -> 10,318 pairs), 646 steps, 5h06m
Final eval: rewards/chosen -0.22, rejected -0.89, acc 0.68, margins 0.68
Spider dev EX: ALL 73.2 | easy 91.5 | medium 74.2 | hard 70.1 | extra 46.4 | errors 37
BIRD dev  EX: ALL 47.3 | simple 55.4 | moderate 36.4 | challenging 31.0 | errors 154
Result: balancing did NOT recover Spider (73.2 < v3 74.6). Hypothesis "BIRD skew causes Spider drop" rejected. Spider loss tracks update strength, not source mix.
Error analysis v3 vs SFT (Spider dev): losses are schema-fidelity errors (wrong/hallucinated column names, spurious joins, column reorder, truncated literals), not logic. Hypothesis: 70% surface pairs inject noisy token-level signal. -> v5: structural pairs only.
## 2026-10-18 — DPO-v5 (v3 settings, structural pairs only: 1854 pairs, 111 steps, 1h01m), margins 0.035
Spider dev EX: ALL 77.2 | easy 92.7 | medium 78.0 | hard 75.9 | extra 53.0 | errors 40
BIRD dev  EX: ALL 46.3 | simple 53.7 | moderate 36.6 | challenging 30.3 | errors 203
Result: Spider preserved (confirms surface pairs cause the Spider drop) but too few pairs to move BIRD. ~= SFT.
EXPERIMENTS CLOSED. Method for paper = DPO-v3. Remaining: Spider test eval (SFT-mixed merged, DPO-v3).
## 2026-10-18 — Spider TEST (2147 q, held out, evaluated once)
SFT-mixed (merged): ALL 76.6 | easy 90.6 | medium 76.9 | hard 70.6 | extra 65.0 | errors 72
DPO-v3:             ALL 75.6 | easy 90.6 | medium 76.5 | hard 70.2 | extra 60.8 | errors 82
Result: no gain on Spider test (-1.0, within noise except extra -4.2). Consistent with dev findings.

## PAPER 2 — 2026-10-19: Gold-query portability (SQLite → MySQL 8.4 / PostgreSQL 18)
Port: port_sqlite.py (data-driven typing, lower-cased identifiers, utf8mb4_0900_bin). Verify: verify_gold.py (sqlglot transpile + ANY_VALUE/LIKE/JULIANDAY/SUBSTR/ROUND/boolean rewrites; result-set equality, 3-decimal numeric tolerance).
Spider dev (1034): MySQL match 981 (94.9%), mismatch 12, tie 26, barecol 13, exec_err 2 | PostgreSQL match 979 (94.7%), mismatch 10, tie 29, barecol 12, exec_err 4
BIRD dev (1534, sqlite gold errors 3): MySQL match 1456 (94.9%), mismatch 23, tie 34, barecol 3, exec_err 15 | PostgreSQL match 1423 (92.8%), mismatch 13, tie 20, barecol 5, exec_err 70
Databases ported: Spider dev 20, BIRD dev 11 (codebase_community needed PyMySQL; mysql-connector failed on escaped text). CPU only.

## PAPER 2 — 2026-10-19: Enterprise dataset v2 — databases generated and ported
gen_enterprise.py (seed = crc32(name)): university 79,305 rows (v3: realistic hostel capacity, Executive MBA) / hospital 112,436 / ecommerce 167,840 / banking 424,404 / manufacturing 23,666 (79 tables, 650 columns). Indian context (INR, GST, IFSC, UHID).
Content SHA-256 (content_hash.py), identical on two machines: banking b8770b7b…, ecommerce 23dfecab…, hospital e13f34b5…, manufacturing 05ebae74…, university 66507dbd…
Ported to MySQL + PostgreSQL with port_sqlite.py (prefix ent_). Total databases now 36 × 3 engines.

## PAPER 2 — 2026-10-19: Enterprise dataset v4 (final generator) + 400 draft gold questions
gen_enterprise.py v4 adds realistic variability (uneven staffing, unsold products, stock-outs, unused coupons, unapproved suppliers, brand-category consistency, ward-appropriate diagnoses, late deliveries).
Rows: university 79,305 / hospital 112,263 / ecommerce 170,037 / banking 424,597 / manufacturing 23,986.
Content SHA-256 (identical on two machines): university 66507dbd… hospital f7081e75… ecommerce 62765e97… banking e0bef289… manufacturing a965ffac…
Gold questions drafted: 80 per database = 400 (10 slot types), all executed with non-empty results; awaiting human review (enterprise_review_v1.xlsx).

## PAPER 2 — 2026-10-20: E3 zero-shot baseline, Qwen2.5-Coder-1.5B-Instruct (bf16, greedy, dialect tag + sample-value hints in prompt)
Spider dev: SQLite 61.0 (1034) | MySQL 57.8 (981) | PostgreSQL 57.6 (979)
BIRD dev:   SQLite 23.5 (1534) | MySQL 21.3 (1456) | PostgreSQL 21.2 (1423)
Enterprise: SQLite 21.0 (400)  | MySQL 23.7 (392)  | PostgreSQL 22.4 (380)
Non-executing (SQLite): Spider 185, BIRD 751, Enterprise 132. Runtime ~2 h total. Files preds/p2/zs15_*.jsonl, results/p2_baselines.txt

## PAPER 2 — 2026-10-20: Training data built (prep_train_p2.py, seed 42)
Spider train 7,000 q / 140 dbs; BIRD train subset 4,000 q sampled from the 56 training dbs <= 300 MB (13 large dbs excluded). All ported to MySQL+PostgreSQL (BLOB columns dropped, MySQL LONGTEXT).
Gold portability: Spider train MySQL 95.9% / PG 95.1%; BIRD train MySQL 90.3% / PG 88.5% (172 PG exec errors; BIRD relies more on SQLite loose typing).
Training files (verified gold only): sqlite 11,000 | mysql 10,325 | postgres 10,196 -> multi-dialect total 31,521. Prompt = infer_p2.build_messages (dialect tag + sample-value hints).

## PAPER 2 — 2026-10-20: E4 SFT control (SQLite only) — Qwen2.5-Coder-1.5B-Instruct, bf16 LoRA r16/a32, lr 2e-4 cosine, 1 epoch, batch 1x16, max_len 3072
Data: spider_train_sqlite 7,000 + bird_train_sqlite 4,000 = 11,000. Steps 630. Wall-clock 1.10 h, peak VRAM 4.4 GB. Train loss 0.15 (final ~0.11), token acc ~0.965. Adapter ckpt/p2_sft_sqlite.

## PAPER 2 — 2026-10-20: E4 SFT control (SQLite only) — Qwen2.5-Coder-1.5B-Instruct, bf16 LoRA r16/a32, lr 2e-4 cosine, 1 epoch, batch 1x16, max_len 3072
Data: spider_train_sqlite 7,000 + bird_train_sqlite 4,000 = 11,000. Steps 630. Wall-clock 1.10 h, peak VRAM 4.4 GB. Train loss 0.15 (final ~0.11), token acc ~0.965. Adapter ckpt/p2_sft_sqlite.

## PAPER 2 — 2026-10-21: E4/E5 results (Qwen2.5-Coder-1.5B-Instruct + LoRA, greedy)
E4 SFT-SQLite (11,000 ex): Spider 74.6 / 65.6 / 67.5 | BIRD 35.5 / 29.9 / 28.3 | Ent 24.0 / 24.0 / 20.0  (SQLite/MySQL/PG). Non-exec Spider 60/159/158.
E5 SFT-multi (31,521 ex, 1841 steps, 3.10 h, 4.4 GB, train loss 0.079): Spider 76.8 / 76.0 / 75.7 | BIRD 39.2 / 38.8 / 38.7 | Ent 27.0 / 24.0 / 23.7. Non-exec Spider 48/41/57.
Finding: SQLite-only SFT widens the dialect gap (Spider 9 pts); multi-dialect SFT closes it (<1.1 pts) and improves SQLite too. Enterprise barely moves -> domain gap, not dialect gap.

## PAPER 2 — 2026-10-21: E6/E7 domain adaptation (continue from E5 adapter; 3 epochs, lr 5e-5, accum 4; split seed 42: 30 adapt / 50 test per db)
Enterprise test (250 q): zero-shot 20.0/23.1/21.7 | E4 23.6/23.1/18.8 | E5 27.6/22.7/22.9 | E7 k=10 25.6/33.2/33.3 | E6a k=30 44.0/43.7/45.4 (SQLite/MySQL/PG). Non-exec SQLite 87 -> 36.
E6a per db (SQLite): banking 40, ecommerce 48, hospital 40, manufacturing 44, university 48 (E5: 28/34/20/28/28).
E6b LODO (held-out db only, SQLite/MySQL/PG): banking 40.0/34.0/34.7 | ecommerce 46.0/32.0/29.8 | hospital 38.0/46.9/48.9 | manufacturing 34.0/34.7/38.8 | university 38.0/44.9/39.6 — vs E5 on same db 18-34.
Finding: in-domain adaptation with 30 q/db ~ +17 pts; adaptation on OTHER domains transfers +10-20 pts to an unseen enterprise database. Each adaptation run < 10 min GPU.

## PAPER 2 — 2026-10-22: Ablations
A2 E5 without sample-value hints (inference only): Spider 71.8/72.7/71.1 | BIRD 37.2/35.4/34.6 | Ent 16.2/16.6/15.3 (vs 76.8/76.0/75.7 | 39.2/38.8/38.7 | 27.0/24.0/23.7). Hints matter most on unfamiliar schemas.
A1 multi-dialect SFT retrained WITHOUT dialect tag (ckpt/p2_sft_multi_notag, 31,521 ex): Spider 74.9/76.1/76.8 | BIRD 37.6/37.5/37.0 | Ent 24.5/26.5/22.9. Tag effect within noise; the multi-dialect data (and dialect-specific schema text) does the work.
GPU phase of Paper 2 complete: 1 zero-shot + 2 SFT + 1 ablation SFT + 7 adaptation runs + 10 evaluation passes (~20 h total).
E8 error analysis (analyze_errors.py, results/p2_error_analysis.txt): SFT-SQLite on Spider-MySQL/PG: 89 GROUP BY-rule errors (vs 1 on SQLite), BIRD syntax 77/118 + dialect-fn 61/61; multi-dialect: 5/2 and 17/12, 0/59. Enterprise unknown-column errors 107 -> 72 -> 54 -> 26 (zero-shot -> E4 -> E5 -> E6a); 'wrong result' remains ~40% after adaptation.

## PAPER 2 — 2026-10-23: EntSQL question review
All 400 drafted questions (enterprise_review_v1.xlsx) reviewed by the author against their executed answers; approved without changes. Gold set frozen = enterprise_gold_draft.jsonl (one wording fix applied earlier: U002 "joined in 2021 or later").

---
# Paper 3: structure-first (plan-then-SQL) generation and curriculum training for a small text-to-SQL model

Hardware/software as before: laptop RTX 5070 Ti 12 GB, 32 GB RAM, WSL2 Ubuntu, Python 3.14 venv at ~/t2sql.
Backbone: Qwen/Qwen3-1.7B (bf16, thinking disabled), LoRA r=16, alpha=32, dropout 0.05, all attention+MLP projections,
lr 2e-4 cosine, 30 warm-up steps, batch 1 x 16 accumulation, 1 epoch, max length 3072, completion-only loss, seed 42 unless stated.
All scripts: make_p3_data.py, infer_p3.py, evaluate_p3.py, train_sft_p3.py, significance_p3.py, analyze_errors_p3.py,
run_p3_zs.sh, run_p3_train.sh, run_p3_more.sh, run_p3_vote.sh. Results lines: results/p3_results.txt; collected stats: results/p3_collect.txt.

## 2026-10-27  Data (make_p3_data.py)
- Sources: Spider train 7,000 / dev 1,034; BIRD train 9,428 / dev 1,534 (unified files from Paper 1); Spider-Syn dev 1,034 (ygan/Spider-Syn),
  Spider-DK 534 (ygan/Spider-DK, 1 query unparsable by sqlglot, dropped), Spider-Realistic 508 (Zenodo 5205322).
- Hardness: Spider official evaluator (evalsuite/process_sql + Evaluator.eval_hardness; needs nltk punkt_tab). Spider dev: easy 248, medium 446, hard 174, extra 166 (matches official).
  Spider train: 1,694 / 2,777 / 1,461 / 1,068. Syn: 248/440/177/169. DK: 110/245/74/105. Realistic: 109/203/99/97. BIRD keeps its own simple/moderate/challenging labels.
- Plan = structural outline extracted from gold SQL with sqlglot: tables; join conditions (aliases resolved); where/group by/having/order by/limit flags;
  nested; set op; distinct; aggregate functions. Written as 3-4 text lines. Complexity score (for curriculum) = (tables-1) + yes-flags + set op + aggregates.

## 2026-10-27/28  E1-E2 zero-shot (run_p3_zs.sh), greedy, max 320 new tokens
| set | direct EX | plan-first EX | direct non-exec | plan non-exec |
| Spider dev | 57.1 (81.5/59.2/42.0/30.7) | 41.0 | 132 | 185 |
| BIRD dev | 21.1 (28.2/10.3/9.7) | 17.8 | 592 | 636 |
| Spider-Syn | 47.9 | 28.7 | 154 | 242 |
| Spider-DK | 45.5 | 34.1 | 69 | 119 |
| Spider-Realistic | 49.0 | 30.3 | 82 | 130 |
Untrained plan quality: tables-F1 65-71, flag accuracy 41-46; EX 52.2 when plan tables right vs 37.9 when wrong (Spider dev).
Conclusion: prompting for a plan without training hurts a 1.7B model; the plan step must be learned.

## 2026-10-28  E3-E6 training (run_p3_train.sh), Spider train + BIRD train
- Training examples after length filter: direct 15,953 (475 dropped > 3072 tokens), plan-first 15,657 (771 dropped).
- A direct SFT: 1.34 h, 4.8 GB. B plan-first SFT: 1.44 h, 4.8 GB. C plan-first + strict curriculum: 1.70 h. D direct + strict curriculum: 1.33 h.
  Strict curriculum = examples sorted by complexity score (shuffled within equal score), sequential sampler.
| model | Spider (easy/med/hard/extra) | BIRD (simple/mod/chal) | Syn | DK | Realistic | non-exec Spider/BIRD |
| A direct | 70.1 (88.3/71.5/61.5/48.2) | 36.7 (44.8/25.2/21.5) | 56.5 | 56.2 | 61.2 | 69 / 232 |
| B plan | 71.9 (87.1/75.8/63.8/47.0) | 37.7 (45.4/26.7/23.6) | 59.1 | 56.0 | 59.3 | 38 / 179 |
| C plan+strict | 43.0 (45.2/42.6/50.0/33.7) | 30.9 | 33.1 | 32.2 | 38.0 | 139 / 219 |
| D direct+strict | 45.0 (41.5/43.5/57.5/41.0) | 28.0 | 32.2 | 39.7 | 36.8 | 116 / 309 |
B plan quality (Spider dev): tables-F1 95.5, flags 96.9; EX 78.0 when tables right vs 30.6 when wrong. BIRD: 90.1 / 94.5; 47.9 vs 11.4.
Strict curriculum collapses on easy questions (catastrophic forgetting: last steps see only complex examples).

## 2026-10-29  E7-E8 competence-based curriculum (Platanios et al. 2019), c0 = 0.1, sampling with replacement, same step count
- E plan + competence: 1.36 h; F direct + competence: 1.25 h. About 8.1k distinct examples seen (of 15.7-16k); mean complexity rises 0.3 -> 2.8 over training.
| E plan+comp | 65.3 (85.1/73.1/52.3/28.3) | 34.1 (42.6/23.9/11.8) | 51.7 | 50.4 | 54.3 | 84 / 241 |
| F direct+comp | 62.1 (87.1/66.4/56.3/19.3) | 32.7 (40.9/21.6/16.0) | 51.1 | 50.6 | 52.4 | 115 / 361 |
Competence curriculum keeps easy accuracy but under-trains hard/extra (fewer exposures). E > F on all five sets (Spider p = 0.022).

## 2026-10-30  E9-E10 seeds 2 and 3 for A and B (run_p3_more.sh); E11 oracle plan
| | Spider | BIRD | Syn | DK | Realistic |
| A s1/s2/s3 | 70.1/69.5/69.4 | 36.7/36.6/36.9 | 56.5/55.6/54.8 | 56.2/56.0/56.2 | 61.2/61.0/61.8 |
| B s1/s2/s3 | 71.9/72.3/70.2 | 37.7/38.7/39.2 | 59.1/59.6/57.5 | 56.0/54.9/54.9 | 59.3/61.4/59.4 |
| A mean +- sd | 69.7+-0.4 | 36.7+-0.2 | 55.6+-0.9 | 56.1+-0.1 | 61.3+-0.4 |
| B mean +- sd | 71.5+-1.1 | 38.5+-0.8 | 58.7+-1.1 | 55.3+-0.6 | 60.0+-1.2 |
| non-exec mean A / B | 74 / 45 | 241 / 179 | 139 / 75 | 48 / 42 | 41 / 33 |
Per-hardness 3-seed mean, Spider: A 89.1/70.7/62.6/45.4, B 86.6/75.3/63.0/47.4. BIRD: A 44.7/26.1/20.1, B 46.8/26.9/22.4.
McNemar exact (A vs B): seed1 Spider p=.14, BIRD .34, Syn .057; seed2 Spider .018*, BIRD .057, Syn .0016**; seed3 Spider .56, BIRD .030*, Syn .044*. DK/Realistic never significant.
E11 oracle plan (gold plan given to B, model writes SQL only): Spider 81.9 (92.3/81.4/79.3/70.5), BIRD 56.8 (65.2/45.0/40.3), Syn 74.1, DK 67.0, Realistic 71.7;
non-exec 18/94/50/18/16. All p < 1e-4 vs B. Each eval set 1-2 h; training 1.34-1.45 h per model.

## 2026-10-31  E12 plan voting (run_p3_vote.sh): greedy plan + 5 samples (T=0.7, top-p 0.95), majority structure (table set + flags), then greedy SQL
| V s1/s2/s3 | 72.1/72.1/69.9 | 37.8/38.7/39.3 | 59.4/59.5/57.5 | 56.6/55.4/53.9 | 60.2/61.2/58.7 |
| V mean | 71.4+-1.3 | 38.6+-0.8 | 58.8+-1.1 | 55.3+-1.4 | 60.0+-1.3 |
Agreement (seed 1): Spider dev 6/6 in 737 of 1,034 questions, >=4/6 in 965; BIRD 6/6 in 694 of 1,533, >=4/6 in 1,194.
Voting changes nothing: plan errors are systematic (same wrong plan under sampling), not random. ~4 s/question.

## 2026-10-30  Error analysis (analyze_errors_p3.py), seed 1, first structural difference from gold
Spider dev (A / B / V / oracle): no-such-column 63/35/37/18; wrong tables 68/72/68/3; nesting/set-op 25/33/33/12; grouping 14/12/13/2;
aggregate 11/12/11/1; wrong details 118/120/121/145; total wrong 309/291/289/187.
BIRD dev: no-such-column 207/154/148/68; wrong tables 242/261/258/8; nesting 53/51/53/32; grouping 24/26/25/2; aggregate 35/53/50/8;
filter 15/24/22/5; details 358/351/359/512; total 971/955/954/663.
Plan-first removes hallucinated columns (-44% Spider, -26% BIRD) and broken SQL; it does not reduce wrong-table errors. With the gold plan, wrong-table
errors almost vanish (68 -> 3, 242 -> 8) and the remaining errors are value/condition details.
