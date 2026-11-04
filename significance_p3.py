#!/usr/bin/env python
"""significance_p3.py  (Paper 3)  paired exact McNemar test between two systems on the same question set.
Usage: python significance_p3.py <sysA_prefix> <sysB_prefix> [sets...]
  prefixes like  preds/p3/A_direct   preds/p3/B_plan   ->  reads <prefix>_<set>.eval.jsonl
Default sets: spider_dev bird_dev spider_syn spider_dk spider_realistic
Prints, per set: n, EX of each, b (A right, B wrong), c (A wrong, B right), exact two-sided p-value.
Rows are matched by line order (both files come from the same input file); hardness is checked to be equal.
"""
import json, sys
from math import comb

SETS = ["spider_dev", "bird_dev", "spider_syn", "spider_dk", "spider_realistic"]


def load(p):
    return [json.loads(l) for l in open(p)]


def mcnemar_exact(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


if __name__ == "__main__":
    pa, pb = sys.argv[1], sys.argv[2]; sets = sys.argv[3:] or SETS
    print(f"{'set':18s} {'n':>5s} {'EX_A':>6s} {'EX_B':>6s} {'b':>5s} {'c':>5s}  p-value")
    for s in sets:
        A, B = load(f"{pa}_{s}.eval.jsonl"), load(f"{pb}_{s}.eval.jsonl")
        assert len(A) == len(B), f"{s}: different lengths {len(A)} vs {len(B)}"
        n = b = c = ca = cb = 0
        for x, y in zip(A, B):
            assert x["hardness"] == y["hardness"], f"{s}: rows not aligned"
            if x["status"] == "gold_error" or y["status"] == "gold_error": continue
            n += 1; ra, rb = x["status"] == "correct", y["status"] == "correct"
            ca += ra; cb += rb; b += ra and not rb; c += rb and not ra
        p = mcnemar_exact(b, c)
        star = "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"{s:18s} {n:5d} {100 * ca / n:6.1f} {100 * cb / n:6.1f} {b:5d} {c:5d}  {p:.4f} {star}")
