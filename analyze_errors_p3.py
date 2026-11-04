#!/usr/bin/env python
"""analyze_errors_p3.py  (Paper 3)  classify every wrong prediction by the first structural difference from the gold query.
Usage: python analyze_errors_p3.py <model_prefix>... [--sets spider_dev bird_dev] [--out results/p3_errors.json]
  model_prefix e.g. preds/p3/A_direct  -> reads <prefix>_<set>.jsonl and <prefix>_<set>.eval.jsonl
Categories (checked in this order, first match wins):
  exec:no-such-column | exec:no-such-table | exec:syntax | exec:other      (query does not run)
  wrong tables          table set differs from gold
  wrong nesting/set op  subquery or UNION/INTERSECT/EXCEPT differs
  wrong grouping        GROUP BY / HAVING presence differs
  wrong ordering/limit  ORDER BY / LIMIT presence differs
  wrong aggregate       aggregate functions differ
  wrong filter          WHERE presence differs
  wrong details         same structure, different columns / conditions / values
Also writes up to 3 examples per category per model to results/p3_error_examples.txt
"""
import argparse, json, collections, os
import sqlglot
from sqlglot import exp


def struct(sql):
    try:
        t = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return None
    has = lambda c: bool(list(t.find_all(c)))
    setop = next((n for c, n in ((exp.Union, "union"), (exp.Intersect, "intersect"), (exp.Except, "except")) if has(c)), "none")
    n_sel = len(list(t.find_all(exp.Select)))
    n_parts = 1 + sum(len(list(t.find_all(c))) for c in (exp.Union, exp.Intersect, exp.Except))
    return {"tables": frozenset(x.name.lower() for x in t.find_all(exp.Table) if x.name),
            "nest": (n_sel > n_parts, setop),
            "group": (has(exp.Group), has(exp.Having)),
            "order": (has(exp.Order), has(exp.Limit)),
            "agg": frozenset(a.sql_name().lower() for a in t.find_all(exp.AggFunc)),
            "where": has(exp.Where)}


def category(gold, pred, status, err):
    if status == "error":
        e = (err or "").lower()
        if "no such column" in e: return "exec:no-such-column"
        if "no such table" in e: return "exec:no-such-table"
        if "syntax" in e or "incomplete" in e or "unrecognized" in e or "empty" in e: return "exec:syntax"
        return "exec:other"
    g, p = struct(gold), struct(pred)
    if g is None or p is None: return "wrong details"
    for key, name in (("tables", "wrong tables"), ("nest", "wrong nesting/set op"), ("group", "wrong grouping"),
                      ("order", "wrong ordering/limit"), ("agg", "wrong aggregate"), ("where", "wrong filter")):
        if g[key] != p[key]: return name
    return "wrong details"


CATS = ["exec:no-such-column", "exec:no-such-table", "exec:syntax", "exec:other", "wrong tables", "wrong nesting/set op",
        "wrong grouping", "wrong ordering/limit", "wrong aggregate", "wrong filter", "wrong details"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("models", nargs="+")
    ap.add_argument("--sets", nargs="+", default=["spider_dev", "bird_dev"]); ap.add_argument("--out", default="results/p3_errors.json")
    a = ap.parse_args()
    res = {}; ex_lines = []
    for s in a.sets:
        for m in a.models:
            name = os.path.basename(m)
            P = [json.loads(l) for l in open(f"{m}_{s}.jsonl")]; E = [json.loads(l) for l in open(f"{m}_{s}.eval.jsonl")]
            cnt = collections.Counter(); hard = collections.defaultdict(collections.Counter); shown = collections.Counter()
            for it, ev in zip(P, E):
                if ev["status"] in ("correct", "gold_error"): continue
                c = category(it["sql"], it["pred_sql"], ev["status"], ev.get("err"))
                cnt[c] += 1; hard[ev["hardness"]][c] += 1
                if shown[c] < 3:
                    shown[c] += 1
                    ex_lines.append(f"[{name} | {s} | {c}]\nQ: {it['question']}\nGOLD: {it['sql']}\nPRED: {it['pred_sql']}\n")
            res[f"{name}|{s}"] = {"total_wrong": sum(cnt.values()), "by_category": dict(cnt), "by_hardness": {h: dict(v) for h, v in hard.items()}}
        # table for this set
        print(f"\n=== {s}: wrong predictions by category")
        names = [os.path.basename(m) for m in a.models]
        print(f"{'category':24s}" + "".join(f"{n[:14]:>15s}" for n in names))
        for c in CATS + ["TOTAL"]:
            row = [res[f"{n}|{s}"]["total_wrong"] if c == "TOTAL" else res[f"{n}|{s}"]["by_category"].get(c, 0) for n in names]
            print(f"{c:24s}" + "".join(f"{v:15d}" for v in row))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    open(a.out.replace(".json", "_examples.txt"), "w").write("\n".join(ex_lines))
    print("\nwritten", a.out, "and", a.out.replace(".json", "_examples.txt"))
