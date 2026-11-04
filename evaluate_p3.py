#!/usr/bin/env python
"""evaluate_p3.py  (Paper 3)  execution accuracy on SQLite, broken down by hardness, plus plan quality.
Usage: python evaluate_p3.py <pred.jsonl> [more pred.jsonl ...]  [--tag NAME]
Writes <pred>.eval.jsonl (per-question status) and <pred>.summary.json; prints a table.
Result-set comparison: ordered if the gold query has ORDER BY, otherwise as multisets; numbers rounded to 3 decimals.
"""
import argparse, json, os, re, sqlite3, collections, threading
import sqlglot
from sqlglot import exp

TIMEOUT = 30


def run(db, sql):
    con = sqlite3.connect(db); con.text_factory = lambda b: b.decode("utf-8", "replace")
    res = {}
    def kill(): con.interrupt()
    t = threading.Timer(TIMEOUT, kill); t.start()
    try:
        res["rows"] = con.execute(sql).fetchall()
    except Exception as e:
        res["err"] = f"{type(e).__name__}: {str(e)[:120]}"
    finally:
        t.cancel(); con.close()
    return res


def norm(v):
    if isinstance(v, float): return round(v, 3)
    if isinstance(v, str):
        try: return round(float(v), 3) if re.fullmatch(r"-?\d+(\.\d+)?", v) else v
        except ValueError: return v
    return v


def same(gold_rows, pred_rows, ordered):
    g = [tuple(norm(x) for x in r) for r in gold_rows]; p = [tuple(norm(x) for x in r) for r in pred_rows]
    if len(g) != len(p): return False
    if g and len(g[0]) != len(p[0]): return False
    if ordered: return g == p
    key = lambda r: tuple((str(type(x)), x) for x in r)
    return sorted(g, key=key) == sorted(p, key=key)


def plan_fields(plan):
    """Parse a plan text into {tables:set, flags:dict}. Tolerant to missing lines."""
    if not plan: return None
    d = {"tables": set(), "flags": {}}
    for line in plan.split("\n"):
        line = line.strip().lower()
        if line.startswith("tables:"): d["tables"] = {t.strip().strip("`\"'") for t in line[7:].split(",") if t.strip()}
        elif line.startswith("joins:"): continue
        else:   # flags: either one "k: v | k: v" line or one "k: v" per line; keep the first word of the value
            for part in line.split("|"):
                if ":" in part:
                    k, v = part.split(":", 1); v = v.strip().split(" ")[0].strip("(),.")
                    if v: d["flags"][k.strip()] = v
    return d


def plan_scores(gold_plan, pred_plan):
    g, p = plan_fields(gold_plan), plan_fields(pred_plan)
    if not g or not p: return None
    tp = len(g["tables"] & p["tables"]); prec = tp / max(1, len(p["tables"])); rec = tp / max(1, len(g["tables"]))
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    keys = ["where", "group by", "having", "order by", "limit", "nested", "set op", "distinct"]
    flag_acc = sum(g["flags"].get(k) == p["flags"].get(k) for k in keys) / len(keys)
    return {"table_f1": f1, "table_exact": g["tables"] == p["tables"], "flag_acc": flag_acc}


def evaluate(path):
    items = [json.loads(l) for l in open(path)]
    out, by = [], collections.defaultdict(lambda: [0, 0, 0])   # hardness -> [n, correct, errors]
    plan_tot = collections.defaultdict(float); plan_n = 0; plan_ok_given = [0, 0, 0, 0]  # [plan tables exact & sql ok, plan exact n, plan wrong & sql ok, plan wrong n]
    for it in items:
        ordered = bool(re.search(r"\bORDER\s+BY\b", it["sql"], re.I))
        g = run(it["db_path"], it["sql"])
        if "err" in g: status = "gold_error"
        else:
            p = run(it["db_path"], it["pred_sql"]) if it["pred_sql"] else {"err": "empty"}
            status = "error" if "err" in p else ("correct" if same(g["rows"], p["rows"], ordered) else "wrong")
        h = it.get("hardness", "all")
        if status != "gold_error":
            by[h][0] += 1; by[h][1] += status == "correct"; by[h][2] += status == "error"
        ps = plan_scores(it.get("plan"), it.get("pred_plan"))
        if ps:
            plan_n += 1
            for k, v in ps.items(): plan_tot[k] += float(v)
            if ps["table_exact"]: plan_ok_given[1] += 1; plan_ok_given[0] += status == "correct"
            else: plan_ok_given[3] += 1; plan_ok_given[2] += status == "correct"
        out.append({"id": it.get("id"), "hardness": h, "status": status, "err": p.get("err") if status == "error" else None, "plan": ps})
    with open(path.replace(".jsonl", ".eval.jsonl"), "w") as f:
        for o in out: f.write(json.dumps(o) + "\n")
    n = sum(v[0] for v in by.values()); c = sum(v[1] for v in by.values()); e = sum(v[2] for v in by.values())
    summ = {"file": path, "n": n, "ex": round(100 * c / max(1, n), 1), "errors": e, "gold_errors": sum(1 for o in out if o["status"] == "gold_error"),
            "by_hardness": {h: {"n": v[0], "ex": round(100 * v[1] / max(1, v[0]), 1), "errors": v[2]} for h, v in by.items()}}
    if plan_n:
        summ["plan"] = {"n": plan_n, "table_f1": round(100 * plan_tot["table_f1"] / plan_n, 1), "table_exact": round(100 * plan_tot["table_exact"] / plan_n, 1),
                        "flag_acc": round(100 * plan_tot["flag_acc"] / plan_n, 1),
                        "ex_when_tables_right": round(100 * plan_ok_given[0] / max(1, plan_ok_given[1]), 1),
                        "ex_when_tables_wrong": round(100 * plan_ok_given[2] / max(1, plan_ok_given[3]), 1)}
    json.dump(summ, open(path.replace(".jsonl", ".summary.json"), "w"), indent=1)
    return summ


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("preds", nargs="+"); ap.add_argument("--tag", default="")
    a = ap.parse_args()
    order = ["easy", "medium", "hard", "extra", "simple", "moderate", "challenging", "unlabeled"]
    for p in a.preds:
        if p.endswith(".eval.jsonl"): continue
        s = evaluate(p)
        hs = sorted(s["by_hardness"], key=lambda h: order.index(h) if h in order else 99)
        line = f"{os.path.basename(p):42s} EX {s['ex']:5.1f}  (n={s['n']}, non-exec={s['errors']})  " + "  ".join(f"{h}={s['by_hardness'][h]['ex']:.1f}" for h in hs)
        if "plan" in s: line += f"  | plan: tables-F1 {s['plan']['table_f1']}, flags {s['plan']['flag_acc']}, EX tables right/wrong {s['plan']['ex_when_tables_right']}/{s['plan']['ex_when_tables_wrong']}"
        print(line)
        if a.tag:
            with open("results/p3_results.txt", "a") as f: f.write(a.tag + "  " + line + "\n")
