#!/usr/bin/env python
"""make_p3_data.py  (Paper 3: structure-first generation)
Builds data/p3/*.jsonl from the unified Spider/BIRD files and the three Spider robustness sets.
Each line: source, id, db_id, db_path, question, evidence, schema, sql, hardness, plan
  hardness : Spider official (easy/medium/hard/extra) via evalsuite; BIRD keeps its own difficulty
  plan     : structural outline of the gold SQL (tables, joins, clauses, aggregates) extracted with sqlglot
Usage: python make_p3_data.py
"""
import json, os, sys, sqlite3, collections
import sqlglot
from sqlglot import exp

sys.path.insert(0, "evalsuite")
try:
    from process_sql import get_schema, Schema, get_sql
    from evaluation import Evaluator
    HAVE_EVAL = True
except Exception as e:
    print("evalsuite not importable, using heuristic hardness:", e); HAVE_EVAL = False

OUT = "data/p3"; os.makedirs(OUT, exist_ok=True)
_schema_cache = {}


def spider_hardness(sql, db_path):
    if HAVE_EVAL:
        try:
            if db_path not in _schema_cache: _schema_cache[db_path] = Schema(get_schema(db_path))
            return Evaluator.eval_hardness(None, get_sql(_schema_cache[db_path], sql))
        except Exception:
            pass
    return heuristic_hardness(sql)


def heuristic_hardness(sql):
    u = sql.upper(); n = 0
    n += u.count(" JOIN ") + u.count("GROUP BY") + u.count("ORDER BY") + u.count("HAVING") + (u.count("SELECT") - 1) * 2
    n += sum(u.count(k) for k in (" UNION ", " INTERSECT ", " EXCEPT ")) * 2 + u.count(" LIMIT ")
    return "easy" if n == 0 else "medium" if n <= 2 else "hard" if n <= 4 else "extra"


def make_plan(sql):
    """Structural outline of a SQL query. Returns None if sqlglot cannot parse it."""
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return None
    tables, joins, aggs, alias = [], [], [], {}
    for t in tree.find_all(exp.Table):
        n = t.name.lower()
        if n and n not in tables: tables.append(n)
        if t.alias: alias[t.alias.lower()] = n
    for j in tree.find_all(exp.Join):
        on = j.args.get("on")
        if on is not None:
            on = on.copy()
            for c in on.find_all(exp.Column):   # write join conditions with real table names, not T1/T2 aliases
                if c.table and c.table.lower() in alias: c.set("table", exp.to_identifier(alias[c.table.lower()]))
            s = on.sql(dialect="sqlite").lower()
            if s not in joins: joins.append(s)
    for a in tree.find_all(exp.AggFunc):
        n = a.sql_name().lower()
        if n not in aggs: aggs.append(n)
    setop = "none"
    for cls, name in ((exp.Union, "union"), (exp.Intersect, "intersect"), (exp.Except, "except")):
        if list(tree.find_all(cls)):
            setop = name; break
    n_select = len(list(tree.find_all(exp.Select)))
    n_setparts = 1 + sum(len(list(tree.find_all(c))) for c in (exp.Union, exp.Intersect, exp.Except))
    nested = "yes" if n_select > n_setparts else "no"
    flag = lambda c: "yes" if list(tree.find_all(c)) else "no"
    lines = ["tables: " + ", ".join(tables)]
    if joins: lines.append("joins: " + " ; ".join(joins))
    lines.append(f"where: {flag(exp.Where)} | group by: {flag(exp.Group)} | having: {flag(exp.Having)} | order by: {flag(exp.Order)} | limit: {flag(exp.Limit)}")
    lines.append(f"nested: {nested} | set op: {setop} | distinct: {flag(exp.Distinct)} | aggregates: " + (", ".join(aggs) if aggs else "none"))
    return "\n".join(lines)


def ddl(db_path, cache={}):
    if db_path not in cache:
        con = sqlite3.connect(db_path)
        cache[db_path] = "\n".join(r[0] + ";" for r in con.execute("SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL ORDER BY rowid"))
        con.close()
    return cache[db_path]


def write(name, rows):
    bad = sum(1 for r in rows if r["plan"] is None)
    rows = [r for r in rows if r["plan"] is not None]
    with open(f"{OUT}/{name}.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    dist = collections.Counter(r["hardness"] for r in rows)
    print(f"{name:14s} {len(rows):6d} questions  (plan unparsable: {bad})  " + ", ".join(f"{k}={v}" for k, v in sorted(dist.items())))


# --- Spider / BIRD from the unified files (Paper 1 format)
for src, fn in [("spider_train", "data/unified/spider_train.jsonl"), ("spider_dev", "data/unified/spider_dev.jsonl"),
                ("bird_train", "data/unified/bird_train.jsonl"), ("bird_dev", "data/unified/bird_dev.jsonl")]:
    if not os.path.exists(fn):
        print("missing", fn); continue
    rows = []
    for l in open(fn):
        it = json.loads(l)
        hard = spider_hardness(it["sql"], it["db_path"]) if src.startswith("spider") else (it.get("difficulty") or "unknown")
        rows.append({"source": src, "id": it.get("id", ""), "db_id": it["db_id"], "db_path": it["db_path"], "question": it["question"],
                     "evidence": it.get("evidence", ""), "schema": it["schema"], "sql": it["sql"], "hardness": hard, "plan": make_plan(it["sql"])})
    write(src, rows)

# --- robustness sets (all use the Spider dev databases)
dev_paths = {}
if os.path.exists("data/unified/spider_dev.jsonl"):
    for l in open("data/unified/spider_dev.jsonl"):
        it = json.loads(l); dev_paths[it["db_id"]] = it["db_path"]

ROBUST = [("spider_syn", "data/p3/Spider-Syn/Spider-Syn/dev.json"), ("spider_dk", "data/p3/Spider-DK/Spider-DK.json"),
          ("spider_realistic", "data/p3/spider-realistic.json")]
for src, fn in ROBUST:
    if not os.path.exists(fn):
        print("missing", fn); continue
    data = json.load(open(fn))
    if src == "spider_syn": print("  spider_syn keys:", sorted(data[0].keys()))
    rows, missing = [], set()
    for i, it in enumerate(data):
        db = it["db_id"]
        path = dev_paths.get(db) or (f"data/p3/Spider-DK/database/{db}/{db}.sqlite" if src == "spider_dk" else None)
        if not path or not os.path.exists(path):
            missing.add(db); continue
        q = it.get("SpiderSynQuestion") or it.get("question")
        sql = it.get("query") or it.get("SQL")
        rows.append({"source": src, "id": f"{src}_{i}", "db_id": db, "db_path": path, "question": q, "evidence": "",
                     "schema": ddl(path), "sql": sql, "hardness": spider_hardness(sql, path), "plan": make_plan(sql)})
    if missing: print(f"  {src}: databases not found: {sorted(missing)}")
    write(src, rows)

# --- one example plan, to see the format
ex = json.loads(open(f"{OUT}/spider_dev.jsonl").readline())
print("\nExample:\nQ:", ex["question"], "\nSQL:", ex["sql"], "\nPLAN:\n" + ex["plan"], "\nhardness:", ex["hardness"])
