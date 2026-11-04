#!/usr/bin/env python
"""infer_p3.py  (Paper 3)  greedy inference, direct SQL or plan-first.
Usage: python infer_p3.py <model> <in.jsonl> <out.jsonl> [adapter|none] [--mode direct|plan] [--max-new 320] [--limit N]
  direct : the model answers with the SQL only
  plan   : the model answers with  PLAN: <outline>  then  SQL: <query>   (structure-first)
Output line = input line + pred_sql, pred_plan, raw.
"""
import argparse, json, re, sys, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

PLAN_FORMAT = ("tables: <comma-separated tables used>\n"
               "joins: <join conditions, or omit this line if none>\n"
               "where: yes/no | group by: yes/no | having: yes/no | order by: yes/no | limit: yes/no\n"
               "nested: yes/no | set op: none/union/intersect/except | distinct: yes/no | aggregates: <count, avg, ... or none>")

SYS_DIRECT = "You are an expert SQLite engineer. Given a database schema and a question, write one correct SQLite query. Output only the SQL."
SYS_PLAN = ("You are an expert SQLite engineer. Given a database schema and a question, first write a short structural plan of the query, "
            "then the SQLite query. Use exactly this format:\nPLAN:\n" + PLAN_FORMAT + "\nSQL:\n<the query>")


def build_messages(it, mode):
    u = "Database schema:\n" + it["schema"]
    if it.get("evidence"): u += "\n\nEvidence: " + it["evidence"]
    u += "\n\nQuestion: " + it["question"]
    return [{"role": "system", "content": SYS_PLAN if mode == "plan" else SYS_DIRECT}, {"role": "user", "content": u}]


def target_text(it, mode):
    """Gold completion used for training."""
    return ("PLAN:\n" + it["plan"] + "\nSQL:\n" + it["sql"].strip()) if mode == "plan" else it["sql"].strip()


def parse_output(text, mode):
    t = text.strip()
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.S).strip()
    plan = None
    if mode == "plan":
        m = re.search(r"PLAN:\s*(.*?)\s*SQL:\s*(.*)", t, flags=re.S)
        if m: plan, t = m.group(1).strip(), m.group(2).strip()
        else:
            m2 = re.search(r"SQL:\s*(.*)", t, flags=re.S)
            if m2: t = m2.group(1).strip()
    m = re.search(r"```(?:sql)?\s*(.*?)```", t, flags=re.S)
    if m: t = m.group(1).strip()
    m = re.search(r"(SELECT\b.*|WITH\b.*)", t, flags=re.S | re.I)
    sql = (m.group(1) if m else t).strip().rstrip(";").split(";")[0].strip()
    return " ".join(sql.split()), plan


def plan_key(p):
    """Structural identity of a plan: table set + clause flags (join text and aggregate names ignored)."""
    tabs, flags = (), []
    for line in (p or "").split("\n"):
        line = line.strip().lower()
        if line.startswith("tables:"): tabs = tuple(sorted(t.strip() for t in line[7:].split(",") if t.strip()))
        elif line.startswith("joins:") or line.startswith("aggregates"): continue
        else:
            for part in line.split("|"):
                k, _, v = part.partition(":"); k, v = k.strip(), v.strip().split(" ")[0] if v.strip() else ""
                if k and k != "aggregates": flags.append((k, v))
    return (tabs, tuple(sorted(flags)))


def cut_plan(text):
    t = text.split("SQL:")[0].strip()
    return t[5:].strip() if t.startswith("PLAN:") else t


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("inp"); ap.add_argument("out"); ap.add_argument("adapter", nargs="?", default="none")
    ap.add_argument("--mode", choices=["direct", "plan"], default="direct"); ap.add_argument("--max-new", type=int, default=320)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--oracle-plan", action="store_true", help="plan mode only: insert the GOLD plan and let the model write only the SQL")
    ap.add_argument("--plan-vote", type=int, default=0, help="plan mode: sample N plans (+ the greedy plan), keep the structure most of them agree on, then write the SQL greedily")
    ap.add_argument("--temp", type=float, default=0.7)
    a = ap.parse_args()
    items = [json.loads(l) for l in open(a.inp)]
    if a.limit: items = items[:a.limit]
    tok = AutoTokenizer.from_pretrained(a.model); tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda")
    if a.adapter != "none":
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter); print("adapter", a.adapter)
    model.eval()
    t0 = time.time(); n_plan = 0
    with open(a.out, "w") as f:
        for i, it in enumerate(items):
            kw = {"enable_thinking": False} if "qwen3" in a.model.lower() else {}
            votes = None
            if a.plan_vote:
                base = tok.apply_chat_template(build_messages(it, "plan"), add_generation_prompt=True, tokenize=False, **kw) + "PLAN:\n"
                penc = tok(base, return_tensors="pt", add_special_tokens=False).to("cuda")
                L = penc["input_ids"].shape[1]; gkw = dict(max_new_tokens=160, stop_strings=["SQL:"], tokenizer=tok, pad_token_id=tok.pad_token_id or tok.eos_token_id)
                with torch.no_grad():
                    g = model.generate(**penc, do_sample=False, **gkw)
                    smp = model.generate(**penc, do_sample=True, temperature=a.temp, top_p=0.95, num_return_sequences=a.plan_vote, **gkw)
                greedy = cut_plan(tok.decode(g[0][L:], skip_special_tokens=True))
                cands = [greedy] + [cut_plan(tok.decode(x[L:], skip_special_tokens=True)) for x in smp]
                from collections import Counter
                cnt = Counter(plan_key(c) for c in cands); top = max(cnt.values())
                best = plan_key(greedy) if cnt[plan_key(greedy)] == top else next(k for k in cnt if cnt[k] == top)
                chosen = next(c for c in cands if plan_key(c) == best)
                votes = f"{cnt[best]}/{len(cands)}"
                text = base + chosen + "\nSQL:\n"
                enc = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
            elif a.oracle_plan:
                prefix = "PLAN:\n" + it["plan"] + "\nSQL:\n"
                text = tok.apply_chat_template(build_messages(it, "plan"), add_generation_prompt=True, tokenize=False, **kw) + prefix
                enc = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
            else:
                enc = tok.apply_chat_template(build_messages(it, a.mode), add_generation_prompt=True, return_dict=True, return_tensors="pt", **kw).to("cuda")
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=a.max_new, do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
            raw = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            if a.plan_vote: sql, plan = parse_output(raw, "direct")[0], chosen
            elif a.oracle_plan: sql, plan = parse_output(raw, "direct")[0], it["plan"]
            else: sql, plan = parse_output(raw, a.mode)
            n_plan += plan is not None
            it = dict(it); it.update({"pred_sql": sql, "pred_plan": plan, "raw": raw, "mode": a.mode, "plan_votes": votes})
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
            if (i + 1) % 100 == 0: print(f"{i + 1}/{len(items)}  {(time.time() - t0) / (i + 1):.2f}s/q", flush=True)
    print(f"done {len(items)} in {(time.time() - t0) / 60:.1f} min; plans parsed: {n_plan}; VRAM {torch.cuda.max_memory_allocated() / 1e9:.1f} GB -> {a.out}")
