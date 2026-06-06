"""Eval runner using execution accuracy.

Reads evals/eval_set.jsonl, calls the agent at AGENT_URL on each question,
then compares the agent's SQL output to the gold SQL by *executed rows*
(canonicalized: sorted, stringified, None-coerced to empty).

Helpers (run_sql / canonicalize / matches) are provided. You implement
eval_one() and summarize().

Run:
    uv run python evals/run_eval.py --out results/eval_baseline.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVAL_FILE = ROOT / "evals" / "eval_set.jsonl"
DEFAULT_OUT_FILE = ROOT / "results" / "eval_baseline.json"
DB_DIR = ROOT / "data" / "bird"
AGENT_URL_DEFAULT = "http://localhost:8001/answer"


# ---------- Helpers (provided) -----------------------------------------

def run_sql(db_id: str, sql: str, timeout: float = 5.0) -> tuple[bool, list[tuple] | None, str | None]:
    """Run sql against db_id in read-only mode. Returns (ok, rows, error)."""
    path = DB_DIR / f"{db_id}.sqlite"
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=timeout) as conn:
            cur = conn.execute(sql)
            rows = cur.fetchall()
            return True, rows, None
    except Exception as e:  # noqa: BLE001
        return False, None, f"{type(e).__name__}: {e}"


def canonicalize(rows: list[tuple] | None) -> list[tuple] | None:
    """Sort rows; coerce cells to str; None -> ''."""
    if rows is None:
        return None
    return sorted(tuple("" if c is None else str(c) for c in row) for row in rows)


def matches(gold_rows: list[tuple] | None, pred_rows: list[tuple] | None) -> bool:
    if gold_rows is None or pred_rows is None:
        return False
    return canonicalize(gold_rows) == canonicalize(pred_rows)


# ---------- Implement these (Phase 5) ----------------------------------

def _candidate_sqls(history: list[dict], final_sql: str) -> list[str]:
    """Ordered SQL the agent produced: generate_sql first, then each revise.

    The /answer history records one entry per node; generate_sql and revise
    both carry a "sql". That sequence IS the per-iteration candidate list
    (index 0 = iter 0 = first generate, index k = after the k-th revise).
    Falls back to [final_sql] if history is missing/empty.
    """
    cands = [h["sql"] for h in history if h.get("node") in ("generate_sql", "revise") and h.get("sql")]
    return cands or ([final_sql] if final_sql else [])


def eval_one(question: dict, agent_url: str) -> dict:
    """Score one question by execution accuracy at every iteration.

    Calls the agent, runs the gold SQL once, then runs each candidate SQL the
    agent emitted and compares canonicalized row sets. per_iter_correct[k] is
    whether the agent's k-th attempt would have been correct if we'd stopped
    there.
    """
    db_id = question["db_id"]
    gold_sql = question["gold_sql"]

    # 1) Ask the agent. Tags flow through to Langfuse metadata (Phase 4/6).
    payload = {
        "question": question["question"],
        "db": db_id,
        "tags": {"source": "eval", "db_id": db_id},
    }
    t0 = time.monotonic()
    try:
        resp = httpx.post(agent_url, json=payload, timeout=120.0)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:  # noqa: BLE001
        return {
            "db_id": db_id,
            "question": question["question"],
            "gold_sql": gold_sql,
            "final_sql": "",
            "iterations": 0,
            "agent_ok": False,
            "agent_error": f"{type(e).__name__}: {e}",
            "per_iter_sql": [],
            "per_iter_correct": [],
            "final_correct": False,
            "latency_seconds": time.monotonic() - t0,
            "error": f"agent call failed: {type(e).__name__}: {e}",
        }
    latency = time.monotonic() - t0

    final_sql = data.get("sql", "")
    history = data.get("history", []) or []
    candidates = _candidate_sqls(history, final_sql)

    # 2) Gold rows, once.
    gold_ok, gold_rows, gold_err = run_sql(db_id, gold_sql)

    # 3) Score each candidate against gold.
    per_iter_correct: list[bool] = []
    for sql in candidates:
        if not gold_ok:
            per_iter_correct.append(False)
            continue
        pred_ok, pred_rows, _ = run_sql(db_id, sql)
        per_iter_correct.append(pred_ok and matches(gold_rows, pred_rows))

    return {
        "db_id": db_id,
        "question": question["question"],
        "gold_sql": gold_sql,
        "final_sql": final_sql,
        "iterations": data.get("iterations", len(candidates)),
        "agent_ok": data.get("ok", False),
        "agent_error": data.get("error"),
        "gold_error": None if gold_ok else gold_err,
        "per_iter_sql": candidates,
        "per_iter_correct": per_iter_correct,
        "final_correct": per_iter_correct[-1] if per_iter_correct else False,
        "latency_seconds": latency,
        "error": None,
    }


def summarize(results: list[dict]) -> dict:
    """Aggregate per-question results.

    Per-iteration carry-forward: if the agent terminated at iteration j < k
    (verify said ok at j, or it hit MAX_ITERATIONS at j < k), treat the
    question's iteration-k result as identical to its iteration-j result.
    The agent stopped emitting; whatever it had at termination is what
    would have been served had we polled at iteration k.
    """
    n = len(results)
    if n == 0:
        return {"n_questions": 0, "overall_pass_rate": 0.0, "pass_rate_by_iteration": []}

    max_iters = max((len(r["per_iter_correct"]) for r in results), default=0)
    max_iters = max(max_iters, 1)

    pass_rate_by_iteration = []
    for k in range(max_iters):
        n_correct = 0
        for r in results:
            seq = r["per_iter_correct"] or [False]
            idx = min(k, len(seq) - 1)  # carry-forward terminated runs
            if seq[idx]:
                n_correct += 1
        pass_rate_by_iteration.append({
            "iteration": k,
            "n_correct": n_correct,
            "pass_rate": round(n_correct / n, 4),
        })

    n_final_correct = sum(1 for r in results if r.get("final_correct"))
    n_revised = sum(1 for r in results if len(r.get("per_iter_sql", [])) > 1)
    n_agent_errors = sum(1 for r in results if r.get("error") or r.get("agent_error"))
    iter_counts = [max(len(r.get("per_iter_sql", [])), 1) for r in results]
    latencies = [r["latency_seconds"] for r in results if r.get("latency_seconds") is not None]

    return {
        "n_questions": n,
        "overall_pass_rate": round(n_final_correct / n, 4),
        "n_correct": n_final_correct,
        "pass_rate_by_iteration": pass_rate_by_iteration,
        "questions_that_revised": n_revised,
        "avg_iterations": round(sum(iter_counts) / n, 3),
        "max_iterations_observed": max_iters,
        "agent_errors": n_agent_errors,
        "mean_latency_seconds": round(sum(latencies) / len(latencies), 3) if latencies else None,
    }


# ---------- Main (provided) --------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_FILE)
    parser.add_argument("--agent-url", default=AGENT_URL_DEFAULT)
    args = parser.parse_args()

    questions = [json.loads(line) for line in args.eval_set.read_text().splitlines() if line.strip()]
    print(f"Loaded {len(questions)} eval questions from {args.eval_set}")

    results: list[dict] = []
    t0 = time.monotonic()
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['db_id']}: {q['question'][:60]}...", flush=True)
        results.append(eval_one(q, args.agent_url))
    elapsed = time.monotonic() - t0

    summary = summarize(results)
    out = {
        "summary": summary,
        "wall_clock_seconds": elapsed,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
