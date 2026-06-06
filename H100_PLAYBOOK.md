# H100 PLAYBOOK — execute, don't author

For the rationed H100 session. Principle: **cheap checks first** (so a
showstopper shows in 5 min, not after an hour), then the pipeline. Use `tmux`
so vLLM / agent / load-test survive disconnects. Everything below is
copy-paste; the thinking was done in Stage A.

The long pole is the **model download (~60 GB from HF)** — so kick off vLLM
*first* and do setup while it downloads.

---

## Pre-flight (cheap, do immediately)

```bash
# in the repo on the H100 VM
git pull
bash scripts/check_env.sh      # what's present vs missing (driver, uv, docker, disk, ports)
cp .env.example .env           # then edit: LLM_BACKEND=h100  and  HF_TOKEN=hf_...
uv sync
uv run python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"  # must be True
uv run python -m agent.config  # MUST print profile=h100, model=...30B...  -> if not, fix .env first
```

Forward ports (VSCode Remote-SSH, host connected first): 8000, 8001, 9090, 3000, 3001.

Start the o11y stack and data download in parallel with the model download:
```bash
docker compose up -d                       # Prometheus/Grafana/Langfuse
uv run python scripts/load_data.py         # BIRD subset (~500 MB)
```

---

## Phase 1 — vLLM (start FIRST; it downloads while you set up)

```bash
tmux new -s vllm
bash scripts/start_vllm.sh                 # initial flags already reasoned in the script
# watch the log until: "Application startup complete" / serving on :8000
```

**GATE 1 — model responds** (don't proceed until this works):
```bash
curl -s http://localhost:8000/v1/models | python3 -m json.tool          # model listed?
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-30B-A3B-Instruct-2507","messages":[{"role":"user","content":"return the SQL: SELECT 1"}]}' \
  | python3 -m json.tool
```

**GATE 2 — metrics exist & match the dashboard:**
```bash
curl -s http://localhost:8000/metrics | grep -E '^vllm:(e2e_request_latency_seconds_bucket|gpu_cache_usage_perc|kv_cache_usage_perc|num_preemptions_total|generation_tokens_total)' | head
```
If a name is missing, that's the only dashboard risk — tell me and it's a 2-min `expr` fix.

📸 **`screenshots/vllm_manual_query.png`** — vLLM serving + the manual query returning SQL.
✍️ Record the final flags + one-line justifications in `REPORT.md` §1.

---

## Phase 3/4 — agent + Langfuse

```bash
tmux new -s agent
uv run uvicorn agent.server:app --host 0.0.0.0 --port 8001
# one end-to-end sanity call (field is "db", not "db_id"):
curl -s -X POST http://localhost:8001/answer -H "Content-Type: application/json" \
  -d '{"question":"What is the coordinates location of the circuits for Australian grand prix?","db":"formula_1"}' | python3 -m json.tool
```

Langfuse: open `:3001`, sign up, create project, copy keys → `.env`, **restart the
agent**. Fire ~10 questions (the eval below does this). Then in the Langfuse UI:

📸 **`screenshots/langfuse_trace.png`** — a trace showing generate→verify→revise
(use the `formula_1` question; it reliably revises).
📸 **`screenshots/langfuse_tags.png`** — trace list with `source=eval` / `db_id` tags visible.

---

## Phase 5 — baseline eval (the real numbers)

```bash
uv run python evals/run_eval.py --out results/eval_baseline.json
```
Watch Grafana while it runs (~60 reqs).
📸 **`screenshots/grafana_eval_run.png`** — dashboard reacting during the eval.
📸 **`screenshots/grafana_serving.png`** — full dashboard reacting to load.
✍️ Copy overall + per-iteration pass rate into `REPORT.md` §2 & §4.

---

## Phase 6 — SLO load test + ONE tuning iteration

```bash
tmux new -s load
uv run python load_test/driver.py --rps 10 --duration 300    # the SLO target
```
📸 **`screenshots/grafana_before.png`** — dashboard during the baseline load.

Diagnose from the dashboard (which metric moves first? queue / KV / TTFT / TPOT?),
change **one** flag in `start_vllm.sh`, restart vLLM, re-run:
📸 **`screenshots/grafana_after.png`** — same view after the change.
```bash
uv run python evals/run_eval.py --out results/eval_after_tuning.json   # did quality survive?
```
✍️ `REPORT.md` §3 iteration log: *saw X → hypothesized Y → changed Z → result W*.

---

## If time is short — minimum viable (in priority order)
1. Phase 1 GATE 1 + 2 and `vllm_manual_query.png` (15% — serving config).
2. `eval_baseline.json` + `grafana_serving.png` (eval + dashboard, 30%).
3. One load test + one tuning iteration with before/after (25% — the big one).
4. Langfuse screenshots (5%) — fast, do them while the eval runs.

Skipped/partial is fine if documented honestly in `REPORT.md` — the grader
rewards a metric-grounded diagnosis over a green checkmark.

---

## Submission reminder
`results/*.json` and `screenshots/*.png` are gitignored. Force-add them:
```bash
git add -f results/*.json screenshots/*.png
git add -A && git commit -m "H100 run: baseline, tuning, screenshots, report" && git push origin main
```
