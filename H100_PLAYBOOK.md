# H100 playbook: execute, do not author

A fail-fast run sheet for a rationed H100 session. Principle: cheap checks first
(so a showstopper shows in 5 min, not after an hour), then the pipeline. Use `tmux`
so vLLM, the agent, and the load test survive disconnects. Everything below is
copy-paste, the thinking was done during local development.

The long pole is the model download (about 60GB from HF), so kick off vLLM first
and do setup while it downloads.

---

## Pre-flight (cheap, do immediately)

```bash
# in the repo on the H100 box
git pull
bash scripts/check_env.sh      # what is present vs missing (driver, uv, docker, disk, ports)
cp .env.example .env           # then edit: LLM_BACKEND=h100  and  HF_TOKEN=hf_...
uv sync
uv run python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"  # must be True
uv run python -m agent.config  # must print profile=h100, model=...30B...  if not, fix .env first
```

Forward ports (VSCode Remote-SSH, host connected first): 8000, 8001, 9090, 3000, 3001.

Start the observability stack and data download in parallel with the model download:
```bash
docker compose up -d                       # Prometheus, Grafana, Langfuse
uv run python scripts/load_data.py         # BIRD subset (~500 MB)
```

---

## Serving: vLLM (start FIRST, it downloads while you set up)

Two one-time fixes before the first launch:
```bash
# (a) vLLM 0.10.2 crashes with transformers 5.x
#     ("Qwen2Tokenizer has no attribute all_special_tokens_extended"). Pin 4.x.
#     NOTE: use `uv add`, NOT `uv pip install`. `uv run` re-syncs the venv to the
#     lock on every call and would otherwise revert a bare pip install back to 5.x.
uv add 'transformers>=4.51,<5'
uv run python -c "import transformers; print(transformers.__version__)"   # expect 4.5x
# (b) vLLM does NOT read .env. Export HF_TOKEN so downloads are authenticated and faster:
export HF_TOKEN=hf_...            # or:  set -a; source .env; set +a
```

```bash
tmux new -s vllm
bash scripts/start_vllm.sh                 # initial flags already reasoned in the script
# watch the log until: "Application startup complete" / serving on :8000
```

**Gate 1, model responds** (do not proceed until this works):
```bash
curl -s http://localhost:8000/v1/models | python3 -m json.tool          # model listed?
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-30B-A3B-Instruct-2507","messages":[{"role":"user","content":"return the SQL: SELECT 1"}]}' \
  | python3 -m json.tool
```

**Gate 2, metrics exist and match the dashboard:**
```bash
curl -s http://localhost:8000/metrics | grep -E '^vllm:(e2e_request_latency_seconds_bucket|gpu_cache_usage_perc|kv_cache_usage_perc|num_preemptions_total|generation_tokens_total)' | head
```
If a name is missing, that is the only dashboard risk, and it is a 2-min `expr` fix.

Screenshot: `screenshots/vllm_manual_query.png` (vLLM serving plus the manual query
returning SQL). Record the final flags and one-line justifications in `REPORT.md`.

---

## Agent and Langfuse

```bash
tmux new -s agent
uv run uvicorn agent.server:app --host 0.0.0.0 --port 8001
# one end-to-end sanity call (the field is "db", not "db_id"):
curl -s -X POST http://localhost:8001/answer -H "Content-Type: application/json" \
  -d '{"question":"What is the coordinates location of the circuits for Australian grand prix?","db":"formula_1"}' | python3 -m json.tool
```

Langfuse: open `:3001`, sign up, create a project, copy the keys into `.env`, then
restart the agent. Fire about 10 questions (the eval below does this). Then in the
Langfuse UI:

Screenshot: `screenshots/langfuse_trace.png` (a trace showing generate, verify,
revise, use the `formula_1` question, it reliably revises).
Screenshot: `screenshots/langfuse_tags.png` (the trace list filtered by metadata
`source=eval`).

---

## Baseline eval (the real numbers)

```bash
uv run python evals/run_eval.py --out results/eval_baseline.json
```
Watch Grafana while it runs (about 60 requests, light and sequential load).
Screenshot: `screenshots/grafana_eval_run.png` (the dashboard during the eval).
Record the overall and per-iteration pass rate in `REPORT.md`.

The `grafana_serving.png` showcase (full dashboard reacting to a burst) is captured
later, during the load test, where the panels actually swing. The eval's gentle
load is a weak showcase for it.

---

## SLO load test and one tuning iteration

```bash
tmux new -s load
uv run python load_test/driver.py --rps 10 --duration 300    # the SLO target
```
Screenshot: `screenshots/grafana_before.png` (the dashboard during the baseline load).
Screenshot: `screenshots/grafana_serving.png` (full dashboard, all panels reacting to
this burst, 10 RPS swings the percentile, KV, and queue panels).

Diagnose from the dashboard (which metric moves first, queue, KV, TTFT, TPOT?),
change one flag in `start_vllm.sh`, restart vLLM, and re-run:
Screenshot: `screenshots/grafana_after.png` (the same view after the change).
```bash
uv run python evals/run_eval.py --out results/eval_after_tuning.json   # did quality survive?
```
Record the iteration log in `REPORT.md`: saw X, hypothesized Y, changed Z, result W.

---

## If time is short, minimum viable (in priority order)

1. Gate 1 and Gate 2 plus `vllm_manual_query.png` (serving config).
2. `eval_baseline.json` plus `grafana_serving.png` (eval and dashboard).
3. One load test plus one tuning iteration with before/after (the SLO investigation).
4. Langfuse screenshots, fast, do them while the eval runs.

Partial is fine if documented honestly in `REPORT.md`. A metric-grounded diagnosis
is worth more than a green checkmark.

---

## Committing results

`results/*.json` and `screenshots/*.png` are gitignored by default, so force-add
them:
```bash
git add -f results/*.json screenshots/*.png
git add -A && git commit -m "H100 run: baseline, tuning, screenshots, report" && git push origin main
```
