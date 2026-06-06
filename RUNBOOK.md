# RUNBOOK — local development on the Ubuntu VM

This is the operational guide for running everything we built in **Stage A**
against **Nebius** (hosted Qwen3-30B) on your Ubuntu 24.04 dev box — no GPU
needed. Stage B (the H100 run) is one `.env` switch away and is covered at the
end.

> The agent is just an OpenAI-compatible HTTP client. The single switch that
> decides where its LLM calls go is `LLM_BACKEND` in `.env` (see `agent/config.py`).

---

## 0. Prereqs (Ubuntu 24.04)

```bash
sudo apt-get update && sudo apt-get install -y python3-dev git curl
# Docker + compose plugin (if not present): https://docs.docker.com/engine/install/ubuntu/
# uv: https://docs.astral.sh/uv/getting-started/installation/
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Docker note: the Langfuse stack is heavy (postgres + clickhouse + redis +
minio + 2 langfuse services). Give Docker **≥ 8 GB RAM** or clickhouse will
OOM-loop.

---

## 1. Phase 0 — setup

```bash
git clone git@github.com:emanuelam00/h100-inference-and-observability.git
cd h100-inference-and-observability

cp .env.example .env
#  -> edit .env:  set LLM_BACKEND=nebius  and  NEBIUS_API_KEY=<your key>

uv sync                              # installs deps from uv.lock
docker compose up -d                 # Prometheus, Grafana, Langfuse stack
uv run python scripts/load_data.py   # downloads BIRD subset (~500 MB) -> data/bird/
```

Confirm the backend resolves before anything else:

```bash
uv run python -m agent.config
# -> LLM backend: profile=nebius model=Qwen/Qwen3-30B-A3B-Instruct-2507 base_url=...tokenfactory... key=****...****
```

Sanity-check the UIs (forward the ports in VSCode Remote, or `-L` over SSH):
Prometheus `:9090`, Grafana `:3000` (admin/admin), Langfuse `:3001`.

---

## 2. Phase 3 — run & test the agent

```bash
uv run uvicorn agent.server:app --host 0.0.0.0 --port 8001
```

In another shell, ask it something (pick real questions from the eval set):

```bash
# look at a few questions:
head -n 3 evals/eval_set.jsonl

curl -s -X POST http://localhost:8001/answer \
  -H "Content-Type: application/json" \
  -d '{"question": "<paste a question>", "db": "<paste its db_id>"}' | python3 -m json.tool
```

The response includes `history` — that's the `generate_sql → verify → (revise)`
trail. **Checkpoint:** find at least one question where `history` contains a
`revise` entry (verify said `ok:false` and the loop re-ran). A good way to force
one is a question whose obvious first query returns zero rows or the wrong
column. Note which question triggered it — you'll cite it in the report.

---

## 3. Phase 4 — Langfuse tracing

1. Open `http://localhost:3001`, sign up (local, instant), create/confirm a project.
2. Settings → API Keys → create. Copy public + secret keys into `.env`:
   ```
   LANGFUSE_PUBLIC_KEY=pk-...
   LANGFUSE_SECRET_KEY=sk-...
   LANGFUSE_HOST=http://localhost:3001
   ```
3. Restart the agent server (it reads the keys at startup and attaches the
   callback handler — already wired in `agent/server.py`).
4. Fire ~10 questions (loop the curl above, or just run the eval in step 4).
5. In Langfuse you should see each run as a trace with `generate_sql`, `verify`,
   and sometimes `revise` as nested spans (prompt, response, latency, tokens).
   Tags `source=eval` / `db_id=...` are attached for Phase 6 filtering.

---

## 4. Phase 5 — eval harness (dev validation)

With the agent running:

```bash
uv run python evals/run_eval.py --out results/eval_baseline.json
cat results/eval_baseline.json | python3 -m json.tool | head -40
```

Look at `pass_rate_by_iteration`: if iter 0 ≈ iter 2, the loop isn't earning its
keep; if iter 2 > iter 0, it is. **These dev numbers are throwaway** — the real
baseline must come from the H100 (Stage B). This step only proves the harness
works end-to-end.

---

## 5. Phase 2 — dashboard (local, with the mock exporter)

The dev box can't run vLLM, so use the mock to give Prometheus something to
scrape on `:8000` (the exact port + metric names real vLLM uses):

```bash
uv run python scripts/mock_vllm_metrics.py        # serves :8000/metrics
```

- Open Grafana `:3000` → the **vLLM serving** dashboard (auto-provisioned).
- Every panel should be moving within ~15s. To make latency/queue/KV/preemptions
  jump for a screenshot: `curl http://localhost:8000/burst` (spikes load ~45s).
- This validates the queries, units, percentiles and thresholds. The numbers are
  fake; the wiring is real. On the H100 you stop this and start real vLLM on the
  same port — the dashboard is unchanged.

---

## Stage A.5 (optional) — real vLLM metrics on CPU

To validate the dashboard against a *real* vLLM exporter (not the mock), run a
tiny model on CPU. This is plumbing-only — the model's SQL quality is irrelevant.

```bash
# CPU vLLM install: https://docs.vllm.ai/en/latest/getting_started/installation/cpu.html
# then serve a small stand-in on :8000
uv run python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-0.6B --port 8000 --device cpu
```

Set `LLM_BACKEND=cpu` in `.env` if you also want the agent to talk to it.

---

## Stage B — the H100 run (numbers that count)

On the H100 VM, only the LLM backend changes:

1. In `.env`: `LLM_BACKEND=h100` and set `HF_TOKEN` (so vLLM can pull the model).
2. Start vLLM with your chosen flags: `bash scripts/start_vllm.sh` (Phase 1).
3. Stop the mock exporter; Prometheus now scrapes real vLLM on `:8000`. Dashboard
   reacts to real load.
4. Re-run the eval → `results/eval_baseline.json` (the real baseline).
5. Phase 6 load test → `load_test/driver.py --rps 10 --duration 300`, diagnose,
   iterate, save `results/eval_after_tuning.json`.

No agent/eval/dashboard code changes between Stage A and Stage B.

---

## Git

Remotes are already set: `origin` = your repo, `upstream` = course repo.
To push Stage A:

```bash
git add -A
git commit -m "Stage A: backend config, agent loop, eval harness, dashboard + mock exporter"
git push -u origin main
```

> ⚠️ **Submission reminder (later):** `.gitignore` currently excludes
> `results/*.json` and `screenshots/*.png`, but those are required deliverables.
> When you submit, force-add them: `git add -f results/*.json screenshots/*.png`.

---

## Quick port reference

| Port | Service        |
|------|----------------|
| 8000 | vLLM (or mock exporter) |
| 8001 | agent server   |
| 9090 | Prometheus     |
| 3000 | Grafana (admin/admin) |
| 3001 | Langfuse       |
