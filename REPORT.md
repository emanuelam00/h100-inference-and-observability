# LLM inference + observability — REPORT

Text-to-SQL PoC on BIRD-bench. `Qwen/Qwen3-30B-A3B-Instruct-2507` served with
vLLM 0.10.2 on 1× H100 80 GB; a LangGraph `generate → execute → verify → revise`
agent; Prometheus/Grafana on vLLM `/metrics` + Langfuse on the agent traces.
SLO target: **P95 end-to-end agent latency < 5 s at ≥ 10 RPS over 5 min.**

---

## 1. Serving configuration (Phase 1)

Workload: ~0.9–1.5 K-token prompts (schema + question; measured p50 918, p95
1414, max 1435 tokens for generate), short SQL outputs, ~2–3 dependent LLM calls
per agent run. Model is a 30 B **MoE** (3.3 B active). Flags in `scripts/start_vllm.sh`:

| Flag | Value | Justification |
|---|---|---|
| `--dtype bfloat16` | bf16 | H100-native; full 30 B MoE weights (56.9 GiB) fit 80 GB without quantization. |
| `--max-model-len` | 8192 | Generous headroom over the measured ~1.5 K-token prompts; smaller len would free KV but KV was never the constraint (see §3). |
| `--gpu-memory-utilization` | 0.90 | Maximize KV while leaving room for activations. Left 12.8 GiB KV (~140 K tokens). |
| `--max-num-seqs` | 256 | High concurrency for throughput; real prompts are small so effective concurrency reached ~37 with KV still ~15 %. |
| `--max-num-batched-tokens` | 8192 | Per-step batching budget balancing prefill vs decode. |
| `--enable-prefix-caching` | on | The DB schema + system prompt are reused across the 2–3 calls per run and across same-DB requests — a direct TTFT win for this workload. |

**Key startup observation (used in §3):** weights take 56.9 GiB, leaving only
12.8 GiB KV = ~17× concurrency *at full 8192-token context*. In practice prompts
are ~1.5 K tokens, so KV stayed ~15 % even under load — **abundant headroom; the
serving layer is not memory-bound.**

---

## 2. Baseline eval results (Phase 5)

`results/eval_baseline.json` — 30 BIRD questions, execution accuracy (canonicalized row sets):

- **Overall: 40.0 %** (12/30), 0 agent errors, mean 1.0 s/run.
- **Pass rate by iteration:** iter 0 **36.7 %** → iter 1 **40.0 %** → iter 2 **40.0 %**.
- 11/30 questions triggered a revise; mean 1.6 generate/revise calls.

Commentary: the verify→revise loop lifts pass rate by one question (+3.3 pts) at
the first revise and **nothing** at the second — the loop earns its keep, but
marginally, and the 3rd iteration is wasted (see §4).

---

## 3. Hitting the SLO (Phase 6)

**Baseline load test** (`--rps 10 --duration 300`, default single-process agent):

| metric | value | vs SLO |
|---|---|---|
| P50 | 69 s | — |
| **P95** | **83 s** | **16× over (target 5 s)** |
| P99 | 90 s | — |
| errors | 379 HTTP + 14 timeout (≈13 %) | — |

The dashboard during this run looked *healthy*: `waiting (queue) = 0` the whole
time, ~30 calls/s completed, KV ~15 %, preemptions 0, per-**call** p99 ~5 s. That
contradiction is the whole diagnosis: vLLM had headroom, yet agent-**run** P95 was
83 s — so the time was being spent **above** the model.

### Iteration log

**1. saw** vLLM queue=0 / KV 15 % yet run-P95 83 s and achieved RPS capped ~8
→ **hypothesized** the bottleneck is the agent server, not vLLM: the `/answer`
handler is synchronous (`def` + blocking `graph.invoke`) so FastAPI runs it in a
~40-thread pool; at 10 runs/s the pool saturates and runs queue *inside the agent*
→ **changed** to `uvicorn --workers 4` → **result:** P95 **83 s → 10.2 s**, P50
69 s → 2.2 s (8×/31×). The vLLM dashboard was *unchanged* — confirming vLLM was
never the constraint. SLO went from 16× to ~2× over.

**2. saw** `http_errors` = exactly **379 in every run** (identical → deterministic,
not load) → **hypothesized** context-length overflow → **disproved** by direct
measurement (max prompt 1435 tokens « 8192); then hypothesized missing DBs →
**disproved** (0/1500 pool questions lack a DB); then **instrumented the actual
exception** → root cause: the *provided* `agent/schema.py` crashes
(`AttributeError: NoneType.replace`) on foreign keys where SQLite leaves the
target column NULL — hits 2 of 11 DBs (`european_football_2`,
`debit_card_specializing`), 193 questions = the 12.6 % → **changed** `render_schema`
to guard the NULL → **result:** load-test `http_errors` **379 → 0**, success rate
99.6 %. These errors were *invalid-input* (schema render), not overload; true
serving error rate under load is ~0.

**Final configuration** (`--workers 4` + schema fix), `results/load_test_after2.json`:
P50 **2.27 s**, **P95 10.0 s**, P99 15.2 s, **0 HTTP errors**, 99.6 % success.

**Quality survived** (`results/eval_after_tuning.json`): **43.3 %** (13/30) vs 40 %
baseline — within run-to-run noise, no regression (the tuning was serving-side +
a bug fix, not a prompt/model change).

**Verdict — SLO missed on latency, honestly.** P95 10.0 s vs 5 s (~2×), though
**P50 (2.27 s) clears it**. The residual is structural, not a serving misconfig:
each agent run is **2–4 sequential 30 B calls** (per-call p95 ~2.7 s), so the run
tail is inherently several seconds. KV/preemptions/queue all stayed healthy
throughout — there is no serving knob left to turn. Closing the last 2× needs
**fewer calls per run**, not more GPU (see §5).

*Dashboard caveat:* `e2e_request_latency` is per **call**, and its histogram
buckets top out at 8 s, so under overload the panel cannot show the true 60 s+
tail — the load driver's client-side measurement is the SLO source of truth.

---

## 4. Agent value

The verify→revise loop (cap 3, `agent/graph.py`) earns its keep, modestly: per-
iteration pass rate rises 36.7 % → 40.0 % at baseline and 40.0 % → 43.3 % after
tuning — one additional correct question each, from the first revise. The
canonical example is the `formula_1` "coordinates" question: `generate` returned
11 duplicate rows, `verify` correctly flagged the duplicates, and `revise` added
`SELECT DISTINCT` to match gold (visible as a `generate → verify → revise`
waterfall in `langfuse_trace.png`). Crucially, the **second** revise adds zero in
both runs — so the loop's value is entirely in iteration 1, and a cap of 2 would
give the same quality at lower latency.

---

## 5. What I'd do with more time

- **Cut calls per run to actually hit the SLO.** The data shows the 2nd revise is
  wasted, so cap at 2; and add a *deterministic* verify pre-check (SQL errored / 0
  rows / duplicate rows) that only calls the LLM-verifier when the cheap checks
  pass — this roughly halves LLM calls per run, directly attacking the p95 tail
  that misses the SLO, with no expected quality loss.
- **Async the agent** (`graph.ainvoke` in an `async def` handler) instead of
  4 processes — one event loop handles the I/O-bound calls far more efficiently
  than a thread pool, and removes the residual queuing tail.
- **Richer eval:** add an LLM-as-judge dimension beyond execution accuracy
  (catches right-rows-wrong-reason), and feed Langfuse-captured production failures
  back into the eval set (the production→research loop).
- **Schema retrieval:** pass only the tables relevant to the question instead of
  the whole DB schema — shrinks prefill further and sidesteps edge-case DBs.

---

### Deliverables
| File | Status |
|---|---|
| `infra/grafana/provisioning/dashboards/serving.json` | latency pctiles + throughput + KV/preemptions, reacts under load |
| `agent/graph.py`, `agent/prompts.py` | verify/revise/router + prompts |
| `agent/schema.py` | NULL foreign-key crash fixed |
| `evals/run_eval.py` | execution-accuracy + per-iteration runner |
| `results/eval_baseline.json` / `eval_after_tuning.json` | 40.0 % / 43.3 % |
| `results/load_test_*.json` | baseline (P95 83 s) → final (P95 10 s, 0 errors) |
| `screenshots/*.png` | vllm_manual_query, grafana_serving, grafana_eval_run, grafana_before, grafana_after, langfuse_trace, langfuse_tags |
