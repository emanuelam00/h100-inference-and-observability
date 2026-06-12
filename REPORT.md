# LLM inference + observability - REPORT

Text-to-SQL PoC on BIRD-bench. `Qwen/Qwen3-30B-A3B-Instruct-2507` served with
vLLM 0.10.2 on 1× H100 80 GB, a LangGraph `generate → execute → verify → revise`
agent, Prometheus/Grafana on vLLM `/metrics` + Langfuse on the agent traces.
SLO target: **P95 end-to-end agent latency < 5 s at ≥ 10 RPS over 5 min.**

---

## 1. Serving configuration (Phase 1)

Workload: ~0.9–1.5 K-token prompts (schema + question, measured p50 918, p95
1414, max 1435 tokens for generate), short SQL outputs, ~2–3 dependent LLM calls
per agent run. Model is a 30 B **MoE** (3.3 B active). Flags in `scripts/start_vllm.sh`:

| Flag | Value | Justification |
|---|---|---|
| `--dtype bfloat16` | bf16 | H100-native, full 30 B MoE weights (56.9 GiB) fit 80 GB without quantization. |
| `--max-model-len` | 8192 | Generous headroom over the measured ~1.5 K-token prompts, smaller len would free KV but KV was never the constraint (see #3). |
| `--gpu-memory-utilization` | 0.90 | Maximize KV while leaving room for activations. Left 12.8 GiB KV (~140 K tokens). |
| `--max-num-seqs` | 256 | High concurrency for throughput, real prompts are small so effective concurrency reached ~37 with KV still ~15 %. |
| `--max-num-batched-tokens` | 8192 | Per-step batching budget balancing prefill vs decode. |
| `--enable-prefix-caching` | on | The DB schema + system prompt are reused across the 2–3 calls per run and across same-DB requests - a direct TTFT win for this workload. |

**Key startup observation (used in #3):** weights take 56.9 GiB, leaving only
12.8 GiB KV = ~17× concurrency *at full 8192-token context*. In practice prompts
are ~1.5 K tokens, so KV stayed ~15 % even under load - **abundant headroom, the
serving layer is not memory-bound.**

---

## 2. Baseline eval results (Phase 5)

`results/eval_baseline.json` - 30 BIRD questions, execution accuracy (canonicalized row sets):

- **Overall: 40.0 %** (12/30), 0 agent errors, mean 1.0 s/run.
- **Pass rate by iteration:** iter 0 **36.7 %** → iter 1 **40.0 %** → iter 2 **40.0 %**.
- 11/30 questions triggered a revise, mean 1.6 generate/revise calls.

Commentary: the verify→revise loop lifts pass rate by one question (+3.3 pts) at
the first revise and **nothing** at the second - the loop earns its keep, but
marginally, and the 3rd iteration is wasted (see #4).

---

## 3. Hitting the SLO (Phase 6)

**Baseline load test** (`--rps 10 --duration 300`, default single-process agent):

| metric | value | vs SLO |
|---|---|---|
| P50 | 69 s | - |
| **P95** | **83 s** | **16× over (target 5 s)** |
| P99 | 90 s | - |
| errors | 379 HTTP + 14 timeout (≈13 %) | - |

The dashboard during this run looked *healthy*: `waiting (queue) = 0` the whole
time, ~30 calls/s completed, KV ~15 %, preemptions 0, per-**call** p99 ~5 s. That
contradiction is the whole diagnosis: vLLM had headroom, yet agent-**run** P95 was
83 s - so the time was being spent **above** the model.

### Iteration log

**1. saw** vLLM queue=0 / KV 15 % yet run-P95 83 s and achieved RPS capped ~8
→ **hypothesized** the bottleneck is the agent server, not vLLM: the `/answer`
handler is synchronous (`def` + blocking `graph.invoke`) so FastAPI runs it in a
~40-thread pool, at 10 runs/s the pool saturates and runs queue *inside the agent*
→ **changed** to `uvicorn --workers 4` → **result:** P95 **83 s → 10.2 s**, P50
69 s → 2.2 s (8×/31×). The vLLM dashboard was *unchanged* - confirming vLLM was
never the constraint. SLO went from 16× to ~2× over.

**2. saw** `http_errors` = exactly **379 in every run** (identical → deterministic,
not load) → **hypothesized** context-length overflow → **disproved** by direct
measurement (max prompt 1435 tokens « 8192), then hypothesized missing DBs →
**disproved** (0/1500 pool questions lack a DB), then **instrumented the actual
exception** → root cause: the *provided* `agent/schema.py` crashes
(`AttributeError: NoneType.replace`) on foreign keys where SQLite leaves the
target column NULL - hits 2 of 11 DBs (`european_football_2`,
`debit_card_specializing`), 193 questions = the 12.6 % → **changed** `render_schema`
to guard the NULL → **result:** load-test `http_errors` **379 → 0**, success rate
99.6 %. These errors were *invalid-input* (schema render), not overload, true
serving error rate under load is ~0.

After this fix, run-P95 was 10.0 s (`results/load_test_after2.json`) with 0
errors - healthy serving, but still ~2× the SLO.

**3. saw** that with serving healthy, run-P95 was dominated by the **number of
sequential calls** per run (2–4 × per-call p95 ~2.7 s), and that the eval's 2nd
revise was wasted (#2) → **hypothesized** cutting calls per run would cut the
tail without hurting quality → **changed** (a) a *deterministic verify
pre-check* that skips the LLM verify on clean results and only escalates SQL
errors / zero-rows / duplicate-bloat to the LLM, and (b) `MAX_ITERATIONS 3 → 2`
→ **result:** most runs drop from ~3 LLM calls to 1 (avg iterations 1.53 → 1.2,
vLLM call rate ~30 → ~16 /s), and **P95 10.0 s → 5.3 s.** The lower vLLM
throughput is the *cause* of the win, not a regression - same offered RPS, half
the work per request.

**Final configuration** (`--workers 4` + schema fix + verify pre-check + cap 2),
`results/load_test_final.json`, measured over the required **5-minute window**:
**P50 1.52 s, P95 5.28 s, P99 8.36 s, 0 HTTP errors, 99.7 % success** (2990/3000).

**Quality survived** (`results/eval_after_tuning.json`): **43.3 %** (13/30),
identical to the post-tuning eval and up from the 40 % baseline. The pre-check
still escalates the duplicate case, so the one revise that earns its keep
(`formula_1`) is preserved, 6 questions still revise.

**Verdict - SLO narrowly missed (honestly).** Over the required 5-minute window
P95 is **5.28 s vs the 5 s target (~5 % over)**. P50 (1.52 s) clears it
comfortably, the system runs at **0 errors / 99.7 % success**, and this is a
**16× improvement** from the 83 s baseline - the system sits right on the SLO
boundary. The path mattered more than the number: a green serving dashboard hid
an agent-layer concurrency bottleneck (workers, 8×), a deterministic schema bug
masquerading as load failures (13 % → 0 % errors), and a structural call-count
tax on latency (skip unnecessary LLM calls, 10 s → 5.3 s) - none of which were
vLLM serving knobs. The remaining ~5 % is the p99-tail work in §5 (a few runs
still take 2 calls plus a slow generation), not a serving-config gap; KV, queue,
and preemptions stayed healthy throughout.

*Dashboard caveat:* `e2e_request_latency` is per **call**, and its histogram
buckets top out at 8 s, so under overload the panel cannot show the true 60 s+
tail - the load driver's client-side measurement is the SLO source of truth.

---

## 4. Agent value

The verify→revise loop earns its keep, modestly: per-iteration pass rate rises
36.7 % → 40.0 % at baseline and 40.0 % → 43.3 % in the final config - one
additional correct question, from the first revise. The canonical example is the
`formula_1` "coordinates" question: `generate` returned 11 duplicate rows,
`verify` flagged the duplicates, and `revise` added `SELECT DISTINCT` to match
gold (visible as a `generate → verify → revise` waterfall in
`langfuse_trace.png`). The **second** revise added zero in every run, so we
capped at 2, and since the loop's only real catch (duplicates) is
programmatically detectable, we replaced the always-on LLM verify with a
deterministic pre-check that escalates to the LLM only for SQL errors, zero-row,
and duplicate results. That preserved the loop's single useful revise (6/30
questions still revise, `formula_1` among them) while cutting it to avg 1.2 LLM
calls/run - the change that brought run-P95 to the SLO boundary (#3) at no
quality cost.

---

## 5. What I'd do with more time

- **Tighten the p99 tail (8 s) and push past 10 RPS.** A few runs still take 2
  calls plus a slow generation, a per-call timeout-and-retry (hedging) would clip
  the tail. With the call count now halved there is also serving headroom, so
  re-running at higher RPS to find the new SLO ceiling is the natural next test.
- **Async the agent** (`graph.ainvoke` in an `async def` handler) instead of
  4 processes - one event loop handles the I/O-bound calls far more efficiently
  than a thread pool, and removes the residual queuing tail.
- **Richer eval:** add an LLM-as-judge dimension beyond execution accuracy
  (catches right-rows-wrong-reason), and feed Langfuse-captured production failures
  back into the eval set (the production→research loop).
- **Schema retrieval:** pass only the tables relevant to the question instead of
  the whole DB schema - shrinks prefill further and sidesteps edge-case DBs.

---

### Deliverables
| File | Status |
|---|---|
| `infra/grafana/provisioning/dashboards/serving.json` | latency pctiles + throughput + KV/preemptions, reacts under load |
| `agent/graph.py`, `agent/prompts.py` | verify/revise/router + prompts |
| `agent/schema.py` | NULL foreign-key crash fixed |
| `evals/run_eval.py` | execution-accuracy + per-iteration runner |
| `results/eval_baseline.json` / `eval_after_tuning.json` | 40.0 % / 43.3 % |
| `results/load_test_*.json` | baseline P95 83 s → workers 10 s → final **P95 5.28 s @ 5-min, 0 errors (~5 % over SLO, 16× better)** |
| `screenshots/*.png` | vllm_manual_query, grafana_serving, grafana_eval_run, grafana_before, grafana_after, langfuse_trace, langfuse_tags |
