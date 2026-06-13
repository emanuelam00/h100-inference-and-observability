# Engineering writeup: text-to-SQL inference and observability

Text-to-SQL over BIRD-bench. `Qwen/Qwen3-30B-A3B-Instruct-2507` served with vLLM
0.10.2 on 1x H100 80GB, a LangGraph generate, execute, verify, revise agent,
Prometheus and Grafana on vLLM `/metrics`, and Langfuse on the agent traces.
Target SLO: **p95 end-to-end agent latency < 5s at >=10 RPS over a 5-minute
window.**

---

## 1. Serving configuration

Workload: prompts of roughly 0.9K to 1.5K tokens (schema plus question, measured
p50 918, p95 1414, max 1435 tokens for generate), short SQL outputs, about 2 to 3
dependent LLM calls per agent run. Model is a 30B MoE (3.3B active). Flags in
`scripts/start_vllm.sh`:

| Flag | Value | Rationale |
|---|---|---|
| `--dtype bfloat16` | bf16 | H100-native. Full 30B MoE weights (56.9 GiB) fit in 80GB without quantization. |
| `--max-model-len` | 8192 | Generous headroom over the measured ~1.5K-token prompts. A smaller value would free KV, but KV was never the constraint (section 3). |
| `--gpu-memory-utilization` | 0.90 | Maximize KV while leaving room for activations. Left 12.8 GiB of KV (~140K tokens). |
| `--max-num-seqs` | 256 | High concurrency for throughput. Real prompts are small, so effective concurrency reached ~37 with KV still around 15%. |
| `--max-num-batched-tokens` | 8192 | Per-step batching budget balancing prefill against decode. |
| `--enable-prefix-caching` | on | The DB schema and system prompt are reused across the 2 to 3 calls per run and across same-DB requests, a direct TTFT win for this workload. |

**Key startup observation (used in section 3).** Weights take 56.9 GiB, leaving
only 12.8 GiB of KV, about 17x concurrency at full 8192-token context. In practice
prompts are ~1.5K tokens, so KV stayed around 15% even under load. There is
abundant headroom and the serving layer is not memory-bound.

---

## 2. Evaluation

`results/eval_baseline.json`, 30 BIRD questions, execution accuracy (canonicalized
row sets):

- **Overall: 40.0%** (12 of 30), 0 agent errors, mean 1.0s per run.
- **Pass rate by iteration:** iter 0 **36.7%**, iter 1 **40.0%**, iter 2 **40.0%**.
- 11 of 30 questions triggered a revise. Mean 1.6 generate or revise calls.

The verify, revise loop lifts pass rate by one question (+3.3 points) at the first
revise and nothing at the second. The loop earns its keep, but marginally, and the
third iteration is wasted (section 4).

---

## 3. SLO investigation

**Baseline load test** (`--rps 10 --duration 300`, default single-process agent):

| metric | value | vs SLO |
|---|---|---|
| p50 | 69s | n/a |
| **p95** | **83s** | **16x over (target 5s)** |
| p99 | 90s | n/a |
| errors | 379 HTTP + 14 timeout (~13%) | n/a |

The dashboard during this run looked healthy: `waiting (queue) = 0` the whole
time, ~30 calls/s completed, KV around 15%, preemptions 0, per-call p99 around 5s.
That contradiction is the whole diagnosis. vLLM had headroom, yet agent-run p95 was
83s, so the time was being spent above the model.

### Iteration log

**1. Saw** vLLM `queue=0` and KV at 15% yet run-p95 of 83s and achieved RPS capped
around 8. **Hypothesized** the bottleneck is the agent server, not vLLM: the
`/answer` handler is synchronous (a blocking `graph.invoke`), so FastAPI runs it in
a fixed thread pool. At 10 runs/s the pool saturates and runs queue inside the
agent. **Changed** to `uvicorn --workers 4`. **Result:** p95 **83s to 10.2s** (8x),
p50 69s to 2.2s. The vLLM dashboard was unchanged, confirming vLLM was never the
constraint. The SLO went from 16x to about 2x over.

**2. Saw** `http_errors` was exactly **379 in every run**, identical and therefore
deterministic, not load-related. **Hypothesized** context-length overflow, then
disproved it by direct measurement (max prompt 1435 tokens, far below 8192). Then
hypothesized missing databases, disproved (0 of 1500 pool questions lack a DB).
Then **instrumented the actual exception**. Root cause: the provided
`agent/schema.py` crashes (`AttributeError: NoneType.replace`) on foreign keys
where SQLite leaves the target column NULL, which hits 2 of 11 databases
(`european_football_2`, `debit_card_specializing`), 193 questions, the 12.6%.
**Changed** `render_schema` to guard the NULL. **Result:** load-test `http_errors`
**379 to 0**, success rate 99.6%. These errors were invalid-input (schema render),
not overload. The true serving error rate under load is about 0.

**3. Saw** that with serving healthy, run-p95 was dominated by the number of
sequential calls per run (2 to 4, each at per-call p95 around 2.7s), and that the
eval's second revise was wasted (section 2). **Hypothesized** cutting calls per run
would cut the tail without hurting quality. **Changed** (a) a deterministic verify
pre-check that skips the LLM verify on clean results and only escalates SQL errors,
zero-rows, and duplicate-bloat to the LLM, and (b) `MAX_ITERATIONS` from 3 to 2.
**Result:** most runs drop from about 3 LLM calls to 1 (avg iterations 1.53 to 1.2,
vLLM call rate ~30 to ~16 calls/s), and p95 **10.0s to 5.3s**. The lower vLLM
throughput is the cause of the win, not a regression, the same offered RPS with
half the work per request.

**Final configuration** (`--workers 4` plus schema fix plus verify pre-check plus
cap 2), `results/load_test_final.json`, measured over the required 5-minute window:
**p50 1.52s, p95 5.28s, p99 8.36s, 0 HTTP errors, 99.7% success** (2990 of 3000).

**Quality survived** (`results/eval_after_tuning.json`): **43.3%** (13 of 30),
identical to the post-tuning eval and up from the 40% baseline. The pre-check still
escalates the duplicate case, so the one revise that earns its keep (`formula_1`)
is preserved, and 6 questions still revise.

**Verdict: SLO narrowly missed, reported honestly.** Over the required 5-minute
window p95 is **5.28s vs the 5s target (about 5% over)**. p50 (1.52s) clears it
comfortably, the system runs at 0 errors and 99.7% success, and this is a **16x
improvement** from the 83s baseline, sitting right on the SLO boundary. The path
mattered more than the number: a green serving dashboard hid an agent-layer
concurrency bottleneck (workers, 8x), a deterministic schema bug masquerading as
load failures (13% to 0% errors), and a structural call-count tax on latency (skip
unnecessary LLM calls, 10s to 5.3s), none of which were vLLM serving knobs. The
remaining 5% is the p99-tail work in section 5 (a few runs still take 2 calls plus
a slow generation), not a serving-config gap. KV, queue, and preemptions stayed
healthy throughout.

**Dashboard caveat.** `e2e_request_latency` is per call, and its histogram buckets
top out at 8s, so under overload the panel cannot show the true 60s-plus tail. The
load driver's client-side measurement is the SLO source of truth.

---

## 4. Agent value

The verify, revise loop earns its keep, modestly: per-iteration pass rate rises
36.7% to 40.0% at baseline and 40.0% to 43.3% in the final config, one additional
correct question, from the first revise. The canonical example is the `formula_1`
coordinates question: `generate` returned 11 duplicate rows, `verify` flagged the
duplicates, and `revise` added `SELECT DISTINCT` to match gold (visible as a
generate, verify, revise waterfall in `screenshots/langfuse_trace.png`). The second
revise added zero in every run, so we capped at 2. And since the loop's only real
catch (duplicates) is programmatically detectable, we replaced the always-on LLM
verify with a deterministic pre-check that escalates to the LLM only for SQL
errors, zero-row, and duplicate results. That preserved the loop's single useful
revise (6 of 30 questions still revise, `formula_1` among them) while cutting it to
avg 1.2 LLM calls per run, the change that brought run-p95 to the SLO boundary
(section 3) at no quality cost.

---

## 5. Limitations and next steps

- **Tighten the p99 tail (8s) and push past 10 RPS.** A few runs still take 2 calls
  plus a slow generation. A per-call timeout-and-retry (hedging) would clip the
  tail. With the call count now halved there is also serving headroom, so re-running
  at higher RPS to find the new SLO ceiling is the natural next test.
- **Async the agent** (`graph.ainvoke` in an `async def` handler) instead of 4
  processes. One event loop handles the I/O-bound calls far more efficiently than a
  thread pool and removes the residual queuing tail.
- **Richer eval.** Add an LLM-as-judge dimension beyond execution accuracy (catches
  right-rows-wrong-reason), and feed Langfuse-captured production failures back into
  the eval set (the production-to-research loop).
- **Schema retrieval.** Pass only the tables relevant to the question instead of the
  whole DB schema, which shrinks prefill further and sidesteps edge-case DBs.

---

## Artifacts

| File | What it is |
|---|---|
| `infra/grafana/provisioning/dashboards/serving.json` | Dashboard: latency pctiles, throughput, KV cache and preemptions, reacts under load |
| `agent/graph.py`, `agent/prompts.py` | The agent: verify, revise, router, prompts |
| `agent/schema.py` | NULL foreign-key crash fixed |
| `evals/run_eval.py` | Execution-accuracy and per-iteration runner |
| `results/eval_baseline.json`, `results/eval_after_tuning.json` | 40.0% then 43.3% |
| `results/load_test_*.json` | baseline p95 83s, workers 10s, final **p95 5.28s at 5-min, 0 errors (~5% over SLO, 16x better)** |
| `screenshots/*.png` | vLLM manual query, Grafana (serving, eval run, before, after), Langfuse (trace, tags) |
