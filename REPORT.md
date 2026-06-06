# LLM inference + observability — REPORT

*Text-to-SQL PoC on BIRD-bench. Qwen3-30B-A3B served with vLLM on 1× H100;
LangGraph verify→revise agent; Prometheus/Grafana + Langfuse observability.*

> **Status:** skeleton. Sections marked **[H100]** must be filled from the real
> 30B run; **[FILL]** from local dev. Target length ≤ 3 pages.

---

## 1. Serving configuration (Phase 1)

Workload profile this config targets: 1.5–3K-token prompts, short structured
(SQL) outputs, ~2–3 dependent LLM calls per agent run, latency SLO P95 < 5s @
≥10 RPS. Model is a 30B **MoE** (3.3B active) on one 80 GB H100.

Final `scripts/start_vllm.sh` flags **[H100 — confirm each on the GPU]**:

| Flag | Value | One-line justification |
|------|-------|------------------------|
| `--max-model-len` | _[H100]_ | Cap context to the real prompt size (~3–4K) so more KV blocks are free for concurrency. |
| `--gpu-memory-utilization` | _[H100]_ | Push KV headroom up while leaving room for activations; watch preemptions. |
| `--max-num-seqs` / `--max-num-batched-tokens` | _[H100]_ | Size the continuous-batching budget for short outputs + medium prompts. |
| `--enable-chunked-prefill` | _[H100]_ | Stop big prefills (3K prompts) from stalling decode of other requests. |
| `--kv-cache-dtype` (fp8?) | _[H100]_ | Trade a little accuracy for KV headroom / throughput — verify quality survives. |
| _(quantization, if used)_ | _[H100]_ | If serving an FP8 weight variant, note the model id change and the speed/VRAM tradeoff. |

*Rationale to expand once measured: which lever moved which metric (see §3).*

## 2. Baseline eval results (Phase 5) **[H100]**

- Overall execution accuracy: _[H100]_ % (n=30).
- Pass rate by iteration (carry-forward): iter0 _[ ]_ → iter1 _[ ]_ → iter2 _[ ]_.
- Questions that triggered a revise: _[ ]_ / 30. Mean iterations: _[ ]_.
- Commentary: _[does the loop earn its keep? see §4]_.

*(Dev sanity-check ran against Nebius to validate the harness end-to-end; those
numbers are not reported here — only the H100 30B counts.)*

## 3. Hitting the SLO (Phase 6) **[H100]**

Baseline vs SLO (P95 < 5s @ ≥10 RPS / 5 min): _[H100]_.

Iteration log — *saw X → hypothesized Y → changed Z → result W*:
1. _[H100]_
2. _[H100]_
3. _[H100]_

Before/after evidence: `screenshots/grafana_before.png`,
`screenshots/grafana_after.png`. Did end-to-end latency follow the metric we
targeted? Did quality survive (`results/eval_after_tuning.json`)? _[H100]_.

Final verdict: SLO hit / missed, with the gap quantified. _[H100]_.

## 4. Agent value (one paragraph) **[H100]**

The verify→revise loop is wired with an iteration cap of 3
(`agent/graph.py:MAX_ITERATIONS`). Verify flags four failure modes (SQL error,
zero rows when rows are implied, columns that can't answer the question, wrong
result shape) and routes a flagged result back through a revise that sees the
failing SQL, its result, and the complaint. Whether this *earned its keep* is
read directly off the per-iteration pass rate in §2: if iter2 ≫ iter0 the loop
is doing real work; if they're equal it isn't. _[H100 — cite the actual numbers
and the example question that revised]_.

## 5. What I'd do with more time (be specific)

Candidates (to be pruned to what's honest after the run):
- Add an **LLM-as-judge** eval dimension beyond execution accuracy (catch
  right-rows-wrong-reason cases) and fold Langfuse-captured failures back into
  the eval set (the production→research loop).
- Try **prefix caching** for the shared schema/system prompt and quantify the
  TTFT/throughput gain on the dashboard.
- Make verify **cheaper** (smaller max-tokens / a deterministic pre-check before
  the LLM call) to cut the loop's latency tax measured in §3.
- Schema-aware retrieval for large DBs so the prompt carries only relevant
  tables, shrinking prefill.

---

### Deliverables checklist
- [ ] `infra/grafana/provisioning/dashboards/serving.json` (done; reacts under load — verify on H100)
- [x] `agent/graph.py`, `agent/prompts.py`
- [x] `evals/run_eval.py`
- [ ] `results/eval_baseline.json` **[H100]**
- [ ] `results/eval_after_tuning.json` **[H100]**
- [ ] `screenshots/vllm_manual_query.png`, `grafana_serving.png`,
      `langfuse_trace.png`, `langfuse_tags.png`, `grafana_eval_run.png`,
      `grafana_before.png`, `grafana_after.png`
