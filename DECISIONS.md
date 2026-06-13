# Engineering decisions and diagnosis log

A record of the design decisions behind this repo, the gotchas resolved along the
way, and the metric-grounded trail that took end-to-end p95 latency from 83s to
5.3s. It complements `REPORT.md` (the results) by capturing the reasoning.

---

## Design decisions

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | Develop and validate the whole pipeline off-GPU, run on the H100 only for numbers that count | GPU time is expensive and rationed. The agent, evals, tracing, and dashboard authoring need no GPU, so they are built and debugged against a hosted endpoint first. |
| D2 | Use the same model (Qwen3-30B-A3B) on the hosted dev endpoint as on the H100 | Prompts and tokenization transfer 1:1, so nothing needs re-tuning when the backend switches. |
| D3 | Select the backend with one `.env` variable (`LLM_BACKEND`), not hardware auto-detection | The agent is only an OpenAI-compatible HTTP client. It never runs on the GPU itself, so the only hardware-specific piece is the vLLM launch script. A config switch is simpler and more robust than detection. |
| D4 | Ship a mock `/metrics` exporter for local dashboard validation | The dashboard needs a live, correctly-named metrics source to prove its panels react. The mock provides that with no GPU, so the dashboard wiring is validated before any GPU time is spent. |
| D5 | Stage the work: local build, then real GPU run | Build the logic cheaply, then collect real numbers on the H100. An optional CPU-vLLM step can validate the real metrics plumbing in between. |
| D6 | Keep the schema renderer defensive | Schema rendering runs on arbitrary databases, so it must tolerate SQLite edge cases (see G3) rather than crash a whole request. |

---

## Gotchas resolved

**G1. vLLM does not run usefully on Apple Silicon.** No CUDA, and the CPU build is
x86-oriented. Development moved to an x86 Linux box, which also mirrors the target
GPU environment.

**G2. `transformers` 5.x is incompatible with vLLM 0.10.2.** A fresh resolve pulls
`transformers` 5.x, which crashes at tokenizer load with
`Qwen2Tokenizer has no attribute all_special_tokens_extended`. The fix is to pin
`transformers>=4.51,<5`. Important subtlety: use `uv add`, not `uv pip install`.
`uv run` re-syncs the virtualenv to the lockfile on every call, so a bare
`uv pip install` is silently reverted to the locked 5.x version. `uv add` edits
`pyproject.toml` and regenerates `uv.lock`, so the pin persists.

**G3. Schema renderer crash on NULL foreign-key targets.** `render_schema`
crashed with `AttributeError: NoneType.replace` on foreign keys where SQLite
leaves the target column NULL (a FK that references a table's primary key without
naming a column). This hit 2 of 11 BIRD databases and produced a deterministic
12.6% error rate under load. Fixed by guarding the NULL target, from, and table
fields. This bug was invisible to a happy-path eval and only surfaced under the
load test, the production-observability lesson in miniature.

**G4. The Langfuse stack is memory-heavy.** It runs postgres, clickhouse, redis,
minio, and two Langfuse services. Give Docker at least 8GB of RAM (16GB is
comfortable) or clickhouse will OOM-loop.

**G5. vLLM does not read `.env`.** Export `HF_TOKEN` in the shell before launching
vLLM, otherwise model downloads are unauthenticated and slow.

**G6. Latency histogram buckets top out at 8s.** Under heavy overload, per-call
latency can exceed 60s, but the dashboard panel clamps at the top finite bucket.
The load driver's client-side measurement is the source of truth for the SLO.

---

## The diagnosis trail (83s to 5.3s)

The baseline missed the SLO by 16x (p95 83s vs the 5s target) with a 12.6% error
rate, yet the vLLM dashboard looked healthy throughout: queue depth 0, KV cache
around 15%, no preemptions, per-call p99 around 5s. That contradiction drove the
investigation. Each step changed one thing and re-measured.

**Iteration 1, agent concurrency.** vLLM having headroom while run-p95 sat at 83s
meant the time was being spent above the model. The agent's request handler was
synchronous (a blocking `graph.invoke` in a fixed thread pool), so at 10 runs/s the
pool saturated and full runs queued inside the agent process. Scaling to 4 worker
processes cut p95 from 83s to 10.2s (8x). The vLLM dashboard was unchanged, which
confirmed the bottleneck had never been the model server.

**Iteration 2, a deterministic bug, not load.** The error count was exactly 379 in
every run, identical regardless of load, which is the fingerprint of a per-input
failure rather than overload. Two hypotheses were disproved by direct measurement
before instrumenting the real exception: context-length overflow (ruled out, the
largest prompt was 1435 tokens against an 8192 limit) and missing databases (ruled
out, 0 of 1500 pool questions referenced an absent DB). The actual exception was
G3, the schema-renderer crash. Fixing it took the load-test error rate from 12.6%
to 0%.

**Iteration 3, the structural latency tax.** With serving healthy, run latency was
dominated by the number of sequential LLM calls per request (2 to 4, each around
2.7s at p95). The eval showed the second revise never helped, and the loop's only
useful catch (duplicate rows) is programmatically detectable. So the loop was
capped at 2 and the always-on LLM verifier was replaced with a deterministic
pre-check that escalates to the model only for ambiguous results (SQL errors,
zero rows, duplicate-bloat). Most runs dropped from about 3 LLM calls to 1, the
vLLM call rate fell from ~30 to ~16 calls/s, and p95 dropped from 10s to 5.3s, all
at unchanged answer quality (43.3%).

**Outcome.** p95 of 5.28s over the required 5-minute window, about 5% over the 5s
target, with 0 errors and a 16x improvement from baseline. The remaining gap is
p99-tail work (a few runs still make 2 calls plus a slow generation), addressable
with request hedging or an async agent, not with any vLLM serving knob.
