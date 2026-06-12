# Session memory — LLM inference + o11y home assignment

> **Purpose.** Durable working memory for this build session: survives context
> compaction and seeds a future project WIKI. Captures decisions, open
> questions, clarifications, failures/gotchas, and progress.
> **Cadence.** Refreshed roughly every 5 interactions.
> **Last updated:** 2026-06-12 · interaction ~52 · *All phases run on H100; REPORT.md written.*

---

## 1. What this assignment is
Text-to-SQL PoC over BIRD-bench. Two halves, deliberately separated:
- **Inference infra:** serve `Qwen/Qwen3-30B-A3B-Instruct-2507` on 1× H100 via
  vLLM; scrape `/metrics` with Prometheus; visualize in Grafana.
- **Agent:** LangGraph self-consistency loop (generate SQL → execute → verify →
  revise), traced with Langfuse, scored by execution-accuracy eval, load-tested
  against an SLO.

**The graded thing is the serving layer + observability, not the model's SQL.**
Phase 6 (SLO diagnosis) = 25%, serving config = 15%, dashboard = 15%. Grading
rewards metric-grounded reasoning over green checkmarks.

**SLO:** P95 end-to-end agent latency < 5s, ≥10 RPS (1 RPS = one full agent
run), sustained over a 5-minute window.

## 2. Key decisions (with rationale)
| # | Decision | Why |
|---|----------|-----|
| D1 | Dev against **Nebius Token Factory**, same Qwen3-30B-A3B | Prompts/tokenization transfer 1:1 to the H100; ~pennies per eval run |
| D2 | Develop on **Ubuntu 24.04 VM**, not the M1 Mac | vLLM doesn't run usefully on Apple Silicon; VM mirrors Nebius + H100 env |
| D3 | **Config-switch via `.env`** (`LLM_BACKEND`), not hardware auto-detect | Agent is just an HTTP client; only the vLLM launch script is hardware-specific |
| D4 | **Mock `/metrics` exporter** for local Phase 2 | Dev box can't run vLLM; validate dashboard wiring before spending H100 time |
| D5 | Stage plan: **A (Nebius) → A.5 (CPU-vLLM, tiny model) → B (H100)** | Build logic cheap, validate metrics plumbing, then collect real numbers |
| D6 | Git: **Claude edits files, user runs git** | Sandbox mount is create-but-not-delete → can't run git locally |
| D7 | Push to **`origin` = github.com/emanuelam00/h100-inference-and-observability**, `upstream` = course repo | Fork pattern; keep ability to pull course updates |

## 3. Questions raised & clarified
- *Why an external model if the H100 can serve it?* → You own the serving layer
  only on the H100; Nebius is a black box (no `/metrics`, no flags). Dev backend
  is disposable; it exists so you don't burn GPU hours debugging Python.
- *On H100, do we serve Qwen ourselves?* → Yes, vLLM pulls it from HF (needs
  `HF_TOKEN`) and serves at `:8000`.
- *What's an SLO?* → target you commit to (here P95<5s @10RPS/5min). SLI = the
  measurement, SLA = the contract.
- *Can we run CPU-vLLM?* → Yes but only with a **tiny** stand-in (`Qwen3-0.6B`);
  the 30B won't fit/run on CPU. Purpose is real metrics plumbing, not quality.

## 4. Failures / gotchas (carry forward)
- **vLLM ≠ Apple Silicon.** No CUDA; CPU build is x86-oriented. → moved dev to VM.
- **Sandbox mount can create but not delete files.** A half-run `git remote
  rename` left `.git/*.lock` files; user cleaned them on the Mac. Lesson: never
  run git from the sandbox.
- **`.gitignore` excludes `results/*.json` and `screenshots/*.png`** — but those
  are required deliverables. ⚠️ At submission: `git add -f` them.
- **Langfuse stack is heavy** (postgres+clickhouse+redis+minio+2 services).
  Needs Docker ≥8 GB RAM or clickhouse OOM-loops.
- **Prompt brace bug:** `VERIFY_SYSTEM` is passed raw (not `.format`-ed), so its
  JSON example must use single braces, not `{{ }}`. Fixed.
- **H100 vLLM crash (2026-06-12):** `transformers 5.x` is incompatible with vLLM
  0.10.2 → `Qwen2Tokenizer has no attribute all_special_tokens_extended` at
  tokenizer load. **`uv pip install 'transformers<5'` does NOT stick** — `uv run`
  re-syncs the venv to the lock on every call and reverts it. Correct fix:
  `uv add 'transformers>=4.51,<5'` (edits pyproject + relocks + installs). Fix
  lives on the **H100** (uv add regenerated pyproject+lock there); Mac pyproject
  left untouched to avoid a merge conflict → **commit pyproject.toml + uv.lock
  from the H100.** Latent until now (tokenizer only loads when serving; Nebius
  never did). Fallback: `source .venv/bin/activate` + python directly (bypasses
  uv run's resync).
- **vLLM ignores `.env`:** must `export HF_TOKEN` in the shell before
  `start_vllm.sh`, else unauthenticated/slow HF downloads.

## 4b. Environment specs (→ future README prerequisite)
- **Dev VM:** Ubuntu 24.04, **4 vCPU / 32 GB RAM** (Emanuel's). Comfortably
  above the Langfuse stack's ≥8 GB floor. ⚠️ **TODO for final README.md:** add a
  "Resource requirements" prerequisites section — dev box (≥4 vCPU / ≥8 GB,
  16 GB+ comfortable) vs H100 box (1× H100 80 GB) — once the build is finished.
- **H100 box:** 1× H100 80 GB (Stage B only).
- **Local laptop:** M1 Mac (kept clean; not used for the build — vLLM won't run on it).

## 5. Reference facts
- Nebius base URL: `https://api.tokenfactory.nebius.com/v1`, key `NEBIUS_API_KEY`.
  Qwen3-30B-A3B-2507 pricing ~$0.10/$0.30 per 1M in/out tokens, ~88 tok/s.
- vLLM metric names used: `vllm:num_requests_running|waiting`,
  `vllm:generation_tokens_total`, `vllm:prompt_tokens_total`,
  `vllm:request_success_total`, `vllm:e2e_request_latency_seconds`,
  `vllm:time_to_first_token_seconds`, `vllm:time_per_output_token_seconds`,
  `vllm:request_queue_time_seconds`, `vllm:gpu_cache_usage_perc` /
  `vllm:kv_cache_usage_perc`, `vllm:num_preemptions_total`.

## 6. Progress
**Stage A — DONE & self-tested (no GPU):**
- `agent/config.py` (backend switch) + `.env.example` rewrite.
- `agent/prompts.py` (6 prompts) + `agent/graph.py` (verify/revise/router, cap=3).
- `evals/run_eval.py` (`eval_one` + per-iteration `summarize`).
- `infra/.../serving.json` (latency pctiles / throughput / KV cache, 3 rows).
- `scripts/mock_vllm_metrics.py` (reacting `/metrics` + `/burst`).
- `RUNBOOK.md`. Self-tests: config resolution, verdict parse, router, eval
  carry-forward (0.25→0.50), dashboard JSON, exporter output, py_compile — all pass.

**VM milestone (2026-06-06, Nebius dev run — THROWAWAY numbers):**
- Agent live on VM, `/answer` working (note: HTTP field is `db`, not `db_id`).
- Dev eval (30 q): overall **40%** (12/30). Per-iter: iter0 36.7% → iter1 40% →
  iter2 40%. 11/30 revised, avg 1.63 iters, 0 errors, mean 1.95s/run.
- **Read:** loop earns its keep but barely (+1 net question, +3.3pts); 2nd revise
  adds nothing. Low yield (11 revises → 1 win).
- Confirmed (1) revises do trigger. (2) dashboard reaction still TODO.

**Grafana validated (2026-06-06):** mock exporter + /burst → all 3 categories
react. Metric-name audit vs installed **vLLM 0.10.2**: ALL dashboard metrics
present (incl. both gpu_cache_usage_perc & kv_cache_usage_perc) → dashboard
guaranteed to light up on H100, no changes.

**Stage A = COMPLETE.** CPU smoke test SKIPPED: dev VM CPU is Xeon E5-2660 v2
(Ivy Bridge) — **no AVX512**, so vLLM CPU build impossible. Audit covered the
only risk it would have.

**H100 prep done:** `H100_PLAYBOOK.md` (fail-fast ordered session + min-viable
path), `scripts/start_vllm.sh` (reasoned initial flags: max-model-len 8192,
gpu-mem-util 0.90, max-num-seqs 256, prefix-caching; Phase-6 levers listed),
`scripts/check_env.sh` (pre-flight: driver/uv/docker/disk/ports), RUNBOOK Stage B
(env check + `uv sync` GPU install + CPU build appendix).

**H100 Phase 1 (2026-06-12): vLLM 0.10.2 SERVING the 30B on :8000.** Prometheus
scraping confirmed (200 OK from docker net). Key startup facts for REPORT/Phase 6:
- weights 56.9 GiB; with gpu-mem-util 0.90 → **KV cache only 12.8 GiB = ~140k
  tokens = ~17x concurrency** at max-model-len 8192. THE lever if queue-bound:
  lower max-model-len (→4096) or raise gpu-mem-util.
- perf warnings (→ "more time" / Phase 6): no tuned MoE kernel config for this
  H100 shape; FlashInfer not installed (PyTorch-native sampling fallback).
- server default sampling temp=0.7 (agent overrides to 0.0 per-request → eval
  determinism preserved).

**H100 REAL BASELINE (Phase 5, 2026-06-12) — results/eval_baseline.json:**
overall **40%** (12/30); per-iter **iter0 36.7% → iter1 40% → iter2 40%**; 10
revised; mean **1.004s/agent-run** (single-shot, well under 5s SLO); 0 errors.
Loop earns its keep modestly (+1 q / +3.3pts at iter1); 2nd revise adds nothing.
Closely matches Nebius dev run → validates the dev approach. Phase 4 Langfuse
confirmed: traces show generate→verify→revise waterfall; metadata source/db_id
attached (filterable). Agent latency ~1s local vs ~1.8s Nebius (no network hop).

**H100 PHASE 6 — baseline load test (10 RPS / 300s): SLO BADLY MISSED.**
load_test.json: p50 **69s**, p95 **83s**, p99 90s, max 113s (target P95<5s → ~16x
over); achieved only 8.3 RPS; **~13% errors** (379 http + 14 timeout / 3000).
- **Diagnosis (metric-grounded): prefill-throughput-bound at vLLM, amplified by
  the agent's ~3 calls/run.** Offered ≈10 RPS×3 = ~30 calls/s vs vLLM ceiling
  ~15-18 calls/s ("requests finished/s" panel); prompt-tok pinned ~13K, gen ~1K.
  vLLM queue grows unbounded → calls exceed 60s client timeout (the 379 errors).
  KV stayed ~8%, preemptions 0 → NOT memory-bound.
- Dashboard caveat: e2e_request_latency is per-CALL (driver is per-RUN); histogram
  buckets top at 8s so panel can't show true 60s+ tail (Phase 2 learning).
- **Lever direction:** no single vLLM flag 4x's prefill on 1 H100 → reduce offered
  load (fewer calls/run e.g. cheaper/skippable verify, or find sustainable RPS).
  Decide after confirming steady-state (waiting↑, requests/s ceiling, KV low).

**Phase 6 full-window steady-state CONFIRMED diagnosis:** waiting(queue)=0 whole
test, requests-finished ~30 calls/s, KV ~15%, preemptions 0, per-call p99 ~5s.
→ vLLM healthy w/ headroom; **SLO miss is the AGENT SERVER, not the model.**

**Phase 6 ITERATION 1 — agent workers (`uvicorn --workers 4`):**
load_test_after.json: p50 **2.2s** (was 69s), p95 **10.2s** (was 83s), p99 17s
(was 90s). **p95 down 8x, p50 down 31x** — confirms agent-concurrency bottleneck.
Grafana dashboard ~unchanged (vLLM was never the constraint = the lesson). SLO
still missed (10.2s vs 5s) but 16x→2x over. *"saw vLLM queue=0 yet run-P95=83s →
hypothesized agent sync-handler concurrency limit → scaled to 4 workers → P95
83s→10.2s."*

**Second finding:** http_errors = **exactly 379 in BOTH runs** → deterministic,
not load → almost certainly **context-length overflow** (big BIRD schemas >
max-model-len 8192). ~12.6% error rate. Separate fix (raise max-model-len / trim
schema). CONFIRM via agent log (BadRequestError/max context length).

**379 errors ROOT-CAUSED (iter 2):** diagnosis arc = context-length (disproved,
max prompt 1435 tok) → missing-db (disproved, 0 missing) → instrumented actual
exception → **bug in provided `agent/schema.py`**: `render_schema` crashes
(`AttributeError: NoneType .replace`) on FKs where SQLite leaves the "to" column
(fk[4]) NULL — hits `european_football_2` (129) + `debit_card_specializing` (64)
= 193 q = all 1500-pool render failures = the 12.6% load errors. **Fixed**
schema.py to guard NULL fk target/from/table. So the 379 were INVALID-INPUT
(schema render), NOT overload — true serving error rate ~0. (eval_set has none of
these 2 DBs → eval baseline unaffected; quality unchanged.)

**Phase 6 final config = uvicorn --workers 4 + schema fix.** Pending final runs:
load_test_final.json (expect errors ~0), eval_after_tuning.json (expect ~40%).

**Phase 6 ITERATION 3 — verify pre-check + MAX_ITERATIONS=2:** deterministic
pre-check (SQL-error auto-fail / clean auto-pass / 0-rows+dups escalate to LLM) +
cap 2. Runs drop ~3 LLM calls → ~1 (avg iters 1.53→1.2; vLLM ~30→16 calls/s).
P95 10.0s → ~5s. eval unchanged **43.3%** (formula_1 still revises; 6/30 revise)
→ quality intact. Lower vLLM throughput = removed overhead, NOT regression.

**CORRECTION (300s rerun):** first final run was only 180s (P95 4.91s); SLO needs
a 5-MIN window. Proper 300s run (`load_test_final.json`): **P95 5.28s** (P50 1.52s,
P99 8.36s, 0 errors, 99.7% ok). So honestly a **NARROW MISS** (~5% over 5s), NOT
a hit — reported as such (rubric rewards honest near-miss > unexplained hit).

**FINAL VERDICT: SLO narrowly missed, ~5% over (P95 5.28s vs 5s @10RPS/5min), 0
errors, 43.3% eval, 16x better than 83s baseline.** Full Phase 6 arc: baseline 83s
(16x miss) → agent-concurrency diagnosed past green vLLM dashboard → workers (8x)
→ schema-bug fix (13%→0% errors) → fewer calls/run (boundary). REPORT.md updated
to honest 300s numbers + near-miss verdict.

**Next:** final submission commit (git add -f results + screenshots).

**Prompt tweak v2 (2026-06-06, Nebius — KEPT):** added DISTINCT nudge
(generate+revise) + verify duplicate-row check. Result: overall flat 40%, but
`formula_1` flipped fail→pass *via the loop* (verify caught dups → revise added
DISTINCT, iter0 false → iter1 true). Aggregate flat = +1 target fixed offset by
−1 elsewhere (noise at n=30, non-deterministic backend). Kept edits (principled,
no regression); real verdict deferred to H100. **`formula_1` is now the Phase-4
Langfuse trace + agent-value example.**

## 6b. Improvement backlog (post-data, do NOT pre-tune)
- **Verify too lenient:** passed `formula_1` which had 11 duplicate rows vs gold's
  `SELECT DISTINCT` (1 row) → scored fail, loop never fired. Lever: tighten verify
  to catch duplicate/wrong-shape results.
- **Duplicates vs gold:** add a `SELECT DISTINCT` nudge to generate prompt when a
  single/unique answer is implied (BIRD gold often uses DISTINCT).
- **Revise effectiveness low:** 2nd revise fixed nothing — investigate.
- **Check composition:** net +1 may hide "+2 fixed, −1 broken" — inspect
  per-question `per_iter_correct` before writing the agent-value paragraph.
- All real tuning + numbers must come from the H100 30B run, not Nebius.

**Not started:** Stage A.5 (CPU-vLLM), Stage B (H100: Phase 1 vLLM flags, real
eval baseline, Phase 6 SLO load test + iteration log), REPORT.md body.

## 7. Next actions
1. User: push Stage A, run on VM, paste back a revised `/answer` + eval JSON.
2. Tune prompts against Nebius if needed.
3. Fill REPORT.md Phase 1 config rationale; then book H100 for Stage B.

## 8. Update log
- 2026-06-06 (int ~13): Initialized. Stage A built & self-tested; report skeleton created.
- 2026-06-06 (int ~16): Added explicit Docker/Compose/uv install steps to RUNBOOK §0.
  Logged dev VM spec (4 vCPU / 32 GB) → flagged as future README resource-spec prerequisite.
- 2026-06-06 (int ~22): Agent live on VM. Dev eval ran (40%, loop +3.3pts). Logged
  improvement backlog (verify leniency, DISTINCT, revise effectiveness). Explained
  per-iteration pass rate + carry-forward to user.
- 2026-06-06 (int ~26): Applied + validated DISTINCT/verify-duplicate tweak on
  Nebius. formula_1 fixed via loop (great Langfuse/agent-value example). Aggregate
  flat (noise). Kept edits, stopped dev-tuning. Pivoting to Phase 2 Grafana check.
- 2026-06-06 (int ~30): Grafana validated (panels react). Metric audit vs vLLM
  0.10.2 = all present. CPU smoke test skipped (no AVX512 on E5-2660 v2). Added
  check_env.sh, H100_PLAYBOOK.md, reasoned start_vllm.sh flags, RUNBOOK Stage B
  bring-up (env check + GPU install + CPU appendix). Stage A complete.
