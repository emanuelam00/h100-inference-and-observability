# Session memory — LLM inference + o11y home assignment

> **Purpose.** Durable working memory for this build session: survives context
> compaction and seeds a future project WIKI. Captures decisions, open
> questions, clarifications, failures/gotchas, and progress.
> **Cadence.** Refreshed roughly every 5 interactions.
> **Last updated:** 2026-06-06 · interaction ~22 · *Agent live on VM/Nebius; dev eval ran.*

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

**Pending (VM, user):** confirm dashboard panels react (mock exporter + /burst).

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
