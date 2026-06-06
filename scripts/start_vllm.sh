#!/usr/bin/env bash
#
# Start vLLM for the H100 run (Phase 1).
# Reference: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
#
# These are *initial, reasoned* flags for the assignment workload, not vLLM
# defaults — chosen for: a 30B MoE (3.3B active) on one H100 80GB, prompts of
# ~1.5–3K tokens (schema + question), short structured SQL outputs, ~2–3
# dependent calls per agent run, and a P95 < 5s @ ≥10 RPS SLO. Treat them as a
# starting point to iterate on in Phase 6 (each tuning change → REPORT.md log).

set -euo pipefail

MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"

exec uv run python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port 8000 \
    --served-model-name "$MODEL" \
    `# --- precision: bf16 is H100-native; full weights fit 80GB for a 30B MoE` \
    --dtype bfloat16 \
    `# --- context: prompts are ~1.5–3K tok; 8192 covers prompt+output with headroom.` \
    `#     Smaller max-model-len => more KV blocks free => more concurrency. Raise only` \
    `#     if you hit context-length errors on the biggest BIRD schemas.` \
    --max-model-len 8192 \
    `# --- memory: give most of the 80GB to weights+KV; leave ~10% for activations.` \
    `#     Watch KV-cache % and preemptions on Grafana; back off if you see evictions.` \
    --gpu-memory-utilization 0.90 \
    `# --- concurrency: high seq cap for throughput (10+ RPS x 2–3 calls). Tune down` \
    `#     in Phase 6 if it inflates queue time / TTFT past the SLO.` \
    --max-num-seqs 256 \
    `# --- batching budget per step: balances prefill (big prompts) vs decode.` \
    --max-num-batched-tokens 8192 \
    `# --- prefix caching: the DB schema + system prompt are reused across the 2–3` \
    `#     calls per request and across requests on the same DB — caching that KV` \
    `#     prefix is a direct TTFT/throughput win for THIS workload. (Default-on in` \
    `#     vLLM V1; explicit here to make the intent visible.)` \
    --enable-prefix-caching

# --- Phase 6 tuning levers to try one at a time (uncomment / adjust, re-measure):
#   --kv-cache-dtype fp8           # more KV headroom + throughput; verify eval quality survives
#   --max-num-seqs 128             # lower concurrency to cut tail latency if queue-bound
#   --max-model-len 4096           # frees more KV if schemas allow it
#   --enable-chunked-prefill       # (default-on in V1) chunk big prefills so they don't stall decode
