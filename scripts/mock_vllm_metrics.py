#!/usr/bin/env python3
"""Mock vLLM /metrics exporter for LOCAL dashboard validation.

Why this exists
---------------
The M1/dev box can't run vLLM, so Prometheus has nothing to scrape and the
Phase 2 Grafana panels stay flat. This stand-in serves a Prometheus-format
/metrics endpoint using the *exact* metric names real vLLM emits, driven by a
simulated workload so every panel visibly reacts. The absolute numbers are
fiction - the point is to prove the dashboard wiring (queries, units,
percentiles, thresholds) is correct before you spend H100 time. On the H100 you
just stop this and start real vLLM on the same port; the dashboard is unchanged.

Run:
    uv run python scripts/mock_vllm_metrics.py          # listens on :8000
    curl http://localhost:8000/metrics                  # what Prometheus scrapes
    curl http://localhost:8000/burst                    # spike load for ~45s

Prometheus already scrapes host:8000 (see infra/prometheus.yml), so nothing
else needs changing.
"""
from __future__ import annotations

import argparse
import math
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_LOCK = threading.Lock()


class Histogram:
    """Minimal Prometheus histogram with cumulative buckets."""

    def __init__(self, name: str, buckets: list[float]) -> None:
        self.name = name
        self.buckets = buckets
        self.counts = [0] * len(buckets)  # cumulative: # obs <= buckets[i]
        self.sum = 0.0
        self.count = 0

    def observe(self, v: float) -> None:
        self.count += 1
        self.sum += v
        for i, b in enumerate(self.buckets):
            if v <= b:
                self.counts[i] += 1

    def render(self) -> str:
        lines = [f"# TYPE {self.name} histogram"]
        for i, b in enumerate(self.buckets):
            lines.append(f'{self.name}_bucket{{le="{b}"}} {self.counts[i]}')
        lines.append(f'{self.name}_bucket{{le="+Inf"}} {self.count}')
        lines.append(f"{self.name}_sum {self.sum}")
        lines.append(f"{self.name}_count {self.count}")
        return "\n".join(lines)


# ── State: counters, gauges, histograms (real vLLM metric names) ────────────
gen_tokens_total = 0.0
prompt_tokens_total = 0.0
request_success_total = 0.0
num_preemptions_total = 0.0

num_requests_running = 0.0
num_requests_waiting = 0.0
gpu_cache_usage_perc = 0.0

e2e = Histogram("vllm:e2e_request_latency_seconds", [0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 4.0, 8.0])
ttft = Histogram("vllm:time_to_first_token_seconds", [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0])
tpot = Histogram("vllm:time_per_output_token_seconds", [0.005, 0.01, 0.02, 0.04, 0.08, 0.16])
queue = Histogram("vllm:request_queue_time_seconds", [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0])

_burst_until = 0.0
_MAX_CONCURRENCY = 32


def _load_now(t: float) -> float:
    """Simulated load factor; 1.0 = capacity. Oscillates, spikes on /burst."""
    base = 0.55 + 0.30 * math.sin(t / 12.0)  # gentle 0.25..0.85 wave
    if t < _burst_until:
        base += 1.0  # push well past capacity so queue/preempt/KV react
    return max(0.05, base + random.uniform(-0.05, 0.05))


def _simulate() -> None:
    """Advance the simulated workload once per tick."""
    global gen_tokens_total, prompt_tokens_total, request_success_total
    global num_preemptions_total, num_requests_running, num_requests_waiting
    global gpu_cache_usage_perc

    tick = 1.0
    while True:
        time.sleep(tick)
        t = time.monotonic()
        load = _load_now(t)
        # Requests completed this tick scales with how busy we are.
        completed = max(0, int(random.gauss(load * 8, 2)))
        with _LOCK:
            num_requests_running = round(min(load, 1.0) * _MAX_CONCURRENCY, 1)
            num_requests_waiting = round(max(0.0, load - 1.0) * _MAX_CONCURRENCY, 1)
            gpu_cache_usage_perc = round(min(0.99, 0.15 + load * 0.7 + random.uniform(-0.03, 0.03)), 4)
            if load > 1.05:
                num_preemptions_total += random.randint(0, int((load - 1.0) * 6) + 1)

            for _ in range(completed):
                p_tok = random.randint(1500, 3000)
                g_tok = random.randint(40, 200)
                # decode cost per token rises with load (memory-bandwidth bound)
                per_tok = 0.008 + 0.02 * max(0.0, load - 0.5) + random.uniform(0, 0.004)
                # queue + prefill rise sharply once we're past capacity
                q = max(0.0, (load - 1.0)) * random.uniform(0.1, 0.6) if load > 1.0 else random.uniform(0, 0.004)
                first_tok = 0.03 + load * 0.08 + q + random.uniform(0, 0.02)
                total = first_tok + g_tok * per_tok

                prompt_tokens_total += p_tok
                gen_tokens_total += g_tok
                request_success_total += 1
                queue.observe(q)
                ttft.observe(first_tok)
                tpot.observe(per_tok)
                e2e.observe(total)


def _render_metrics() -> str:
    with _LOCK:
        out = [
            "# TYPE vllm:num_requests_running gauge",
            f"vllm:num_requests_running {num_requests_running}",
            "# TYPE vllm:num_requests_waiting gauge",
            f"vllm:num_requests_waiting {num_requests_waiting}",
            "# TYPE vllm:gpu_cache_usage_perc gauge",
            f"vllm:gpu_cache_usage_perc {gpu_cache_usage_perc}",
            "# TYPE vllm:kv_cache_usage_perc gauge",
            f"vllm:kv_cache_usage_perc {gpu_cache_usage_perc}",
            "# TYPE vllm:generation_tokens_total counter",
            f"vllm:generation_tokens_total {gen_tokens_total}",
            "# TYPE vllm:prompt_tokens_total counter",
            f"vllm:prompt_tokens_total {prompt_tokens_total}",
            "# TYPE vllm:request_success_total counter",
            f"vllm:request_success_total {request_success_total}",
            "# TYPE vllm:num_preemptions_total counter",
            f"vllm:num_preemptions_total {num_preemptions_total}",
            e2e.render(),
            ttft.render(),
            tpot.render(),
            queue.render(),
            "",
        ]
    return "\n".join(out)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        global _burst_until
        if self.path.startswith("/burst"):
            _burst_until = time.monotonic() + 45.0
            body = b"burst: simulating overload for 45s\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.startswith("/metrics"):
            body = _render_metrics().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"mock vLLM metrics. GET /metrics or /burst\n")

    def log_message(self, *_args) -> None:  # silence per-request logging
        return


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()

    threading.Thread(target=_simulate, daemon=True).start()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"mock vLLM /metrics on http://{args.host}:{args.port}/metrics  (GET /burst to spike load)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
