# RUNBOOK — local development on the Ubuntu VM

This is the operational guide for running everything we built in **Stage A**
against **Nebius** (hosted Qwen3-30B) on your Ubuntu 24.04 dev box — no GPU
needed. Stage B (the H100 run) is one `.env` switch away and is covered at the
end.

> The agent is just an OpenAI-compatible HTTP client. The single switch that
> decides where its LLM calls go is `LLM_BACKEND` in `.env` (see `agent/config.py`).

---

## 0. Prereqs (Ubuntu 24.04)

### 0a. Base packages

```bash
sudo apt-get update
sudo apt-get install -y python3-dev git curl ca-certificates
```

### 0b. Docker Engine + Compose plugin (official apt repo)

```bash
# Remove any distro/old Docker packages that conflict
for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc; do
  sudo apt-get remove -y "$pkg" 2>/dev/null || true
done

# Add Docker's official GPG key
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the Docker apt repository (auto-detects the Ubuntu codename, e.g. noble)
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

Run Docker without `sudo` (needed so `docker compose up` works as your user):

```bash
sudo usermod -aG docker "$USER"
newgrp docker        # applies the group now; or just log out and back in
```

Verify:

```bash
docker --version            # Docker version 2x.x.x
docker compose version      # Docker Compose version v2.x.x  (note: "compose", no hyphen)
docker run --rm hello-world # should print "Hello from Docker!"
```

> The Compose **plugin** (`docker compose`, space) is what `docker-compose.yml`
> in this repo expects — not the old standalone `docker-compose` (hyphen).

### 0c. uv (Python package/run manager)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# add uv to PATH for the current shell (or open a new shell):
source "$HOME/.local/bin/env"
uv --version
```

### 0d. Memory note

The Langfuse stack is heavy (postgres + clickhouse + redis + minio + 2 langfuse
services). Give Docker **≥ 8 GB RAM** (16 GB comfortable) or clickhouse will
OOM-loop. Check with `free -h`.

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

> Full ordered, fail-fast session is in **`H100_PLAYBOOK.md`**. This section is
> just the *bring-up*: how to see what the VM has and how vLLM gets installed.

### B0. Environment check (what's already on the Nebius H100 VM?)

Nebius H100 images usually ship the **NVIDIA driver + CUDA** (and often Docker)
but may lack `uv`, the Docker Compose plugin, or `python3-dev`. After cloning the
repo, run the pre-flight to see exactly what's present vs missing — it installs
nothing, only reports:

```bash
bash scripts/check_env.sh
```

The must-haves it checks: `nvidia-smi` working (driver + the H100 visible),
`python3` + `python3-dev` headers, `uv`, `git`, `docker` + `docker compose`
plugin + daemon reachable, ~70 GB free disk (the 30B weights), and that ports
8000/8001/9090/3000/3001 are free. Install anything marked `✗ MISSING` using the
commands in §0 above.

### B1. Install vLLM (GPU path — the one you'll use)

vLLM (CUDA build, pinned to **0.10.2** in `uv.lock`) and all other deps install
in one step — no manual build, no `nvcc` needed (the wheel bundles its own CUDA
runtime; you only need the NVIDIA **driver** present, which `check_env.sh`
verifies):

```bash
uv sync
# verify the GPU is visible to torch/vLLM before serving:
uv run python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"
```

If `torch.cuda.is_available()` is `False`, the driver isn't visible to the
container/venv — fix that before `start_vllm.sh` (it would otherwise fail at
model load). Then proceed with `H100_PLAYBOOK.md`.

### B2. The run

On the H100 only the LLM backend changes — no agent/eval/dashboard code edits:

1. `.env`: `LLM_BACKEND=h100` and set `HF_TOKEN` (so vLLM can pull the model).
2. `bash scripts/start_vllm.sh` (flags pre-reasoned in the script; Phase 1).
3. Stop the mock exporter; Prometheus now scrapes real vLLM on `:8000`.
4. Eval → `results/eval_baseline.json` (the real baseline).
5. Load test → `uv run python load_test/driver.py --rps 10 --duration 300`;
   diagnose, iterate, save `results/eval_after_tuning.json`.

---

## Appendix — vLLM CPU build (reference only; NOT used here)

We do **not** use this: it requires an **x86 CPU with AVX512**, and our dev VM
(Xeon E5-2660 v2) lacks it, while the H100 box uses the GPU wheel above. Kept for
completeness if you ever need a CPU stand-in on AVX512 hardware.

```bash
# Prereqs: gcc/g++ >= 12.3, libnuma-dev, in a SEPARATE venv (conflicts with the
# CUDA wheel in uv.lock). Build from source:
sudo apt-get install -y gcc-12 g++-12 libnuma-dev
git clone https://github.com/vllm-project/vllm.git && cd vllm
uv venv && source .venv/bin/activate
uv pip install -r requirements/cpu.txt --torch-backend cpu
VLLM_TARGET_DEVICE=cpu uv pip install --editable .
# Serve a small stand-in (reserve 1-2 cores for the framework):
export VLLM_CPU_KVCACHE_SPACE=8
vllm serve Qwen/Qwen3-0.6B --port 8000
```

Ref: https://docs.vllm.ai/en/latest/getting_started/installation/cpu/

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
