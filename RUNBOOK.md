# Runbook: local development (Ubuntu, no GPU)

The operational guide for building and validating the whole pipeline against a
hosted OpenAI-compatible endpoint on an Ubuntu box, with no GPU. The real GPU run
is one `.env` switch away and is covered at the end (see also `H100_PLAYBOOK.md`).

> The agent is just an OpenAI-compatible HTTP client. The single switch that
> decides where its LLM calls go is `LLM_BACKEND` in `.env` (see `agent/config.py`).

---

## 0. Prerequisites (Ubuntu 24.04)

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
newgrp docker        # applies the group now, or just log out and back in
```

Verify:

```bash
docker --version            # Docker version 2x.x.x
docker compose version      # Docker Compose version v2.x.x  (note: "compose", no hyphen)
docker run --rm hello-world # should print "Hello from Docker!"
```

> The Compose plugin (`docker compose`, with a space) is what `docker-compose.yml`
> in this repo expects, not the old standalone `docker-compose` (hyphen).

### 0c. uv (Python package and run manager)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# add uv to PATH for the current shell (or open a new shell):
source "$HOME/.local/bin/env"
uv --version
```

### 0d. Memory note

The Langfuse stack is heavy (postgres, clickhouse, redis, minio, and 2 Langfuse
services). Give Docker at least **8GB RAM** (16GB comfortable) or clickhouse will
OOM-loop. Check with `free -h`.

---

## 1. Setup

```bash
git clone <your-repo-url>
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

## 2. Run and test the agent

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

The response includes `history`, the generate_sql, verify, (revise) trail. As a
check, find at least one question where `history` contains a `revise` entry (verify
returned `ok:false` and the loop re-ran). A reliable way to trigger one is a
question whose obvious first query returns zero rows or duplicate rows.

---

## 3. Langfuse tracing

1. Open `http://localhost:3001`, sign up (local, instant), create or confirm a project.
2. Settings, API Keys, create. Copy the public and secret keys into `.env`:
   ```
   LANGFUSE_PUBLIC_KEY=pk-...
   LANGFUSE_SECRET_KEY=sk-...
   LANGFUSE_HOST=http://localhost:3001
   ```
3. Restart the agent server. It reads the keys at startup and attaches the callback
   handler (already wired in `agent/server.py`).
4. Fire about 10 questions (loop the curl above, or just run the eval in step 4).
5. In Langfuse you should see each run as a trace with `generate_sql`, `verify`, and
   sometimes `revise` as nested spans (prompt, response, latency, tokens). Tags
   `source=eval` and `db_id=...` are attached as filterable metadata.

---

## 4. Eval harness (dev validation)

With the agent running:

```bash
uv run python evals/run_eval.py --out results/eval_baseline.json
cat results/eval_baseline.json | python3 -m json.tool | head -40
```

Look at `pass_rate_by_iteration`. If iter 0 is about equal to iter 2, the loop is
not earning its keep. If iter 2 is higher than iter 0, it is. These dev numbers are
throwaway. The real baseline comes from the H100. This step only proves the harness
works end-to-end.

---

## 5. Dashboard (local, with the mock exporter)

The dev box cannot run vLLM, so use the mock to give Prometheus something to scrape
on `:8000` (the exact port and metric names real vLLM uses):

```bash
uv run python scripts/mock_vllm_metrics.py        # serves :8000/metrics
```

- Open Grafana `:3000`, the **vLLM serving** dashboard (auto-provisioned).
- Every panel should be moving within about 15s. To make latency, queue, KV, and
  preemptions jump for a screenshot: `curl http://localhost:8000/burst` (spikes load
  for about 45s).
- This validates the queries, units, percentiles, and thresholds. The numbers are
  synthetic, the wiring is real. On the H100 you stop this and start real vLLM on the
  same port, and the dashboard is unchanged.

---

## Optional: real vLLM metrics on CPU

To validate the dashboard against a real vLLM exporter (not the mock), run a tiny
model on CPU. This is plumbing only, the model's SQL quality is irrelevant.

```bash
# CPU vLLM install: https://docs.vllm.ai/en/latest/getting_started/installation/cpu/
# then serve a small stand-in on :8000
uv run python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-0.6B --port 8000 --device cpu
```

Set `LLM_BACKEND=cpu` in `.env` if you also want the agent to talk to it.

---

## Moving to the H100

> Full ordered, fail-fast session is in **`H100_PLAYBOOK.md`**. This section is just
> the bring-up: how to see what the box has and how vLLM gets installed.

### Environment check

A fresh GPU box usually ships the NVIDIA driver and CUDA (and often Docker) but may
lack `uv`, the Docker Compose plugin, or `python3-dev`. After cloning the repo, run
the pre-flight to see exactly what is present versus missing. It installs nothing,
it only reports:

```bash
bash scripts/check_env.sh
```

It checks `nvidia-smi` (driver and GPU visible), `python3` plus `python3-dev`
headers, `uv`, `git`, `docker` plus the compose plugin and a reachable daemon, about
70GB of free disk (the 30B weights), and that ports 8000/8001/9090/3000/3001 are
free. Install anything marked MISSING using the commands in section 0.

### Install vLLM (GPU path)

vLLM (CUDA build, pinned to 0.10.2 in `uv.lock`) and all other deps install in one
step. No manual build, no `nvcc` needed (the wheel bundles its own CUDA runtime, you
only need the NVIDIA driver present, which `check_env.sh` verifies):

```bash
uv sync
# verify the GPU is visible to torch/vLLM before serving:
uv run python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"
```

If `torch.cuda.is_available()` is `False`, the driver is not visible to the
container or venv. Fix that before `start_vllm.sh`, which would otherwise fail at
model load.

**Known issue, transformers 5.x.** vLLM 0.10.2 crashes at tokenizer load with
`Qwen2Tokenizer has no attribute all_special_tokens_extended` because the lock
resolves `transformers` to 5.x. Fix it with `uv add` (not `uv pip install`, because
`uv run` re-syncs the venv to the lock on every call and reverts a bare pip
install):

```bash
uv add 'transformers>=4.51,<5'
uv run python -c "import transformers; print(transformers.__version__)"   # expect 4.5x
```
This rewrites `pyproject.toml` and `uv.lock`. Commit both so the pin sticks.

### The run

On the H100 only the LLM backend changes, no agent, eval, or dashboard code edits:

1. `.env`: `LLM_BACKEND=h100` and set `HF_TOKEN` (so vLLM can pull the model).
2. `bash scripts/start_vllm.sh` (flags pre-reasoned in the script).
3. Stop the mock exporter. Prometheus now scrapes real vLLM on `:8000`.
4. Eval to `results/eval_baseline.json` (the real baseline).
5. Load test: `uv run python load_test/driver.py --rps 10 --duration 300`, then
   diagnose, iterate, and save `results/eval_after_tuning.json`.

---

## Appendix: vLLM CPU build (reference only, not used here)

This is not used in this repo. It requires an x86 CPU with AVX512. Kept for
completeness if you ever need a CPU stand-in on AVX512 hardware.

```bash
# Prereqs: gcc/g++ >= 12.3, libnuma-dev, in a SEPARATE venv (conflicts with the
# CUDA wheel in uv.lock). Build from source:
sudo apt-get install -y gcc-12 g++-12 libnuma-dev
git clone https://github.com/vllm-project/vllm.git && cd vllm
uv venv && source .venv/bin/activate
uv pip install -r requirements/cpu.txt --torch-backend cpu
VLLM_TARGET_DEVICE=cpu uv pip install --editable .
# Serve a small stand-in (reserve 1 to 2 cores for the framework):
export VLLM_CPU_KVCACHE_SPACE=8
vllm serve Qwen/Qwen3-0.6B --port 8000
```

Ref: https://docs.vllm.ai/en/latest/getting_started/installation/cpu/

---

## Quick port reference

| Port | Service |
|------|---------|
| 8000 | vLLM (or mock exporter) |
| 8001 | agent server |
| 9090 | Prometheus |
| 3000 | Grafana (admin/admin) |
| 3001 | Langfuse |
