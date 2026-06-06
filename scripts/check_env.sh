#!/usr/bin/env bash
#
# Environment pre-flight check. Run on ANY box (dev VM or the H100) to see what
# this assignment needs and what's already present, before you start installing.
#
#   bash scripts/check_env.sh
#
# Nothing is installed or changed - this only reports.

set -uo pipefail

ok()   { printf "  \033[32m✓\033[0m %s\n" "$1"; }
miss() { printf "  \033[31m✗ MISSING\033[0m %s\n" "$1"; }
info() { printf "  \033[34m·\033[0m %s\n" "$1"; }

have() { command -v "$1" >/dev/null 2>&1; }

echo "==================== SYSTEM ===================="
info "OS:   $(. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME" || uname -s)"
info "Arch: $(uname -m)"
info "CPU:  $(lscpu 2>/dev/null | sed -n 's/^Model name:\s*//p' | head -1)"
info "RAM:  $(free -h 2>/dev/null | awk '/^Mem:/{print $2" total, "$7" available"}')"
info "Disk (home): $(df -h "$HOME" 2>/dev/null | awk 'NR==2{print $4" free of "$2}')  # need ~70GB for the 30B model"
if grep -q avx512 /proc/cpuinfo 2>/dev/null; then
  ok "AVX512 present (CPU-vLLM build would be possible)"
else
  info "AVX512 absent (CPU-vLLM not possible here; irrelevant on a GPU box)"
fi

echo "==================== GPU / CUDA ===================="
if have nvidia-smi; then
  ok "nvidia-smi present"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | sed 's/^/      GPU: /'
else
  miss "nvidia-smi  (no NVIDIA driver visible — required to serve the model on the H100)"
fi
if have nvcc; then ok "nvcc: $(nvcc --version | sed -n 's/.*release //p' | head -1)"; else info "nvcc absent (not required: the vLLM wheel ships its own CUDA runtime)"; fi

echo "==================== TOOLCHAIN ===================="
if have python3; then ok "python3: $(python3 --version 2>&1)"; else miss "python3"; fi
# python3-dev (Python.h) — vLLM's torch.compile path needs headers
PYINC="$(python3 -c 'import sysconfig; print(sysconfig.get_path("include"))' 2>/dev/null)"
if [ -n "$PYINC" ] && [ -f "$PYINC/Python.h" ]; then ok "python3-dev headers ($PYINC/Python.h)"; else miss "python3-dev  (sudo apt-get install -y python3-dev)"; fi
if have uv; then ok "uv: $(uv --version 2>&1)"; else miss "uv  (curl -LsSf https://astral.sh/uv/install.sh | sh)"; fi
if have git; then ok "git: $(git --version | awk '{print $3}')"; else miss "git"; fi
if have gcc; then ok "gcc: $(gcc -dumpversion 2>/dev/null)"; else info "gcc absent (only needed for a CPU source build)"; fi

echo "==================== DOCKER ===================="
if have docker; then
  ok "docker: $(docker --version | awk '{print $3}' | tr -d ,)"
  if docker compose version >/dev/null 2>&1; then ok "docker compose plugin: $(docker compose version --short 2>/dev/null)"; else miss "docker compose plugin  (need 'docker compose', not legacy 'docker-compose')"; fi
  if docker info >/dev/null 2>&1; then ok "docker daemon reachable as this user"; else miss "docker daemon not reachable (start it, or add user to 'docker' group)"; fi
else
  miss "docker  (needed for the Prometheus/Grafana/Langfuse stack)"
fi

echo "==================== PORTS (should be FREE before starting) ===================="
for p in 8000 8001 9090 3000 3001; do
  if (have ss && ss -ltn 2>/dev/null | grep -q ":$p ") || (have lsof && lsof -iTCP:"$p" -sTCP:LISTEN >/dev/null 2>&1); then
    info "port $p: in use"
  else
    ok "port $p: free"
  fi
done

echo "==================== SUMMARY ===================="
echo "  Fix anything marked ✗ MISSING above. On a fresh H100 box the usual gaps"
echo "  are: uv, docker compose plugin, python3-dev. Then run 'uv sync'."
