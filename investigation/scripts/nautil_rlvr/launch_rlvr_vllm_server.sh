#!/usr/bin/env bash
set -euo pipefail
gpu=${1:?gpu}; port=${2:?port}; util=${3:?gpu_memory_utilization}; shift 3
source /path/to/run/vllm_v3/scripts/env_vllm_v3.sh     # existing vLLM 0.30 venv, read-only
export CUDA_VISIBLE_DEVICES="$gpu"
export OMP_NUM_THREADS=8
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
export VLLM_SERVER_DEV_MODE=1
exec vllm serve /path/to/run/model \
  --served-model-name qwen35-9b-base \
  --host 127.0.0.1 --port "$port" \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --language-model-only \
  --generation-config vllm \
  --enable-lora --max-lora-rank 16 --max-loras 4 --max-cpu-loras 8 \
  --enable-prefix-caching \
  --enable-sleep-mode \
  --gpu-memory-utilization "$util" \
  --max-num-seqs 256 \
  --enable-prompt-tokens-details \
  --seed 20260925 \
  "$@"
