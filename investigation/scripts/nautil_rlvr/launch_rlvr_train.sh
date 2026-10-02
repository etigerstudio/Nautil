#!/usr/bin/env bash
set -euo pipefail
nproc=${1:?nproc}; log=${2:?log}; shift 2
[ "${1:-}" = "--" ] && shift
root=/path/to/run/rlvr_v1
allowed='^(API_KEY)='
if [ ! -t 0 ]; then
  while IFS= read -r line || [ -n "$line" ]; do
    if [[ "$line" =~ $allowed ]]; then
      name=${line%%=*}; value=${line#*=}; value=${value%\"}; value=${value#\"}; value=${value%\'}; value=${value#\'}
      export "$name=$value"
    fi
  done
fi
export PYTHONPATH="/path/to/run/site:$root/extra_site:$root/scripts"
export NCCL_NVLS_ENABLE=0 TRANSFORMERS_NO_TF=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="$root/cache/triton" CUDA_CACHE_PATH="$root/cache/cuda" TMPDIR="$root/tmp"
mkdir -p "$root/cache/triton" "$root/cache/cuda" "$root/tmp" "$(dirname "$log")"
cd "$root/scripts"
if [ "$nproc" = "1" ]; then
  export CUDA_VISIBLE_DEVICES=${TRAINER_GPU:-1}
  setsid nohup python3 -m nautil_rlvr.train "$@" > "$log" 2>&1 < /dev/null &
else
  setsid nohup python3 -m torch.distributed.run --standalone --nproc_per_node="$nproc" \
    -m nautil_rlvr.train "$@" > "$log" 2>&1 < /dev/null &
fi
echo "{\"launched_pid\": $!, \"log\": \"$log\"}"
