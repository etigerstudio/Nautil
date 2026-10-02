#!/usr/bin/env bash
set -uo pipefail

run_root=/path/to/run
run_id=sft_v2_2_lora_v2
output_dir="$run_root/runs/$run_id"
persistent_dir="${NAUTIL_DATA:-./data}/runs/$run_id"
status_file="$run_root/${run_id}_status.json"

export PYTHONPATH="/path/to/run/site:$run_root/input"
export NCCL_NVLS_ENABLE=0
export TRANSFORMERS_NO_TF=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export TRITON_CACHE_DIR="$run_root/cache/triton"
export CUDA_CACHE_PATH="$run_root/cache/cuda"

mkdir -p "$persistent_dir" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"
printf '{"status":"running","started_utc":"%s"}\n' "$(date -u +%FT%TZ)" > "$status_file"
python3 -m torch.distributed.run --standalone --nproc_per_node=2 \
  "$run_root/input/train_qwen35_lora_ddp_v2_2.py" \
  --model /path/to/run/model \
  --data "$run_root/prepared_v1/trainval_tokenized.jsonl" \
  --plan "$run_root/prepared_v1/plan.json" \
  --frozen-manifest "$run_root/input/sft_v2_2_frozen_v2.json" \
  --output-dir "$output_dir" \
  --persistent-dir "$persistent_dir"
exit_code=$?
if [ "$exit_code" -eq 0 ]; then
  run_status=completed
else
  run_status=failed
fi
printf '{"status":"%s","exit_code":%s,"finished_utc":"%s"}\n' \
  "$run_status" "$exit_code" "$(date -u +%FT%TZ)" > "$status_file"
exit "$exit_code"
