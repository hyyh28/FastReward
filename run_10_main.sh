#!/usr/bin/env bash
set -euo pipefail

# Run 10 parallel main jobs with separate process logs.
# Usage:
#   bash run_10_main.sh
#   JOBS=10 PYTHON_BIN=python3 bash run_10_main.sh

JOBS="${JOBS:-10}"
PYTHON_BIN="${PYTHON_BIN:-python}"
ENTRY_MODULE="${ENTRY_MODULE:-src.main}"

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

BATCH_ID="$(date +"%Y%m%d_%H%M%S")"
BATCH_LOG_DIR="$ROOT_DIR/logs/batch_$BATCH_ID"
mkdir -p "$BATCH_LOG_DIR"

echo "Starting $JOBS parallel runs..."
echo "Batch logs: $BATCH_LOG_DIR"

pids=()

cleanup() {
  echo
  echo "Interrupted. Stopping child processes..."
  for pid in "${pids[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup INT TERM

for i in $(seq 1 "$JOBS"); do
  out_file="$BATCH_LOG_DIR/run_${i}.log"
  echo "[$(date +"%H:%M:%S")] Launching run $i -> $out_file"
  "$PYTHON_BIN" -m "$ENTRY_MODULE" >"$out_file" 2>&1 &
  pids+=("$!")

  # RunLogger uses second-level run_id; avoid same-second collisions.
  sleep 1
done

fail_count=0
for idx in "${!pids[@]}"; do
  pid="${pids[$idx]}"
  run_no="$((idx + 1))"
  if wait "$pid"; then
    echo "Run $run_no finished successfully (pid=$pid)"
  else
    echo "Run $run_no failed (pid=$pid)"
    fail_count=$((fail_count + 1))
  fi
done

echo "All runs finished. Failed: $fail_count / $JOBS"
echo "Per-process logs: $BATCH_LOG_DIR"
