#!/usr/bin/env bash
set -euo pipefail

# One-click runner:
# 1) Optionally run OpenEvolve search from initial reward and export full candidate manifest
# 2) Run 5 offline comparison groups (OCBA vs Uniform) with different seeds
# 3) Save per-group final metrics + per-round traces + combined summary
#
# Usage:
#   # Fully one-click (auto search + export + offline compare):
#   bash scripts/run_frozenlake_offline_5groups.sh
#   # Reuse an existing manifest:
#   MANIFEST_FILE=logs/openevolve_frozenlake/search_xxx/candidate_manifest.jsonl bash scripts/run_frozenlake_offline_5groups.sh
#   bash scripts/run_frozenlake_offline_5groups.sh /path/to/candidate_manifest.jsonl
# Optional env overrides:
#   SEARCH_IF_MISSING=true ITERATIONS=10 SEARCH_OUTPUT_ROOT=logs/openevolve_frozenlake
#   GROUPS=5 BASE_SEED=42 TOTAL_BUDGET=120000 WARMUP_BUDGET_PER_ARM=6000 DELTA_BUDGET=3000
#   N_ARMS=4 N_ENVS=4 MAP_NAME=4x4 IS_SLIPPERY=true N_EVAL_EPISODES=60

# Ensure execution from project root so `src/...` imports are resolvable.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

if ! python -c "import openevolve" >/dev/null 2>&1; then
  echo "Python dependency 'openevolve' is not available in current environment."
  echo "Please install/activate the environment that contains OpenEvolve first."
  exit 1
fi

MANIFEST_FILE="${MANIFEST_FILE:-${1:-}}"
SEARCH_IF_MISSING="${SEARCH_IF_MISSING:-true}"
ITERATIONS="${ITERATIONS:-10}"
SEARCH_OUTPUT_ROOT="${SEARCH_OUTPUT_ROOT:-logs/openevolve_frozenlake}"

GROUPS="${GROUPS:-5}"
BASE_SEED="${BASE_SEED:-42}"
TOTAL_BUDGET="${TOTAL_BUDGET:-120000}"
WARMUP_BUDGET_PER_ARM="${WARMUP_BUDGET_PER_ARM:-6000}"
DELTA_BUDGET="${DELTA_BUDGET:-3000}"
N_ARMS="${N_ARMS:-4}"
N_ENVS="${N_ENVS:-4}"
MAP_NAME="${MAP_NAME:-4x4}"
IS_SLIPPERY="${IS_SLIPPERY:-true}"
N_STEPS="${N_STEPS:-256}"
BATCH_SIZE="${BATCH_SIZE:-64}"
N_EPOCHS="${N_EPOCHS:-4}"
N_EVAL_EPISODES="${N_EVAL_EPISODES:-60}"
EARLY_STOP_ENABLE="${EARLY_STOP_ENABLE:-true}"
EARLY_STOP_Z="${EARLY_STOP_Z:-1.96}"
EARLY_STOP_MARGIN="${EARLY_STOP_MARGIN:-0.01}"
MIN_ROUNDS_BEFORE_STOP="${MIN_ROUNDS_BEFORE_STOP:-2}"
ELIMINATION_ENABLE="${ELIMINATION_ENABLE:-true}"
ELIMINATION_Z="${ELIMINATION_Z:-1.0}"
MIN_ACTIVE_ARMS="${MIN_ACTIVE_ARMS:-2}"

to_bool_flag() {
  local value
  value="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
  local true_flag="$2"
  local false_flag="$3"
  if [[ "$value" == "true" || "$value" == "1" || "$value" == "yes" ]]; then
    echo "$true_flag"
  else
    echo "$false_flag"
  fi
}

run_search_and_export_manifest() {
  local search_ts search_dir is_slippery_flag early_stop_flag elimination_flag
  search_ts="$(date +%Y%m%d_%H%M%S)"
  search_dir="${SEARCH_OUTPUT_ROOT}/search_${search_ts}"
  mkdir -p "${search_dir}"

  is_slippery_flag="$(to_bool_flag "$IS_SLIPPERY" "--is-slippery" "--no-is-slippery")"
  early_stop_flag="$(to_bool_flag "$EARLY_STOP_ENABLE" "--early-stop-enable" "--no-early-stop-enable")"
  elimination_flag="$(to_bool_flag "$ELIMINATION_ENABLE" "--elimination-enable" "--no-elimination-enable")"

  echo "No valid manifest provided. Running OpenEvolve search first..."
  echo "Search output dir: ${search_dir}"
  python -m src.openevolve_frozenlake.run_search \
    --iterations "${ITERATIONS}" \
    --output-dir "${search_dir}" \
    --allocation-strategy ocba \
    --total-eval-budget "${TOTAL_BUDGET}" \
    --warmup-budget-per-arm "${WARMUP_BUDGET_PER_ARM}" \
    --delta-budget "${DELTA_BUDGET}" \
    --n-arms "${N_ARMS}" \
    --n-envs "${N_ENVS}" \
    --seed "${BASE_SEED}" \
    --map-name "${MAP_NAME}" \
    "${is_slippery_flag}" \
    --n-steps "${N_STEPS}" \
    --batch-size "${BATCH_SIZE}" \
    --n-epochs "${N_EPOCHS}" \
    --n-eval-episodes "${N_EVAL_EPISODES}" \
    "${early_stop_flag}" \
    --early-stop-z "${EARLY_STOP_Z}" \
    --early-stop-margin "${EARLY_STOP_MARGIN}" \
    --min-rounds-before-stop "${MIN_ROUNDS_BEFORE_STOP}" \
    "${elimination_flag}" \
    --elimination-z "${ELIMINATION_Z}" \
    --min-active-arms "${MIN_ACTIVE_ARMS}" \
    --export-candidates-manifest "${search_dir}/candidate_manifest.jsonl"

  MANIFEST_FILE="${search_dir}/candidate_manifest.jsonl"
}

if [[ -z "${MANIFEST_FILE}" || ! -s "${MANIFEST_FILE}" ]]; then
  SEARCH_IF_MISSING_NORM="$(printf '%s' "${SEARCH_IF_MISSING}" | tr '[:upper:]' '[:lower:]')"
  if [[ "${SEARCH_IF_MISSING_NORM}" == "true" || "${SEARCH_IF_MISSING}" == "1" || "${SEARCH_IF_MISSING_NORM}" == "yes" ]]; then
    run_search_and_export_manifest
  else
    echo "Manifest is missing/empty and SEARCH_IF_MISSING=false: ${MANIFEST_FILE}"
    exit 1
  fi
fi

if [[ ! -s "${MANIFEST_FILE}" ]]; then
  echo "Manifest file not found or empty after search: ${MANIFEST_FILE}"
  exit 1
fi

TS="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="logs/openevolve_frozenlake/offline_compare_5groups_${TS}"
mkdir -p "$RUN_ROOT"

echo "Run root: $RUN_ROOT"
echo "Manifest: $MANIFEST_FILE"
echo "Groups: $GROUPS (base seed: $BASE_SEED)"

for ((g=1; g<=GROUPS; g++)); do
  SEED=$((BASE_SEED + (g-1) * 1000))
  GROUP_DIR="${RUN_ROOT}/group_${g}"
  mkdir -p "$GROUP_DIR"

  OUT_JSONL="${GROUP_DIR}/offline_compare.jsonl"
  OUT_TRACE="${GROUP_DIR}/offline_compare_rounds.jsonl"
  OUT_SUMMARY_JSON="${GROUP_DIR}/summary.json"
  OUT_SUMMARY_CSV="${GROUP_DIR}/summary.csv"
  IS_SLIPPERY_FLAG="$(to_bool_flag "$IS_SLIPPERY" "--is-slippery" "--no-is-slippery")"
  EARLY_STOP_FLAG="$(to_bool_flag "$EARLY_STOP_ENABLE" "--early-stop-enable" "--no-early-stop-enable")"
  ELIMINATION_FLAG="$(to_bool_flag "$ELIMINATION_ENABLE" "--elimination-enable" "--no-elimination-enable")"

  echo "========== Group ${g}/${GROUPS} | seed=${SEED} =========="
  python -m src.openevolve_frozenlake.run_offline_compare \
    --manifest-file "$MANIFEST_FILE" \
    --output "$OUT_JSONL" \
    --round-trace-output "$OUT_TRACE" \
    --seed "$SEED" \
    --total-eval-budget "$TOTAL_BUDGET" \
    --warmup-budget-per-arm "$WARMUP_BUDGET_PER_ARM" \
    --delta-budget "$DELTA_BUDGET" \
    --n-arms "$N_ARMS" \
    --n-envs "$N_ENVS" \
    --map-name "$MAP_NAME" \
    "$IS_SLIPPERY_FLAG" \
    --n-steps "$N_STEPS" \
    --batch-size "$BATCH_SIZE" \
    --n-epochs "$N_EPOCHS" \
    --n-eval-episodes "$N_EVAL_EPISODES" \
    "$EARLY_STOP_FLAG" \
    --early-stop-z "$EARLY_STOP_Z" \
    --early-stop-margin "$EARLY_STOP_MARGIN" \
    --min-rounds-before-stop "$MIN_ROUNDS_BEFORE_STOP" \
    "$ELIMINATION_FLAG" \
    --elimination-z "$ELIMINATION_Z" \
    --min-active-arms "$MIN_ACTIVE_ARMS"

  python -m src.process_results.frozenlake_offline_compare_summary \
    --input "$OUT_JSONL" \
    --csv-output "$OUT_SUMMARY_CSV" > "$OUT_SUMMARY_JSON"
done

COMBINED_JSONL="${RUN_ROOT}/combined_offline_compare.jsonl"
COMBINED_SUMMARY_JSON="${RUN_ROOT}/combined_summary.json"
COMBINED_SUMMARY_CSV="${RUN_ROOT}/combined_summary.csv"

: > "$COMBINED_JSONL"
for ((g=1; g<=GROUPS; g++)); do
  cat "${RUN_ROOT}/group_${g}/offline_compare.jsonl" >> "$COMBINED_JSONL"
done

python -m src.process_results.frozenlake_offline_compare_summary \
  --input "$COMBINED_JSONL" \
  --csv-output "$COMBINED_SUMMARY_CSV" > "$COMBINED_SUMMARY_JSON"

echo "==============================================="
echo "All done."
echo "Per-group results: ${RUN_ROOT}/group_*/"
echo "Combined summary:  ${COMBINED_SUMMARY_JSON}"
echo "Combined CSV:      ${COMBINED_SUMMARY_CSV}"
echo "Round traces:      ${RUN_ROOT}/group_*/offline_compare_rounds.jsonl"
