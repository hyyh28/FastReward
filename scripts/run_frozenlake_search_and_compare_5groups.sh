#!/usr/bin/env bash
set -euo pipefail

# End-to-end pipeline:
# 1) Run OpenEvolve search (OCBA online)
# 2) Export candidate manifest from generated programs
# 3) Run 5-group paired offline comparison (OCBA vs Uniform)
#
# Usage:
#   bash scripts/run_frozenlake_search_and_compare_5groups.sh
# Optional env vars:
#   ITERATIONS=10 OUTPUT_DIR=logs/openevolve_frozenlake GROUPS=5 BASE_SEED=42

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

ITERATIONS="${ITERATIONS:-10}"
OUTPUT_DIR="${OUTPUT_DIR:-logs/openevolve_frozenlake}"
GROUPS="${GROUPS:-5}"
BASE_SEED="${BASE_SEED:-42}"

mkdir -p "${OUTPUT_DIR}"

MANIFEST_FILE="${OUTPUT_DIR}/candidate_manifest.jsonl"

python -m src.openevolve_frozenlake.run_search \
  --iterations "${ITERATIONS}" \
  --output-dir "${OUTPUT_DIR}" \
  --allocation-strategy ocba \
  --export-candidates-manifest "${MANIFEST_FILE}"

if [[ ! -s "${MANIFEST_FILE}" ]]; then
  echo "Manifest is empty or missing: ${MANIFEST_FILE}"
  exit 1
fi

GROUPS="${GROUPS}" BASE_SEED="${BASE_SEED}" MANIFEST_FILE="${MANIFEST_FILE}" \
  bash scripts/run_frozenlake_offline_5groups.sh

echo "Pipeline finished."
echo "Search output: ${OUTPUT_DIR}"
echo "Manifest: ${MANIFEST_FILE}"
