#!/usr/bin/env bash
# Unit-level check of compute_workers() via pipeline.sh's `sizing` debug role
# (case dispatch hook added for exactly this) -- no fixtures, no export, just
# verifies WORKERS scales with MEM_TARGET_FRACTION/WORKER_MEM_ESTIMATE_MB and
# clamps to >=1 and to a fraction of nproc, using this real machine's actual
# /proc/meminfo and nproc (no mocking needed -- the formula is pure arithmetic
# over real values, so this is a faithful check without a fixture harness).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./lib.sh

TEST_ROOT="/tmp/pipeline_test_sizing_$$"
require_test_path "$TEST_ROOT"
trap 'rm -rf "$TEST_ROOT"' EXIT
mkdir -p "$TEST_ROOT/watch" "$TEST_ROOT/export"

echo "=== scenario_sizing ==="

mem_total_kb=$(awk '/MemTotal/{print $2}' /proc/meminfo)
mem_total_mb=$((mem_total_kb / 1024))
cores=$(nproc)

sizing() {
  PIPELINE_MODE=local PIPELINE_WATCH_DIR="$TEST_ROOT/watch" PIPELINE_EXPORT_ROOT="$TEST_ROOT/export" \
    PIPELINE_MEM_TARGET_FRACTION="$1" PIPELINE_WORKER_MEM_ESTIMATE_MB="$2" \
    "$PIPELINE" sizing --mode local
}

# 1. a tiny fraction / huge per-worker estimate should clamp to the floor of 1
w=$(sizing 0.01 999999999)
assert_eq "$w" "1" "clamps to minimum of 1 worker when budget is far below one worker's estimate"

# 2. an unrealistically low per-worker estimate should clamp to the nproc-derived ceiling
w=$(sizing 0.99 1)
core_ceiling=$(( cores * 8 / 10 ))
assert_eq "$w" "$core_ceiling" "clamps to the nproc-derived ceiling (80% of $cores cores) when memory allows far more"

# 3. doubling MEM_TARGET_FRACTION (while staying under the core ceiling) should
#    roughly double the worker count -- proves it actually scales with the
#    fraction instead of being some other hardcoded number
small_est=$((mem_total_mb / 20))  # deliberately large estimate so both fractions stay core-ceiling-safe
w_low=$(sizing 0.10 "$small_est")
w_high=$(sizing 0.20 "$small_est")
if [[ $w_high -ge $((w_low * 2 - 1)) && $w_high -le $((w_low * 2 + 1)) ]]; then
  pass "worker count scales with MEM_TARGET_FRACTION ($w_low -> $w_high for 0.10 -> 0.20)"
else
  fail "worker count did not scale as expected ($w_low -> $w_high for 0.10 -> 0.20)"
fi

summary
