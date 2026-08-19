#!/usr/bin/env bash
# Dedicated wall-clock proof that workers genuinely overlap, isolated from
# scenario_local_backlog's bisection/quarantine logic (there, batch
# composition is claim-order-dependent, so a tight timing assertion would be
# flaky). Here CHUNK=1 forces every slow fixture into its own singleton
# batch deterministically, so the concurrency proof isn't at the mercy of
# how bisection happens to group things.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./lib.sh

TEST_ROOT="/tmp/pipeline_test_concurrency_$$"
require_test_path "$TEST_ROOT"
trap 'kill -- "-${pid:-}" 2>/dev/null || true; rm -rf "$TEST_ROOT"' EXIT
WATCH="$TEST_ROOT/watch"
EXPORT_ROOT="$TEST_ROOT/export"
mkdir -p "$WATCH" "$EXPORT_ROOT"

echo "=== scenario_concurrency ==="

SLEEP_S=6
N=6  # >= WORKERS on any reasonable machine, so every worker gets at least one
for i in $(seq -w 1 "$N"); do make_fixture local "$WATCH" "slow$i" "sleep:$SLEEP_S"; done

export PIPELINE_MODE=local
export PIPELINE_WATCH_DIR="$WATCH"
export PIPELINE_EXPORT_ROOT="$EXPORT_ROOT"
export PIPELINE_EXPORT_CLI="$FAKE_CLI"
export PIPELINE_EXPORT_CONFIG="$TEST_ROOT/fake_config.json"
touch "$PIPELINE_EXPORT_CONFIG"
export PIPELINE_CHUNK=1
export PIPELINE_POLL_SECONDS=3
export PIPELINE_IDLE_MINUTES=1

"$PIPELINE" supervisor >"$TEST_ROOT/supervisor.log" 2>&1 &
pid=$!

# Only wait for the N batches to commit, not for the whole idle-timeout tail
# (which would dominate wall time and isn't part of what we're proving here).
# Local mode's mandatory ~60s candidate-maturity wait (is_candidate requires
# recording.rrd to sit untouched for 60s) happens before any claim, so budget
# for that plus the actual export work.
deadline=$((SECONDS + 60 + (N * SLEEP_S) + 40))  # +40: systemd-run/claim overhead per wave, observed ~4-10s each
while [[ $(committed_ids "$EXPORT_ROOT" 2>/dev/null | wc -l) -lt $N ]]; do
  [[ $SECONDS -lt $deadline ]] || break
  sleep 1
done
kill -- "-$pid" 2>/dev/null || true
wait "$pid" 2>/dev/null || true

committed_count=$(committed_ids "$EXPORT_ROOT" 2>/dev/null | wc -l)
assert_eq "$committed_count" "$N" "all $N slow recordings committed"

# Measure the export phase itself (first commit -> last commit), excluding
# local mode's fixed candidate-maturity wait, which is orthogonal to whether
# workers overlap.
mtimes=$(stat -c %Y "$EXPORT_ROOT"/batch_*/dataset 2>/dev/null | sort -n)
first=$(head -1 <<< "$mtimes")
last=$(tail -1 <<< "$mtimes")
span=$((last - first + 1))  # +1: same-second commits would otherwise show span=0
serial_floor=$((N * SLEEP_S))
echo "export phase span: ${span}s (first->last commit); fully-serial floor would be ${serial_floor}s"
if [[ $span -lt $serial_floor ]]; then
  pass "export phase span (${span}s) below serial floor (${serial_floor}s) -> workers overlapped"
else
  fail "export phase span (${span}s) not below serial floor (${serial_floor}s)"
fi

summary
