#!/usr/bin/env bash
# Local mode, recordings trickle in progressively (simulating a collector
# process still running on this machine) instead of all existing upfront —
# verifies exporting actually starts while new recordings are still arriving
# (not after collection "finishes"), and that collection_done doesn't fire
# while the feed is still active.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./lib.sh

TEST_ROOT="/tmp/pipeline_test_local_live_$$"
require_test_path "$TEST_ROOT"
trap 'kill -- "-${pid:-}" 2>/dev/null || true; kill "${collector_pid:-0}" 2>/dev/null || true; rm -rf "$TEST_ROOT"' EXIT
WATCH="$TEST_ROOT/watch"
EXPORT_ROOT="$TEST_ROOT/export"
mkdir -p "$WATCH" "$EXPORT_ROOT"

echo "=== scenario_local_live ==="

IDS=(live01 live02 live03 live04 live05 live06)
INTERVAL_S=15  # span (5 gaps * 15s = 75s) must exceed local mode's 60s candidate-maturity
              # floor, so the first arrival matures and can be exported *before*
              # the collector finishes adding the rest -> genuine overlap.

# Fake "collector": writes one fixture every INTERVAL_S seconds, then marks itself done.
(
  for id in "${IDS[@]}"; do
    make_fixture local "$WATCH" "$id" success
    sleep "$INTERVAL_S"
  done
  date +%s > "$TEST_ROOT/collector_finished_at"
) &
collector_pid=$!

export PIPELINE_MODE=local
export PIPELINE_WATCH_DIR="$WATCH"
export PIPELINE_EXPORT_ROOT="$EXPORT_ROOT"
export PIPELINE_EXPORT_CLI="$FAKE_CLI"
export PIPELINE_EXPORT_CONFIG="$TEST_ROOT/fake_config.json"
touch "$PIPELINE_EXPORT_CONFIG"
export PIPELINE_CHUNK=2
export PIPELINE_POLL_SECONDS=3
export PIPELINE_IDLE_MINUTES=1

start_epoch=$(date +%s)
"$PIPELINE" supervisor >"$TEST_ROOT/supervisor.log" 2>&1 &
pid=$!
wait "$collector_pid"
wait "$pid" || true
echo "pipeline finished"

# --- assertions ---
committed=$(committed_ids "$EXPORT_ROOT" | sort -u)
for id in "${IDS[@]}"; do
  count=$(grep -cx "$id" <<< "$committed" || true)
  assert_eq "$count" "1" "$id committed exactly once"
done

collector_finished_at=$(cat "$TEST_ROOT/collector_finished_at")
first_commit_at=$(stat -c %Y "$EXPORT_ROOT"/batch_*/dataset 2>/dev/null | sort -n | head -1)
if [[ -n "$first_commit_at" && "$first_commit_at" -lt "$collector_finished_at" ]]; then
  pass "first export committed ($((first_commit_at - start_epoch))s in) before collector finished ($((collector_finished_at - start_epoch))s in) -> pull+export overlapped"
else
  fail "first export ($first_commit_at) did not precede collector finishing ($collector_finished_at) -- no overlap observed"
fi

collection_done_at=$(stat -c %Y "$EXPORT_ROOT/.pipeline/collection_done" 2>/dev/null || echo 0)
if [[ "$collection_done_at" -gt "$collector_finished_at" ]]; then
  pass "collection_done ($collection_done_at) fired after collector finished ($collector_finished_at), not before"
else
  fail "collection_done ($collection_done_at) fired at/before collector finished ($collector_finished_at) -- premature idle detection"
fi

merged="${EXPORT_ROOT}_merged"
if [[ -f "$merged/meta/info.json" ]]; then
  actual_episodes=$(python3 -c "import json;print(json.load(open('$merged/meta/info.json'))['total_episodes'])")
  assert_eq "$actual_episodes" "${#IDS[@]}" "merged dataset episode count"
else
  fail "merged dataset exists at $merged"
fi

summary
