#!/usr/bin/env bash
# Remote mode (real SSH/rsync), recordings arrive progressively on the remote
# side while the pipeline is already running -- verifies pull+export overlap
# and that idle-detection doesn't fire while the remote feed is still active.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./lib.sh

RUN_ID="run_live_$$"
REMOTE_DIR="$REMOTE_TEST_ROOT/$RUN_ID"
require_test_path "$REMOTE_DIR"

TEST_ROOT="/tmp/pipeline_test_remote_live_$$"
require_test_path "$TEST_ROOT"
cleanup() {
  kill -- "-${pid:-}" 2>/dev/null || true
  kill "${collector_pid:-0}" 2>/dev/null || true
  ssh -o BatchMode=yes "$REMOTE_TEST_HOST" "rm -rf '$REMOTE_DIR'" 2>/dev/null || true
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT
mkdir -p "$TEST_ROOT/watch" "$TEST_ROOT/export"

echo "=== scenario_remote_live ==="

IDS=(rlive01 rlive02 rlive03 rlive04)
INTERVAL_S=12  # remote mode has no candidate-maturity floor (unlike local), so
              # this just needs to comfortably outlast a couple of rsync polls

(
  for id in "${IDS[@]}"; do
    make_fixture remote "$REMOTE_DIR" "$id" success
    sleep "$INTERVAL_S"
  done
  date +%s > "$TEST_ROOT/collector_finished_at"
) &
collector_pid=$!

export PIPELINE_MODE=remote
export PIPELINE_REMOTE_HOST="$REMOTE_TEST_HOST"
export PIPELINE_REMOTE_DIRS="$REMOTE_DIR"
export PIPELINE_WATCH_DIR="$TEST_ROOT/watch/placeholder"
export PIPELINE_EXPORT_ROOT="$TEST_ROOT/export"
export PIPELINE_EXPORT_CLI="$FAKE_CLI"
export PIPELINE_EXPORT_CONFIG="$TEST_ROOT/fake_config.json"
touch "$PIPELINE_EXPORT_CONFIG"
export PIPELINE_CHUNK=1
export PIPELINE_POLL_SECONDS=4
export PIPELINE_IDLE_MINUTES=1

start_epoch=$(date +%s)
"$PIPELINE" supervisor >"$TEST_ROOT/supervisor.log" 2>&1 &
pid=$!
wait "$collector_pid"
wait "$pid" || true
echo "pipeline finished"

committed=$(committed_ids "$TEST_ROOT/export" | sort -u)
for id in "${IDS[@]}"; do
  count=$(grep -cx "$id" <<< "$committed" || true)
  assert_eq "$count" "1" "$id committed exactly once"
done

collector_finished_at=$(cat "$TEST_ROOT/collector_finished_at")
first_commit_at=$(stat -c %Y "$TEST_ROOT"/export/batch_*/dataset 2>/dev/null | sort -n | head -1)
if [[ -n "$first_commit_at" && "$first_commit_at" -lt "$collector_finished_at" ]]; then
  pass "first export committed ($((first_commit_at - start_epoch))s in) before remote collector finished ($((collector_finished_at - start_epoch))s in) -> pull+export overlapped"
else
  fail "first export ($first_commit_at) did not precede collector finishing ($collector_finished_at) -- no overlap observed"
fi

collection_done_at=$(stat -c %Y "$TEST_ROOT/export/.pipeline/collection_done" 2>/dev/null || echo 0)
if [[ "$collection_done_at" -gt "$collector_finished_at" ]]; then
  pass "collection_done fired after remote collector finished, not before"
else
  fail "collection_done ($collection_done_at) fired at/before collector finished ($collector_finished_at) -- premature idle detection"
fi

merged="${TEST_ROOT}/export_merged"
if [[ -f "$merged/meta/info.json" ]]; then
  actual_episodes=$(python3 -c "import json;print(json.load(open('$merged/meta/info.json'))['total_episodes'])")
  assert_eq "$actual_episodes" "${#IDS[@]}" "merged dataset episode count"
else
  fail "merged dataset exists at $merged"
fi

summary
