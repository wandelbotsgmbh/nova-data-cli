#!/usr/bin/env bash
# Kills the whole supervisor process group mid-run (SIGKILL, no graceful
# shutdown trap -- simulates a real crash/OOM, not a clean stop) while some
# batches are still in-flight (claimed but not yet committed), then restarts
# pipeline.sh against the SAME EXPORT_ROOT/WATCH_DIR and verifies: the
# restart isn't blocked by the dead process's stale pgid/lock, the stale
# claims get swept and requeued (not stuck forever), nothing that was
# already committed before the kill gets re-exported, and everything ends
# up committed exactly once with a correct final merge.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./lib.sh

TEST_ROOT="/tmp/pipeline_test_crash_restart_$$"
require_test_path "$TEST_ROOT"
trap 'kill -- "-${pid:-}" 2>/dev/null || true; rm -rf "$TEST_ROOT"' EXIT
WATCH="$TEST_ROOT/watch"
EXPORT_ROOT="$TEST_ROOT/export"
mkdir -p "$WATCH" "$EXPORT_ROOT"

echo "=== scenario_crash_restart ==="

IDS=(crec01 crec02 crec03 crec04 crec05 crec06)
for id in "${IDS[@]}"; do make_fixture local "$WATCH" "$id" sleep:8; done

export PIPELINE_MODE=local
export PIPELINE_WATCH_DIR="$WATCH"
export PIPELINE_EXPORT_ROOT="$EXPORT_ROOT"
export PIPELINE_EXPORT_CLI="$FAKE_CLI"
export PIPELINE_EXPORT_CONFIG="$TEST_ROOT/fake_config.json"
touch "$PIPELINE_EXPORT_CONFIG"
export PIPELINE_CHUNK=1
export PIPELINE_POLL_SECONDS=3
export PIPELINE_IDLE_MINUTES=1

echo "--- first run: will SIGKILL mid-flight ---"
"$PIPELINE" supervisor >"$TEST_ROOT/run1.log" 2>&1 &
pid=$!
# Local mode's 60s candidate-maturity floor means nothing is claimable before
# t=60s; give it a few seconds into the export wave, so some batches are
# genuinely in-flight (claimed, mid sleep:8, not yet committed) when killed.
sleep 66
old_pgid="$pid"
kill -9 -- "-$pid" 2>/dev/null || true
wait "$pid" 2>/dev/null || true
sleep 1  # let the kernel finish reaping before we look

committed_before_kill=$(committed_ids "$EXPORT_ROOT" | sort -u)
claimed_before_restart=$(ls "$EXPORT_ROOT/.pipeline/claimed" 2>/dev/null || true)
echo "after kill: committed=[$committed_before_kill] still-claimed=[$claimed_before_restart]"
if [[ -n "$claimed_before_restart" ]]; then
  pass "at least one recording was genuinely in-flight (claimed, uncommitted) at kill time"
else
  fail "nothing was in-flight at kill time -- test didn't exercise the crash window (timing needs tuning)"
fi
# sanity: nothing should be BOTH committed and still claimed
overlap=$(comm -12 <(sort <<< "$committed_before_kill") <(sort <<< "$claimed_before_restart"))
assert_eq "$overlap" "" "no recording is both committed and still claimed after the kill"

echo "--- second run: restart against the same EXPORT_ROOT ---"
"$PIPELINE" supervisor >"$TEST_ROOT/run2.log" 2>&1 &
pid=$!
wait "$pid" || true
echo "restart finished"

if grep -q "still alive" "$TEST_ROOT/run2.log"; then
  fail "restart refused to start, claiming the old (dead) process group was still alive"
else
  pass "restart was not blocked by the dead process's stale pgid"
fi

committed_after=$(committed_ids "$EXPORT_ROOT" | sort -u)
for id in "${IDS[@]}"; do
  count=$(grep -cx "$id" <<< "$committed_after" || true)
  assert_eq "$count" "1" "$id committed exactly once across both runs (no loss, no duplicate)"
done
dupes=$(committed_ids "$EXPORT_ROOT" | sort | uniq -d)
assert_eq "$dupes" "" "no ID committed into more than one batch across both runs"

merged="${EXPORT_ROOT}_merged"
if [[ -f "$merged/meta/info.json" ]]; then
  actual_episodes=$(python3 -c "import json;print(json.load(open('$merged/meta/info.json'))['total_episodes'])")
  assert_eq "$actual_episodes" "${#IDS[@]}" "merged dataset episode count after crash+restart"
else
  fail "merged dataset exists at $merged"
fi

summary
