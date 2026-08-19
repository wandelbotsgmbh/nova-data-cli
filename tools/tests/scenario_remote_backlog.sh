#!/usr/bin/env bash
# Remote mode (real SSH/rsync against the real workstation host), entire
# backlog present upfront on the remote side. Uses a dedicated, brand-new
# subdir under /mnt/data/sebastian/pipeline_test_fixtures/ on the real remote
# host -- NEVER the real raw_datasets/ collection dir.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./lib.sh

RUN_ID="run_backlog_$$"
REMOTE_DIR="$REMOTE_TEST_ROOT/$RUN_ID"
require_test_path "$REMOTE_DIR"  # contains "pipeline_test" -- extra guard before any remote rm -rf

TEST_ROOT="/tmp/pipeline_test_remote_backlog_$$"
require_test_path "$TEST_ROOT"
cleanup() {
  kill -- "-${pid:-}" 2>/dev/null || true
  ssh -o BatchMode=yes "$REMOTE_TEST_HOST" "rm -rf '$REMOTE_DIR'" 2>/dev/null || true
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT
mkdir -p "$TEST_ROOT/watch" "$TEST_ROOT/export"

echo "=== scenario_remote_backlog ==="

IDS=(rrec01 rrec02 rrec03 rrec04 rrec05)
for id in "${IDS[@]}"; do make_fixture remote "$REMOTE_DIR" "$id" success; done
make_fixture remote "$REMOTE_DIR" rbad skip  # should end up quarantined

export PIPELINE_MODE=remote
export PIPELINE_REMOTE_HOST="$REMOTE_TEST_HOST"
export PIPELINE_REMOTE_DIRS="$REMOTE_DIR"
export PIPELINE_WATCH_DIR="$TEST_ROOT/watch/placeholder"  # only its dirname is used; basename comes from REMOTE_DIRS
export PIPELINE_EXPORT_ROOT="$TEST_ROOT/export"
export PIPELINE_EXPORT_CLI="$FAKE_CLI"
export PIPELINE_EXPORT_CONFIG="$TEST_ROOT/fake_config.json"
touch "$PIPELINE_EXPORT_CONFIG"
export PIPELINE_CHUNK=3
export PIPELINE_POLL_SECONDS=5
export PIPELINE_IDLE_MINUTES=1

"$PIPELINE" supervisor >"$TEST_ROOT/supervisor.log" 2>&1 &
pid=$!
wait "$pid" || true
echo "pipeline finished"

WATCH_ACTUAL="$TEST_ROOT/watch/$RUN_ID"
[[ -d "$WATCH_ACTUAL" ]] && pass "rsync pulled recordings into local watch dir" \
  || fail "rsync did not create expected local watch dir $WATCH_ACTUAL"

committed=$(committed_ids "$TEST_ROOT/export" | sort -u)
quarantined=$(quarantined_ids "$TEST_ROOT/export" | sort -u)
for id in "${IDS[@]}"; do
  count=$(grep -cx "$id" <<< "$committed" || true)
  assert_eq "$count" "1" "$id committed exactly once"
done
is_quarantined=$(grep -cx "rbad" <<< "$quarantined" || true)
assert_eq "$is_quarantined" "1" "rbad quarantined"

merged="${TEST_ROOT}/export_merged"
if [[ -f "$merged/meta/info.json" ]]; then
  actual_episodes=$(python3 -c "import json;print(json.load(open('$merged/meta/info.json'))['total_episodes'])")
  assert_eq "$actual_episodes" "${#IDS[@]}" "merged dataset episode count"
else
  fail "merged dataset exists at $merged"
fi

summary
