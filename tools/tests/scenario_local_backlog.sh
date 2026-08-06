#!/usr/bin/env bash
# Local mode, entire backlog present upfront (collection already finished) —
# the "just export, but still use parallelism" case. Also covers: bisection
# isolating a bad recording without quarantining its batch-mates, quarantine
# after 3 permanent failures, and a flaky recording recovering via retry
# instead of being quarantined. Genuine wall-clock concurrency proof is a
# separate test (scenario_concurrency.sh) — batch composition here is
# claim-order-dependent (bisection can group/split slow fixtures
# unpredictably), so a tight timing assertion here would be flaky by
# construction; this scenario instead asserts >1 distinct worker actually
# committed real work, which is deterministic regardless of batch shuffling.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./lib.sh

TEST_ROOT="/tmp/pipeline_test_local_backlog_$$"
require_test_path "$TEST_ROOT"
# pid's process group == pid (pipeline.sh's supervisor re-execs itself under
# setsid before doing anything else), so this reaches the whole acquire/
# worker/nova-data-cli tree even if this scenario is itself killed/timed out.
trap 'kill -- "-${pid:-}" 2>/dev/null || true; rm -rf "$TEST_ROOT" "$FAKE_CLI_STATE_DIR"' EXIT
export FAKE_CLI_STATE_DIR="$TEST_ROOT/fake_cli_state"
WATCH="$TEST_ROOT/watch"
EXPORT_ROOT="$TEST_ROOT/export"
mkdir -p "$WATCH" "$EXPORT_ROOT"

echo "=== scenario_local_backlog ==="

FAST_IDS=(rec01 rec02 rec03 rec04 rec05 rec06 rec07 rec08)
for id in "${FAST_IDS[@]}"; do make_fixture local "$WATCH" "$id" success; done
make_fixture local "$WATCH" bad_always skip        # should end up quarantined
make_fixture local "$WATCH" bad_flaky flaky:2      # should recover on 3rd attempt, not quarantined
SLOW_IDS=(slow01 slow02 slow03)
for id in "${SLOW_IDS[@]}"; do make_fixture local "$WATCH" "$id" sleep:5; done

ALL_IDS=("${FAST_IDS[@]}" bad_always bad_flaky "${SLOW_IDS[@]}")

export PIPELINE_MODE=local
export PIPELINE_WATCH_DIR="$WATCH"
export PIPELINE_EXPORT_ROOT="$EXPORT_ROOT"
export PIPELINE_EXPORT_CLI="$FAKE_CLI"
export PIPELINE_EXPORT_CONFIG="$TEST_ROOT/fake_config.json"
touch "$PIPELINE_EXPORT_CONFIG"
export PIPELINE_CHUNK=3
export PIPELINE_POLL_SECONDS=3
export PIPELINE_IDLE_MINUTES=1

"$PIPELINE" supervisor >"$TEST_ROOT/supervisor.log" 2>&1 &
pid=$!
wait "$pid" || true
echo "pipeline finished"

# --- assertions ---
committed=$(committed_ids "$EXPORT_ROOT" | sort -u)
quarantined=$(quarantined_ids "$EXPORT_ROOT" | sort -u)

# 1. every fixture ends up committed XOR quarantined, nothing dropped, nothing duplicated
for id in "${ALL_IDS[@]}"; do
  count_committed=$(grep -cx "$id" <<< "$committed" || true)
  is_quarantined=$(grep -cx "$id" <<< "$quarantined" || true)
  if [[ "$id" == "bad_always" ]]; then
    assert_eq "$count_committed" "0" "bad_always never committed"
    assert_eq "$is_quarantined" "1" "bad_always quarantined"
  else
    assert_eq "$count_committed" "1" "$id committed exactly once"
    assert_eq "$is_quarantined" "0" "$id NOT quarantined"
  fi
done

# 2. no ID appears in more than one batch's .claimed_ids (no duplicate export)
dupes=$(committed_ids "$EXPORT_ROOT" | sort | uniq -d)
assert_eq "$dupes" "" "no ID committed into more than one batch"

# 3. merge ran exactly once and produced a dataset with the right episode count
expected_episodes=$(( ${#FAST_IDS[@]} + 1 + ${#SLOW_IDS[@]} ))  # fast + bad_flaky (recovers) + slow; bad_always contributes 0
merged="${EXPORT_ROOT}_merged"
if [[ -f "$merged/meta/info.json" ]]; then
  actual_episodes=$(python3 -c "import json;print(json.load(open('$merged/meta/info.json'))['total_episodes'])")
  assert_eq "$actual_episodes" "$expected_episodes" "merged dataset episode count"
else
  fail "merged dataset exists at $merged"
fi

# 4. more than one worker index actually committed a batch (real parallel use,
#    not just WORKERS>1 sitting idle while one worker does everything)
distinct_workers=$(grep -oh 'committed batch_[0-9_]*_w[0-9]*' "$EXPORT_ROOT"/.pipeline/logs/w*.log 2>/dev/null \
  | grep -o '_w[0-9]*$' | sort -u | wc -l)
if [[ $distinct_workers -ge 2 ]]; then
  pass "$distinct_workers distinct workers committed batches (real parallel use)"
else
  fail "only $distinct_workers distinct worker(s) committed batches"
fi

summary
