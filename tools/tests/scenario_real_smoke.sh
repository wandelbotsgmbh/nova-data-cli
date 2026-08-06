#!/usr/bin/env bash
# Tier 2: ONE real end-to-end run through the actual nova-data-cli (no stub),
# using two small pre-existing sample recordings, copied (never symlinked or
# moved) from /home/sebi/ws/Data/5-pick-cube-sim-raw into an isolated test
# dir -- catches anything the fake_nova_data_cli.py stub can't: real
# export_summary.json shape, real aggregate_datasets behavior on a real
# dataset, real timing. Local mode only (remote transport is already covered
# for real in scenario_remote_*.sh with the stub; this test's job is the real
# CLI, not re-proving rsync).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./lib.sh

SRC_ROOT="/home/sebi/ws/Data/5-pick-cube-sim-raw"
REAL_CONFIG="/home/sebi/ws/pick_and_place_imitation_learning/data_collection/configs/lerobot_export.json"

TEST_ROOT="/tmp/pipeline_test_real_smoke_$$"
require_test_path "$TEST_ROOT"
trap 'kill -- "-${pid:-}" 2>/dev/null || true; rm -rf "$TEST_ROOT"' EXIT
WATCH="$TEST_ROOT/watch"
EXPORT_ROOT="$TEST_ROOT/export"
mkdir -p "$WATCH" "$EXPORT_ROOT"

echo "=== scenario_real_smoke (real nova-data-cli, no stub) ==="

if [[ ! -d "$SRC_ROOT" || ! -f "$REAL_CONFIG" ]]; then
  echo "SKIP: sample recordings ($SRC_ROOT) or real config ($REAL_CONFIG) not found on this machine"
  exit 0
fi

# Two smallest available recordings, to keep this test's runtime reasonable.
mapfile -t SAMPLE_IDS < <(
  for d in "$SRC_ROOT"/*/; do
    [[ -f "$d/recording.rrd" ]] || continue
    echo "$(stat -c %s "$d/recording.rrd") $(basename "$d")"
  done | sort -n | head -2 | awk '{print $2}'
)
[[ ${#SAMPLE_IDS[@]} -eq 2 ]] || { echo "SKIP: fewer than 2 sample recordings with recording.rrd found"; exit 0; }

for id in "${SAMPLE_IDS[@]}"; do
  cp -r "$SRC_ROOT/$id" "$WATCH/$id"  # copy, never touch the source
done

export PIPELINE_MODE=local
export PIPELINE_WATCH_DIR="$WATCH"
export PIPELINE_EXPORT_ROOT="$EXPORT_ROOT"
export PIPELINE_EXPORT_CONFIG="$REAL_CONFIG"
# PIPELINE_EXPORT_CLI left unset -> real `uv run nova-data-cli`
export PIPELINE_CHUNK=2
export PIPELINE_POLL_SECONDS=5
export PIPELINE_IDLE_MINUTES=1

echo "exporting ${SAMPLE_IDS[*]} through the real CLI (this actually decodes/encodes, expect ~1-3 min)..."
"$PIPELINE" supervisor >"$TEST_ROOT/supervisor.log" 2>&1 &
pid=$!
wait "$pid" || true
echo "pipeline finished"

committed=$(committed_ids "$EXPORT_ROOT" | sort -u)
for id in "${SAMPLE_IDS[@]}"; do
  count=$(grep -cx "$id" <<< "$committed" || true)
  assert_eq "$count" "1" "$id committed exactly once (real CLI)"
done

batch_dir=$(find "$EXPORT_ROOT" -maxdepth 1 -name 'batch_*' | head -1)
if [[ -n "$batch_dir" && -f "$batch_dir/dataset/export_summary.json" ]]; then
  pass "real export_summary.json written: $(cat "$batch_dir/dataset/export_summary.json")"
else
  fail "no real export_summary.json found in committed batch"
fi

merged="${EXPORT_ROOT}_merged"
if [[ -f "$merged/meta/info.json" ]]; then
  actual_episodes=$(python3 -c "import json;print(json.load(open('$merged/meta/info.json'))['total_episodes'])")
  if [[ "$actual_episodes" -ge 1 ]]; then
    pass "real merged dataset produced ($actual_episodes episode(s))"
  else
    fail "real merged dataset has 0 episodes"
  fi
else
  fail "merged dataset exists at $merged"
fi

summary
