#!/usr/bin/env bash
# Shared helpers for tools/tests/scenario_*.sh. Sourced, not executed.
set -euo pipefail
shopt -s nullglob

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PIPELINE="$REPO_ROOT/tools/pipeline.sh"
FAKE_CLI="uv run python $REPO_ROOT/tools/tests/fake_nova_data_cli.py"
REMOTE_TEST_HOST="intern@172.31.11.129"
REMOTE_TEST_ROOT="/mnt/data/sebastian/pipeline_test_fixtures"

PASS=0
FAIL=0

# Guard against ever pointing a destructive rm/ssh-rm at a real data dir —
# every test path must contain this marker. Call before any rm -rf/ssh cleanup.
require_test_path() {
  case "$1" in
    *pipeline_test*) ;;
    *) echo "REFUSING to touch non-test-looking path: $1" >&2; exit 1 ;;
  esac
}

pass() { PASS=$((PASS + 1)); echo "  PASS: $1"; }
fail() { FAIL=$((FAIL + 1)); echo "  FAIL: $1"; }
assert_eq() { [[ "$1" == "$2" ]] && pass "$3 ($1)" || fail "$3 (expected [$2], got [$1])"; }

summary() {
  echo
  echo "=== $PASS passed, $FAIL failed ==="
  [[ $FAIL -eq 0 ]]
}

# make_fixture <local|remote> <base_dir> <id> <behavior>
#   behavior: success (default) | skip | crash | sleep:N
make_fixture() {
  local kind="$1" base="$2" id="$3" behavior="${4:-success}"
  if [[ "$kind" == "local" ]]; then
    mkdir -p "$base/$id"
    touch "$base/$id/recording.rrd"
    echo "$behavior" > "$base/$id/.behavior"
  else
    ssh -o BatchMode=yes "$REMOTE_TEST_HOST" \
      "mkdir -p '$base/$id' && touch '$base/$id/recording.rrd' && echo '$behavior' > '$base/$id/.behavior'"
  fi
}

# All IDs actually committed into a batch anywhere under $1 (an EXPORT_ROOT).
committed_ids() {
  local export_root="$1" f
  for f in "$export_root"/batch_*/.claimed_ids; do
    [[ -f "$f" ]] && cat "$f"
  done
}

quarantined_ids() {
  local export_root="$1"
  ls "$export_root/.pipeline/quarantine" 2>/dev/null || true
}
