#!/usr/bin/env bash
# Runs every tools/tests/scenario_*.sh in turn and reports pass/fail per
# scenario. Each scenario is independently runnable too: bash tools/tests/scenario_X.sh
set -uo pipefail  # not -e: one scenario's failure shouldn't stop the rest
cd "$(dirname "${BASH_SOURCE[0]}")"

SCENARIOS=(
  scenario_sizing.sh
  scenario_concurrency.sh
  scenario_local_backlog.sh
  scenario_local_live.sh
  scenario_remote_backlog.sh
  scenario_remote_live.sh
  scenario_crash_restart.sh
  scenario_real_smoke.sh
)

overall_pass=0
overall_fail=0
declare -A results

for s in "${SCENARIOS[@]}"; do
  echo
  echo "############################################################"
  echo "# $s"
  echo "############################################################"
  if bash "./$s"; then
    results["$s"]="PASS"
    overall_pass=$((overall_pass + 1))
  else
    results["$s"]="FAIL"
    overall_fail=$((overall_fail + 1))
  fi
done

echo
echo "============================================================"
echo "SUMMARY"
echo "============================================================"
for s in "${SCENARIOS[@]}"; do
  printf '%-30s %s\n' "$s" "${results[$s]}"
done
echo "------------------------------------------------------------"
echo "$overall_pass scenario(s) passed, $overall_fail failed"

[[ $overall_fail -eq 0 ]]
