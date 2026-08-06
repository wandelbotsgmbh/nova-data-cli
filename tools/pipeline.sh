#!/usr/bin/env bash
# Pulls episode recordings (remote over rsync/ssh, or a local dir) and exports
# them in parallel as they finish, instead of waiting for collection to end.
# See README.md for usage, AGENT.md for how/why this is built the way it is.
set -euo pipefail
shopt -s nullglob

# ---- config -----------------------------------------------------------
# Every value is overridable via a PIPELINE_* env var; see README.md.
MODE="${PIPELINE_MODE:-remote}"
REMOTE_HOST="${PIPELINE_REMOTE_HOST:-intern@172.31.11.129}"
if [[ -n "${PIPELINE_REMOTE_DIRS:-}" ]]; then
  IFS=':' read -r -a REMOTE_DIRS <<< "$PIPELINE_REMOTE_DIRS"
else
  REMOTE_DIRS=(
    "/mnt/data/sebastian/raw_datasets/pick_and_place_sim_20260805_191245"
  )
fi
WATCH_DIR="${PIPELINE_WATCH_DIR:-/home/sebi/ws/Data/raw_data/choreo2/pick_and_place_sim_20260805_191245}"

NOVA_CLI_DIR="${PIPELINE_NOVA_CLI_DIR:-/home/sebi/ws/nova-data-cli}"
EXPORT_CONFIG="${PIPELINE_EXPORT_CONFIG:-/home/sebi/ws/pick_and_place_imitation_learning/data_collection/configs/lerobot_export.json}"
EXPORT_ROOT="${PIPELINE_EXPORT_ROOT:-/home/sebi/ws/Data/choreo2_export}"
read -r -a EXPORT_CLI_CMD <<< "${PIPELINE_EXPORT_CLI:-uv run nova-data-cli}"  # swap in a stub for tests

CHUNK="${PIPELINE_CHUNK:-8}"
POLL_SECONDS="${PIPELINE_POLL_SECONDS:-180}"
IDLE_MINUTES="${PIPELINE_IDLE_MINUTES:-10}"

# Worker count/memory cap are computed at startup from this machine's actual
# resources, not hardcoded — tune these two, not compute_workers() below.
MEM_TARGET_FRACTION="${PIPELINE_MEM_TARGET_FRACTION:-0.55}"
WORKER_MEM_ESTIMATE_MB="${PIPELINE_WORKER_MEM_ESTIMATE_MB:-3200}"

STATE="${EXPORT_ROOT}/.pipeline"
LOCK="${STATE}/lock"
CLAIMED="${STATE}/claimed"
FAILED="${STATE}/failed"
QUARANTINE="${STATE}/quarantine"
SCRATCH="${STATE}/scratch"
TMP_OUT="${EXPORT_ROOT}/.tmp"
LOGS="${STATE}/logs"
COLLECTION_DONE="${STATE}/collection_done"
PGID_FILE="${STATE}/supervisor.pgid"

# ---- arg parsing --------------------------------------------------------
ROLE="${1:-supervisor}"
shift || true
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    *) POSITIONAL+=("$1"); shift ;;
  esac
done
set -- "${POSITIONAL[@]}"

if [[ "$MODE" == "remote" ]]; then
  [[ ${#REMOTE_DIRS[@]} -eq 0 ]] && { echo "remote mode needs REMOTE_DIRS" >&2; exit 1; }
  WATCH_DIR="$(dirname "$WATCH_DIR")/$(basename "${REMOTE_DIRS[0]}")"
elif [[ "$MODE" == "local" ]]; then
  [[ -d "$WATCH_DIR" ]] || { echo "local mode needs WATCH_DIR to already exist: $WATCH_DIR" >&2; exit 1; }
else
  echo "Unknown --mode: $MODE (expected remote|local)" >&2; exit 1
fi

mkdir -p "$STATE" "$CLAIMED" "$FAILED" "$QUARANTINE" "$SCRATCH" "$TMP_OUT" "$LOGS"

# ---- resource sizing (supervisor computes once, workers inherit via env) --
compute_workers() {
  local mem_total_kb mem_workers core_workers
  mem_total_kb="$(awk '/MemTotal/{print $2}' /proc/meminfo)"
  mem_workers="$(awk -v kb="$mem_total_kb" -v f="$MEM_TARGET_FRACTION" -v est="$WORKER_MEM_ESTIMATE_MB" \
    'BEGIN{printf "%d", (kb/1024*f)/est}')"  # awk not $(( )): bash treats leading-zero numbers as octal
  core_workers=$(( $(nproc) * 8 / 10 ))  # leave headroom, don't claim every core
  local workers=$mem_workers
  [[ $core_workers -lt $workers ]] && workers=$core_workers
  [[ $workers -lt 1 ]] && workers=1
  echo "$workers"
}

mem_available_floor_mb() {
  awk -v est="$WORKER_MEM_ESTIMATE_MB" 'BEGIN{printf "%d", est * 1.2}'
}

# ---- shared helpers ------------------------------------------------------
log() { echo "[$(date +%H:%M:%S)] [$ROLE] $*"; }

mem_available_mb() {
  awk '/MemAvailable/{printf "%d", $2/1024}' /proc/meminfo
}

wait_for_memory() {
  local floor; floor="$(mem_available_floor_mb)"
  while [[ "$(mem_available_mb)" -lt "$floor" ]]; do
    log "MemAvailable below ${floor}MB, waiting for headroom before claiming more work"
    sleep 15
  done
}

# "done" is derived from .claimed_ids in every committed batch, not a marker
# file — see AGENT.md for why.
declare -A DONE_IDS
rebuild_done() {
  DONE_IDS=()
  local batch f id
  for batch in "$EXPORT_ROOT"/batch_*; do
    f="$batch/.claimed_ids"
    [[ -f "$f" ]] || continue
    while IFS= read -r id; do
      [[ -n "$id" ]] && DONE_IDS["$id"]=1
    done < "$f"
  done
}

is_candidate() {
  local id="$1" rrd="$WATCH_DIR/$id/recording.rrd"
  [[ -f "$rrd" ]] || return 1
  [[ -n "${DONE_IDS[$id]:-}" ]] && return 1
  [[ -d "$CLAIMED/$id" ]] && return 1
  [[ -f "$QUARANTINE/$id" ]] && return 1
  if [[ "$MODE" == "local" ]]; then
    [[ -n "$(find "$rrd" -mmin +1 2>/dev/null)" ]] || return 1  # must be untouched 60s (no rsync atomicity here)
  fi
  return 0
}

list_candidates() {
  local id
  for d in "$WATCH_DIR"/*/; do
    id="$(basename "$d")"
    is_candidate "$id" && echo "$id"
  done
}

failure_count() { local f="$FAILED/$1"; [[ -f "$f" ]] && wc -c < "$f" || echo 0; }
record_attempt() { printf x >> "$FAILED/$1"; }

# ---- acquisition ---------------------------------------------------------
role_acquire() {
  mkdir -p "$WATCH_DIR"
  if [[ "$MODE" == "local" ]]; then
    log "local mode: watching $WATCH_DIR, no network step"
    local idle_polls_needed=$(( (IDLE_MINUTES * 60 + POLL_SECONDS - 1) / POLL_SECONDS )) idle_count=0
    while true; do
      if [[ -z "$(find "$WATCH_DIR" -type f -mmin "-${IDLE_MINUTES}" -print -quit 2>/dev/null)" ]]; then
        idle_count=$((idle_count + 1))
        [[ $idle_count -ge $idle_polls_needed ]] && break
      else
        idle_count=0
      fi
      sleep "$POLL_SECONDS"
    done
    touch "$COLLECTION_DONE"
    log "local collection idle for ${IDLE_MINUTES}m, done"
    return
  fi

  local ssh_opts=(-o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=4)
  local idle_polls_needed=$(( (IDLE_MINUTES * 60 + POLL_SECONDS - 1) / POLL_SECONDS )) idle_count=0 last_mtime=""

  sync_once() {
    local rc=0
    rsync -a --partial --info=stats2 --timeout=300 \
      "${REMOTE_DIRS[@]/#/$REMOTE_HOST:}" "$(dirname "$WATCH_DIR")/" || rc=$?
    case "$rc" in
      0|23|24) ;;  # 23/24: source still being written, expected
      *) log "rsync exited $rc (non-fatal, will retry next poll)" ;;
    esac
  }

  remote_max_mtime() {
    timeout 60 ssh "${ssh_opts[@]}" "$REMOTE_HOST" \
      "find ${REMOTE_DIRS[*]} -type f -printf '%T@\n' 2>/dev/null | sort -n | tail -1"
  }

  while true; do
    sync_once
    local mtime rc=0
    mtime="$(remote_max_mtime)" || rc=$?
    if [[ $rc -ne 0 || -z "$mtime" ]]; then
      log "idle-check ssh/find failed or timed out — inconclusive, not counted toward idle"
    elif [[ "$mtime" == "$last_mtime" ]]; then
      idle_count=$((idle_count + 1))
    else
      idle_count=0
      last_mtime="$mtime"
    fi

    if [[ $idle_count -ge $idle_polls_needed ]]; then
      log "no new remote data for ${IDLE_MINUTES}m, final sync"
      sync_once
      break
    fi
    sleep "$POLL_SECONDS"
  done
  touch "$COLLECTION_DONE"
}

# ---- worker ---------------------------------------------------------------
# Claims up to CHUNK candidates, exports them in one nova-data-cli invocation,
# validates, then atomically commits. See AGENT.md for the full protocol.
export_and_commit() {
  local -a ids=("$@")
  local batch_name="batch_$(date +%Y%m%d_%H%M%S)_w${WORKER_IDX}_$$_${RANDOM}"
  local scratch="$SCRATCH/$batch_name" out="$TMP_OUT/$batch_name"

  rm -rf "$scratch"
  mkdir -p "$scratch" "$out"
  printf '%s\n' "${ids[@]}" > "$out/.claimed_ids"
  local id
  for id in "${ids[@]}"; do
    ln -s "$WATCH_DIR/$id" "$scratch/$id"
  done

  local ok=1
  # nice/ionice: yield cores/disk to other processes instead of a CPU watchdog.
  # PYTHONUNBUFFERED: so progress lines stream live instead of buffering.
  local export_cmd=(env PYTHONUNBUFFERED=1 nice -n 10 ionice -c2 -n7 "${EXPORT_CLI_CMD[@]}" --dataset "$scratch" --config "$EXPORT_CONFIG" --output "$out/dataset")
  if command -v systemd-run >/dev/null 2>&1; then
    (cd "$NOVA_CLI_DIR" && systemd-run --user --scope -p "MemoryMax=${WORKER_MEM_ESTIMATE_MB}M" -p MemorySwapMax=0 --collect -- "${export_cmd[@]}") || ok=0
  else
    log "systemd-run not available — running without a memory cgroup cap (best effort only)"
    (cd "$NOVA_CLI_DIR" && "${export_cmd[@]}") || ok=0
  fi

  if [[ $ok -eq 1 ]]; then
    if uv run python "$NOVA_CLI_DIR/tools/validate_batch.py" --output-dir "$out/dataset" --claimed-ids "$out/.claimed_ids"; then
      mv -T "$out" "$EXPORT_ROOT/$batch_name"
      rm -rf "$scratch"
      log "committed $batch_name (${#ids[@]} recording(s))"
      return 0
    fi
  fi

  rm -rf "$out" "$scratch"
  return 1
}

# Failure counter/quarantine only apply at batch size 1 (see AGENT.md for why).
attempt_batch() {
  local -a ids=("$@")

  if [[ ${#ids[@]} -eq 1 ]]; then
    local id="${ids[0]}"
    record_attempt "$id"  # before running, so a hard kill/OOM still counts
    if export_and_commit "$id"; then
      rmdir "$CLAIMED/$id" 2>/dev/null || true
    else
      if [[ "$(failure_count "$id")" -ge 3 ]]; then
        log "quarantining $id after 3 failed attempts"
        touch "$QUARANTINE/$id"
        rm -f "$FAILED/$id"
      fi
      rmdir "$CLAIMED/$id" 2>/dev/null || true
    fi
    return
  fi

  if export_and_commit "${ids[@]}"; then
    local id
    for id in "${ids[@]}"; do rmdir "$CLAIMED/$id" 2>/dev/null || true; done
    return
  fi

  local half=$(( ${#ids[@]} / 2 ))
  local -a left=("${ids[@]:0:half}") right=("${ids[@]:half}")
  log "batch of ${#ids[@]} failed/ambiguous, bisecting into ${#left[@]} + ${#right[@]}"
  attempt_batch "${left[@]}"
  attempt_batch "${right[@]}"
}

role_worker() {
  WORKER_IDX="$1"
  local empty_scans=0
  while true; do
    rebuild_done
    wait_for_memory

    local -a claimed=()
    local id
    while IFS= read -r id; do
      [[ ${#claimed[@]} -ge $CHUNK ]] && break
      mkdir "$CLAIMED/$id" 2>/dev/null && claimed+=("$id")
    done < <(list_candidates)

    # Recheck against a fresh rebuild_done: another worker may have committed
    # (and released) one of these IDs in the gap since our scan (see AGENT.md).
    if [[ ${#claimed[@]} -gt 0 ]]; then
      rebuild_done
      local -a fresh=()
      for id in "${claimed[@]}"; do
        if [[ -n "${DONE_IDS[$id]:-}" ]]; then
          rmdir "$CLAIMED/$id" 2>/dev/null || true
        else
          fresh+=("$id")
        fi
      done
      claimed=("${fresh[@]}")
    fi

    if [[ ${#claimed[@]} -eq 0 ]]; then
      # A single empty scan can be a transient glitch (e.g. under heavy
      # system load), not proof there's no more work — require 3 consecutive
      # empty scans before treating "collection done" as "exit for good",
      # the same way acquire's idle-detection needs sustained evidence rather
      # than a single reading. A false-empty scan just wastes 30s here; a
      # false-permanent worker exit silently cuts throughput for the rest of
      # the run.
      empty_scans=$((empty_scans + 1))
      if [[ -f "$COLLECTION_DONE" && $empty_scans -ge 3 ]]; then
        log "no candidates across 3 consecutive scans and collection done, exiting"
        return 0
      fi
      sleep 30
      continue
    fi

    empty_scans=0
    attempt_batch "${claimed[@]}"
  done
}

# ---- supervisor -----------------------------------------------------------
role_supervisor() {
  # setsid makes this the leader of a fresh process group so `kill -- -$$` on
  # shutdown reaches every descendant (see AGENT.md).
  if [[ -z "${PIPELINE_RESPAWNED:-}" ]]; then
    exec env PIPELINE_RESPAWNED=1 setsid "$0" supervisor --mode "$MODE"
  fi

  exec 9>"$LOCK"
  if ! flock -n 9; then
    echo "Another pipeline.sh is already running (lock held: $LOCK)" >&2
    exit 1
  fi

  # Holding the lock means it's safe to sweep stale state, unless a killed
  # run's process group is somehow still alive.
  if [[ -f "$PGID_FILE" ]]; then
    old_pgid="$(cat "$PGID_FILE")"
    if [[ -n "$old_pgid" ]] && kill -0 -- "-$old_pgid" 2>/dev/null; then
      echo "Previous run's process group ($old_pgid) still alive — kill it before restarting" >&2
      exit 1
    fi
  fi
  rm -rf "${CLAIMED:?}"/* "${TMP_OUT:?}"/* "${SCRATCH:?}"/*
  rm -f "$COLLECTION_DONE"
  mkdir -p "$CLAIMED"

  echo "$$" > "$PGID_FILE"  # setsid above made pid == pgid

  WORKERS="$(compute_workers)"
  log "sizing: $(nproc) cores, $(awk '/MemTotal/{printf "%.1fGB", $2/1024/1024}' /proc/meminfo) RAM, MEM_TARGET_FRACTION=$MEM_TARGET_FRACTION, WORKER_MEM_ESTIMATE_MB=$WORKER_MEM_ESTIMATE_MB -> WORKERS=$WORKERS"

  trap 'log "shutting down"; kill -- -$$ 2>/dev/null || true' INT TERM

  # Tee each role's output to both terminal (prefixed) and its log file.
  ( "$0" acquire --mode "$MODE" 2>&1 | sed -u 's/^/[acquire] /' | tee -a "$LOGS/acquire.log" ) &
  local acquire_pid=$!

  local -a worker_pids=()
  spawn_workers() {
    worker_pids=()
    local i
    for ((i = 0; i < WORKERS; i++)); do
      ( "$0" worker --mode "$MODE" "$i" 2>&1 | sed -u "s/^/[w${i}] /" | tee -a "$LOGS/w${i}.log" ) &
      worker_pids+=($!)
    done
  }
  spawn_workers
  wait "$acquire_pid" || true
  wait "${worker_pids[@]}" || true

  # Drain: relaunch until 2 consecutive passes find nothing left (see
  # AGENT.md) — a single clean scan could be a transient glitch, and trusting
  # it alone here risks merging before everything's actually exported, not
  # just losing a worker the way role_worker's equivalent check would.
  local drained_scans=0
  while [[ $drained_scans -lt 2 ]]; do
    rebuild_done
    if [[ -z "$(list_candidates | head -1)" ]] && [[ -z "$(ls -A "$CLAIMED" 2>/dev/null)" ]]; then
      drained_scans=$((drained_scans + 1))
      sleep 5
      continue
    fi
    drained_scans=0
    log "drain pass found leftover work, relaunching workers"
    spawn_workers
    wait "${worker_pids[@]}" || true
  done

  local -a batch_dirs=("$EXPORT_ROOT"/batch_*)
  if [[ ${#batch_dirs[@]} -eq 0 ]]; then
    log "no batches produced, nothing to merge"
    print_report
    [[ -n "$(ls -A "$QUARANTINE" 2>/dev/null)" ]] && exit 1
    return 0
  fi

  log "all workers drained, merging"
  (cd "$NOVA_CLI_DIR" && uv run python tools/merge_batches.py \
    --batches-root "$EXPORT_ROOT" \
    --output "${EXPORT_ROOT}_merged")
  log "done: ${EXPORT_ROOT}_merged"
  print_report
  [[ -n "$(ls -A "$QUARANTINE" 2>/dev/null)" ]] && exit 1
  return 0
}

# recordings found vs. exported vs. quarantined, and episodes in the merged
# dataset (not the same number — one recording can yield several episodes).
print_report() {
  local total=0
  local d
  for d in "$WATCH_DIR"/*/; do
    [[ -f "${d}recording.rrd" ]] && total=$((total + 1))
  done
  rebuild_done
  local exported=${#DONE_IDS[@]}
  local -a quarantined_ids=("$QUARANTINE"/*)
  local quarantined_n=${#quarantined_ids[@]}

  log "=== report ==="
  log "recordings found:     $total"
  log "exported:              $exported"
  log "quarantined (3x fail): $quarantined_n"
  if [[ $quarantined_n -gt 0 ]]; then
    log "quarantined IDs: $(basename -a "${quarantined_ids[@]}" | tr '\n' ' ')"
  fi
  if [[ -f "${EXPORT_ROOT}_merged/meta/info.json" ]]; then
    local episodes
    episodes="$(python3 -c "import json;print(json.load(open('${EXPORT_ROOT}_merged/meta/info.json'))['total_episodes'])" 2>/dev/null || echo "?")"
    log "episodes in merged dataset: $episodes"
  fi
}

case "$ROLE" in
  supervisor) role_supervisor ;;
  acquire) role_acquire ;;
  worker) role_worker "${1:?worker index required}" ;;
  sizing) compute_workers ;;  # test hook: print the computed WORKERS count and exit
  *) echo "Unknown role: $ROLE" >&2; exit 1 ;;
esac
