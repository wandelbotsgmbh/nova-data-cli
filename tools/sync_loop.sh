#!/usr/bin/env bash
# Pull recordings from the WS while collection is still running there, and
# export them in the background as they arrive, so pulling + exporting happen
# in parallel instead of waiting for all 1000 episodes then exporting.
#
# A recording is only "done" once <recording_id>/recording.rrd exists (it's
# written once on stop; chunks/ alone means still recording) — that's the
# completeness check, both for what's safe to export.
#
# nova-data-cli has no incremental/append mode (--output must not exist), so
# this can't grow one dataset live: each newly-arrived batch of recordings
# gets exported into its own batch_N output dir. Once collection is finished
# (idle timeout below), the batches are merged into one final LeRobot dataset
# via lerobot's own aggregate_datasets (tools/merge_batches.py).
set -euo pipefail
shopt -s nullglob

REMOTE_HOST="intern@172.31.11.129"
REMOTE_DIRS=(
  "/mnt/data/sebastian/raw_datasets/pick_and_place_sim_20260805_191245"
)
LOCAL_DEST="/home/sebi/ws/Data/raw_data/choreo2"

NOVA_CLI_DIR="/home/sebi/ws/nova-data-cli"
EXPORT_CONFIG="/home/sebi/ws/pick_and_place_imitation_learning/data_collection/configs/lerobot_export.json"
EXPORT_OUTPUT_ROOT="/home/sebi/ws/Data/choreo2_export"

LEDGER="${LOCAL_DEST}/.exported_ids"
INFLIGHT="${LOCAL_DEST}/.inflight_ids"
EXPORT_PID_FILE="${LOCAL_DEST}/.export.pid"

POLL_SECONDS=180
IDLE_MINUTES=10

mkdir -p "$LOCAL_DEST" "$EXPORT_OUTPUT_ROOT"
touch "$LEDGER" "$INFLIGHT"

idle_polls_needed=$(( (IDLE_MINUTES * 60 + POLL_SECONDS - 1) / POLL_SECONDS ))
idle_count=0
last_mtime=""
batch_num=0

remote_max_mtime() {
  ssh "$REMOTE_HOST" "find ${REMOTE_DIRS[*]} -type f -printf '%T@\n' 2>/dev/null | sort -n | tail -1"
}

sync_once() {
  rsync -avP "${REMOTE_DIRS[@]/#/$REMOTE_HOST:}" "$LOCAL_DEST/"
}

export_running() {
  [[ -f "$EXPORT_PID_FILE" ]] && kill -0 "$(cat "$EXPORT_PID_FILE")" 2>/dev/null
}

# Recordings that have a recording.rrd (complete) and aren't already
# exported or queued in a currently-running batch.
find_new_recordings() {
  for remote_dir in "${REMOTE_DIRS[@]}"; do
    local_root="${LOCAL_DEST}/$(basename "$remote_dir")"
    for rrd in "$local_root"/*/recording.rrd; do
      recording_dir="$(dirname "$rrd")"
      recording_id="$(basename "$recording_dir")"
      if ! grep -qxF "$recording_id" "$LEDGER" && ! grep -qxF "$recording_id" "$INFLIGHT"; then
        echo "$recording_dir"
      fi
    done
  done
}

run_export_batch() {
  local batch_dir="$1"
  shift
  local recordings=("$@")

  local merged_dir="${EXPORT_OUTPUT_ROOT}/_merged_${batch_dir}"
  local output_dir="${EXPORT_OUTPUT_ROOT}/${batch_dir}"

  rm -rf "$merged_dir"
  mkdir -p "$merged_dir"
  for recording in "${recordings[@]}"; do
    ln -s "$recording" "${merged_dir}/$(basename "$recording")"
  done

  if (cd "$NOVA_CLI_DIR" && uv run nova-data-cli \
      --dataset "$merged_dir" \
      --config "$EXPORT_CONFIG" \
      --output "$output_dir"); then
    for recording in "${recordings[@]}"; do
      basename "$recording" >> "$LEDGER"
    done
    echo "Exported ${#recordings[@]} recordings -> $output_dir"
  else
    echo "Export batch $batch_dir failed, will retry these next round: ${recordings[*]}" >&2
  fi

  for recording in "${recordings[@]}"; do
    sed -i "\|^$(basename "$recording")\$|d" "$INFLIGHT"
  done
  rm -rf "$merged_dir"
}

maybe_start_export_batch() {
  if export_running; then
    return
  fi

  new_recordings=()
  while IFS= read -r line; do
    [[ -n "$line" ]] && new_recordings+=("$line")
  done < <(find_new_recordings)

  if [[ ${#new_recordings[@]} -eq 0 ]]; then
    return
  fi

  batch_num=$((batch_num + 1))
  batch_name="batch_$(printf '%04d' "$batch_num")"
  for recording in "${new_recordings[@]}"; do
    basename "$recording" >> "$INFLIGHT"
  done

  echo "Starting export batch $batch_name with ${#new_recordings[@]} new recordings"
  run_export_batch "$batch_name" "${new_recordings[@]}" &
  echo $! > "$EXPORT_PID_FILE"
}

while true; do
  sync_once
  maybe_start_export_batch

  mtime="$(remote_max_mtime)"
  if [[ "$mtime" == "$last_mtime" ]]; then
    idle_count=$((idle_count + 1))
  else
    idle_count=0
    last_mtime="$mtime"
  fi

  if [[ $idle_count -ge $idle_polls_needed ]]; then
    echo "No new remote data for ${IDLE_MINUTES}m, assuming collection finished."
    break
  fi

  sleep "$POLL_SECONDS"
done

echo "Final sync + export of any remaining recordings..."
sync_once
while export_running; do
  sleep 5
done
maybe_start_export_batch
while export_running; do
  sleep 5
done

MERGED_OUTPUT="${EXPORT_OUTPUT_ROOT}_merged"
echo "All batches exported. Merging into $MERGED_OUTPUT..."
(cd "$NOVA_CLI_DIR" && uv run python tools/merge_batches.py \
  --batches-root "$EXPORT_OUTPUT_ROOT" \
  --output "$MERGED_OUTPUT")
echo "Done. Merged dataset: $MERGED_OUTPUT"
