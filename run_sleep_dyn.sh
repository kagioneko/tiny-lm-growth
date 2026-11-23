#!/bin/bash
# 睡眠＋動的評価。落ちたら再実行: tmux new -d -s tinysleep ./run_sleep_dyn.sh
# sleep_dyn.py は済んだ設定を JSON から飛ばす。test の npy があればその段は丸ごと飛ばす
cd "$(dirname "$0")" || exit 1
mkdir -p results checkpoints
PY=${PY:-python3}
ST=results/pipeline_status.txt
log() { echo "$(date '+%F %T') $*" | tee -a "$ST"; }
log "=== sleep_dyn 起動 ==="
for ck in aozora_iid_48k aozora_iid_24k; do
  if [ -e results/sleep_dyn_aozora_${ck}_test.npy ]; then log "skip  sleep_dyn $ck"; continue; fi
  log "start sleep_dyn $ck"
  if LM_CORPUS=aozora nice -n 19 $PY sleep_dyn.py checkpoints/$ck.pt >> results/sleep_dyn_aozora_$ck.log 2>&1; then
    log "done  sleep_dyn $ck"; else log "FAIL  sleep_dyn $ck rc=$?"; fi
done
log "=== sleep_dyn 終了 ==="
# 第2弾（11/24）: 格子を広げた分（済んだ設定は飛ばす）→ 全部混ぜ（睡眠＋動的版）
log "=== sleep_dyn2 起動 ==="
for ck in aozora_iid_48k aozora_iid_24k; do
  log "start sleep_dyn2 $ck"
  if LM_CORPUS=aozora nice -n 19 $PY sleep_dyn.py checkpoints/$ck.pt >> results/sleep_dyn_aozora_$ck.log 2>&1; then
    log "done  sleep_dyn2 $ck"; else log "FAIL  sleep_dyn2 $ck rc=$?"; fi
done
if [ ! -e results/mix_all_aozora_sleep_dyn.json ]; then
  log "start mix sleep_dyn"
  LM_CORPUS=aozora LM_MIX_DYN=1 LM_MIX_PREFIX=sleep_dyn nice -n 19 $PY mix_all.py checkpoints/aozora_iid_48k.pt checkpoints/aozora_iid_24k.pt > results/mix_all_aozora_sleep_dyn.log 2>&1 && log "done  mix sleep_dyn" || log "FAIL  mix sleep_dyn"
fi
log "=== sleep_dyn2 終了 ==="
