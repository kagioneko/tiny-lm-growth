#!/bin/bash
# 動的評価 → 全部混ぜ（動的版）。落ちたら再実行: tmux new -d -s tinydyn ./run_dyn.sh
# 各段は出力があれば飛ばす（dyn_eval.py は済んだ lr も飛ばす）
cd "$(dirname "$0")" || exit 1
mkdir -p results checkpoints
PY=${PY:-python3}
ST=results/pipeline_status.txt
log() { echo "$(date '+%F %T') $*" | tee -a "$ST"; }
step() {
  local name=$1 done_file=$2; shift 2
  if [ -e "$done_file" ]; then log "skip  $name（$done_file あり）"; return 0; fi
  log "start $name"
  if "$@"; then log "done  $name"; else log "FAIL  $name rc=$?"; return 1; fi
}
log "=== dyn 起動 ==="
for ck in aozora_iid_48k aozora_iid_24k; do
  step "動的評価 $ck" results/dyn_aozora_${ck}_valid.npy bash -c \
    "LM_CORPUS=aozora nice -n 19 $PY dyn_eval.py checkpoints/$ck.pt >> results/dyn_aozora_$ck.log 2>&1"
done
step "全部混ぜ（動的）" results/mix_all_aozora_dyn.json bash -c \
  "LM_CORPUS=aozora LM_MIX_DYN=1 nice -n 19 $PY mix_all.py checkpoints/aozora_iid_48k.pt checkpoints/aozora_iid_24k.pt > results/mix_all_aozora_dyn.log 2>&1"
log "=== dyn 終了 ==="
