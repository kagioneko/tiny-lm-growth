#!/usr/bin/env bash
# 落ちても続く実験パイプライン（tmux で起動する）。各段は出力があれば飛ばすので、何度起動し直してもよい。
#   起動: tmux new -d -s tinylm ./run_pipeline.sh
#   様子: cat results/pipeline_status.txt / tmux attach -t tinylm
set -u
cd "$(dirname "$0")"
mkdir -p results checkpoints
PY=${PY:-python3}
ST=results/pipeline_status.txt
mkdir -p checkpoints
log() { echo "$(date '+%F %T') $*" | tee -a "$ST"; }

step() {  # step 名前 完了の印ファイル コマンド...
  local name=$1 done_file=$2; shift 2
  if [ -e "$done_file" ]; then log "skip  $name（$done_file あり）"; return 0; fi
  log "start $name"
  if "$@"; then log "done  $name"; else log "FAIL  $name rc=$?"; return 1; fi
}

log "=== pipeline 起動 ==="
# 1. 空間系の青空文庫版（space_lm.py は1設定ごとに results/space_aozora.json へ追記し、再開時は済んだ設定を飛ばす）
step "空間系 aozora" results/space_aozora.done bash -c \
  "LM_CORPUS=aozora LM_THREADS=6 LM_VALID_LIMIT=100000 nice -n 19 $PY space_lm.py >> results/space_aozora.log 2>&1 \
   && .venv/bin/python plot_space.py aozora >> results/space_aozora.log 2>&1 && touch results/space_aozora.done"
# 2. Zenn の GPT（iid 3000、前回と同じ設定）を重みつきで → キャッシュ
step "GPT zenn 保存" results/neural_zenn_iid_saved.json bash -c \
  "LM_CORPUS=zenn nice -n 19 $PY neural_lm.py --modes iid --steps 3000 --out results/neural_zenn_iid_saved.json \
   --save checkpoints/zenn_iid.pt > results/neural_zenn_iid_saved.log 2>&1"
step "GPT+キャッシュ zenn" results/gpt_cache_zenn.json bash -c \
  "LM_CORPUS=zenn nice -n 19 $PY gpt_cache.py checkpoints/zenn_iid.pt > results/gpt_cache_zenn.log 2>&1"
# 3. 青空文庫の GPT をさらに長く（24000 ステップ、約8時間）重みつきで → キャッシュ
step "GPT aozora 24k" results/neural_aozora_iid_24k.json bash -c \
  "LM_CORPUS=aozora LM_THREADS=6 nice -n 19 $PY neural_lm.py --modes iid --steps 24000 --d 256 --layers 6 --heads 8 \
   --drop 0.1 --lr 2e-3 --eval-every 500 --valid-limit 100000 --out results/neural_aozora_iid_24k.json \
   --save checkpoints/aozora_iid_24k.pt > results/neural_aozora_iid_24k.log 2>&1"
step "GPT+キャッシュ aozora" results/gpt_cache_aozora.json bash -c \
  "LM_CORPUS=aozora nice -n 19 $PY gpt_cache.py checkpoints/aozora_iid_24k.pt > results/gpt_cache_aozora.log 2>&1"
# 4. さらに長く（48000 ステップ、約13時間）→ キャッシュ。LLM（4.045）越えを狙う
step "GPT aozora 48k" results/neural_aozora_iid_48k.json bash -c \
  "LM_CORPUS=aozora LM_THREADS=5 nice -n 19 $PY neural_lm.py --modes iid --steps 48000 --d 256 --layers 6 --heads 8 \
   --drop 0.1 --lr 2e-3 --eval-every 1000 --valid-limit 100000 --out results/neural_aozora_iid_48k.json \
   --save checkpoints/aozora_iid_48k.pt > results/neural_aozora_iid_48k.log 2>&1"
step "GPT+キャッシュ aozora 48k" results/gpt_cache_aozora_48k.json bash -c \
  "LM_CORPUS=aozora LM_TAG=_48k nice -n 19 $PY gpt_cache.py checkpoints/aozora_iid_48k.pt > results/gpt_cache_aozora_48k.log 2>&1"
log "=== pipeline 終了 ==="
