#!/bin/bash
# 青空文庫だけで、主要な結果（数え上げのはしご → GPT 24000/48000回 → キャッシュ → 全部混ぜ → 動的評価 → 睡眠＋動的評価 → LLM）を
# 依存の順に一から作る。Zenn 教材（作者の記事集合）は不要。
#   起動: tmux new -d -s repro ./reproduce_aozora.sh   （合計で CPU 数十時間。GPT 48000回だけで約14時間）
#   出力先: RESULTS（既定 results_repro/）。同梱の results/ は上書きしない。各段は出力があれば飛ばすので、落ちても再実行で続きから。
#   比較: 終わったら results_repro/ と同梱の results/ の数値を見比べる（乱数・CPU 差で小数第3位前後はずれうる）。
set -u
cd "$(dirname "$0")" || exit 1
PY=${PY:-python3}
OUT=${RESULTS:-results_repro}
mkdir -p "$OUT" checkpoints_repro
ST=$OUT/status.txt
log() { echo "$(date '+%F %T') $*" | tee -a "$ST"; }
step() {  # step 名前 完了の印ファイル コマンド...
  local name=$1 done_file=$2; shift 2
  if [ -e "$done_file" ]; then log "skip  $name"; return 0; fi
  log "start $name"
  if "$@"; then log "done  $name"; else log "FAIL  $name rc=$?"; exit 1; fi
}
# スクリプトは results/ に書くので、作業中は results を $OUT に差し替える（同梱の results/ は退避して最後に戻す）
[ -e results_shipped ] && { echo "results_shipped が既にあります（前回の中断？）。中身を確認して results に戻してから再実行してください" >&2; exit 1; }
if [ -e results ] && [ ! -L results ]; then mv results results_shipped; fi
ln -sfn "$OUT" results
trap 'rm -f results; [ -e results_shipped ] && mv results_shipped results' EXIT
export LM_CORPUS=aozora
CK=checkpoints_repro

step "青空文庫の取得"        data/aozora/raw/.fetched bash -c "$PY fetch_aozora.py && touch data/aozora/raw/.fetched"
step "作品を train/valid/test に" data/aozora/test.txt $PY corpus_aozora.py
step "数え上げのはしご（test）"   results/ladder_aozora.json $PY ladder_scale.py 0
step "数え上げのはしご（valid）"  results/ladder_aozora_valid.json $PY ladder_valid_aozora.py
step "記憶系（混ぜ合わせ型キャッシュの λ,H）" results/memory_aozora.json $PY memory_lm.py --order 3 --valid-chars 50000
for spec in 24000:500 48000:1000; do       # 学習回数:途中評価の間隔（実験時と同じ）
  n=${spec%%:*}; ev=${spec##*:}; k=$((n / 1000))k
  step "GPT ${k}" results/neural_aozora_iid_${k}.json bash -c \
    "$PY neural_lm.py --modes iid --steps $n --d 256 --layers 6 --heads 8 --drop 0.1 --lr 2e-3 --eval-every $ev \
     --valid-limit 100000 --out results/neural_aozora_iid_${k}.json --save $CK/aozora_iid_${k}.pt > results/neural_aozora_iid_${k}.log 2>&1"
done
step "GPT 48k＋キャッシュ"   results/gpt_cache_aozora_48k.json bash -c "LM_TAG=_48k $PY gpt_cache.py $CK/aozora_iid_48k.pt > results/gpt_cache_aozora_48k.log 2>&1"
step "全部混ぜ"              results/mix_all_aozora.json bash -c "$PY mix_all.py $CK/aozora_iid_48k.pt $CK/aozora_iid_24k.pt > results/mix_all_aozora.log 2>&1"
for k in 48k 24k; do
  step "動的評価 $k"         results/dyn_aozora_aozora_iid_${k}_valid.npy bash -c "$PY dyn_eval.py $CK/aozora_iid_${k}.pt >> results/dyn_aozora_aozora_iid_${k}.log 2>&1"
done
step "全部混ぜ（動的）"      results/mix_all_aozora_dyn.json bash -c "LM_MIX_DYN=1 $PY mix_all.py $CK/aozora_iid_48k.pt $CK/aozora_iid_24k.pt > results/mix_all_aozora_dyn.log 2>&1"
for k in 48k 24k; do
  step "睡眠＋動的評価 $k"   results/sleep_dyn_aozora_aozora_iid_${k}_valid.npy bash -c "$PY sleep_dyn.py $CK/aozora_iid_${k}.pt >> results/sleep_dyn_aozora_aozora_iid_${k}.log 2>&1"
done
step "全部混ぜ（睡眠＋動的）" results/mix_all_aozora_sleep_dyn.json bash -c "LM_MIX_DYN=1 LM_MIX_PREFIX=sleep_dyn $PY mix_all.py $CK/aozora_iid_48k.pt $CK/aozora_iid_24k.pt > results/mix_all_aozora_sleep_dyn.log 2>&1"
step "LLM（Qwen2.5-0.5B）"   results/llm_Qwen2.5-0.5B.json bash -c "LLM_CORPORA=aozora $PY llm_rung.py > results/llm_rung.log 2>&1"
log "=== 再現 終了（結果は $OUT/）==="
