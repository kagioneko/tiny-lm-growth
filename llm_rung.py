"""はしごの最上段: 本物の（小さい）LLM を同じ物差し（test の bits/char）で測る。

モデル: Qwen/Qwen2.5-0.5B（Apache-2.0、HuggingFace から取得）。CPU で推論。
測り方: test 本文をトークン化し、1024トークンの窓を512ずつずらして各窓の後半だけ数える（先頭窓は全部）。
       bits/char = 総負対数尤度（nat）/ ln2 / 文字数。
注意:
- 青空文庫・公開済み Zenn 記事は LLM の事前学習に含まれている可能性が高い（読んだことがある文章＝有利）。
- 自作モデルは珍しい字を <unk> にまとめているが、LLM は全ての字を正確に当てる必要がある（自作側が少し有利）。
"""
import json
import math
import os
import pathlib
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_num_threads(int(os.environ.get("LM_THREADS", "2")))
MODEL = os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-0.5B")
ROOT = pathlib.Path(__file__).resolve().parent


@torch.no_grad()
def bits_per_char(model, tok, text, win=1024, stride=512):
    ids = tok(text, return_tensors="pt").input_ids[0]
    nll, counted = 0.0, 0
    for start in range(0, len(ids), stride):
        chunk = ids[start:start + win]
        if len(chunk) < 2:
            break
        logits = model(chunk[None]).logits[0, :-1].float()
        lp = torch.log_softmax(logits, -1).gather(-1, chunk[1:, None])[:, 0]
        k = 0 if start == 0 else win - stride - 1  # 前の窓と重なる部分は数えない
        nll -= lp[k:].sum().item()
        counted += len(lp) - k
        if start + win >= len(ids):
            break
    return nll / math.log(2) / len(text), len(ids), counted


def main():
    pathlib.Path("results").mkdir(exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
    model.eval()
    out = {"model": MODEL}
    paths = {"zenn": ROOT / "data" / "test.txt", "aozora": ROOT / "data" / "aozora" / "test.txt"}
    wanted = os.environ.get("LLM_CORPORA", "zenn,aozora").split(",")      # 例: LLM_CORPORA=aozora
    prev = ROOT / "results" / f"llm_{MODEL.split('/')[-1]}.json"
    if prev.exists():
        out.update({k: v for k, v in json.loads(prev.read_text()).items() if k in paths})
    for name, path in [(n, paths[n]) for n in wanted]:
        text = path.read_text(encoding="utf-8")
        limit = int(os.environ.get("LLM_CHAR_LIMIT", "0"))
        if limit:
            text = text[:limit]
        t0 = time.time()
        b, ntok, counted = bits_per_char(model, tok, text)
        out[name] = {"bpc": b, "chars": len(text), "tokens": ntok, "counted_tokens": counted,
                     "sec": round(time.time() - t0, 1)}
        print(f"{name}: {b:.3f} bits/char（{len(text):,}字 / {ntok:,}トークン / {out[name]['sec']}s）", flush=True)
    tag = MODEL.split("/")[-1]
    (ROOT / "results" / f"llm_{tag}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
