"""小さい GPT ＋ 混ぜ合わせ型キャッシュ — 記憶系の発想を GPT の出力に足して、はしごを1段持ち上げられるか。

  P(c) = (1−λ)·P_GPT(c | 直前 ctx 字) + λ·痕跡_c / Σ痕跡
  痕跡: 字が出ると +1、毎字 半減期 H で減衰（memory_lm.py の忘却LM・混ぜ合わせ型と同じ）
GPT は文脈 ctx=128 字しか見えないが、キャッシュは H 次第でもっと昔（数千字前の登場人物名など）まで覚えている。
学ぶつまみは (λ, H) だけ。valid（先頭 N 字）で格子探索し、選んだ設定の test を報告。λ=0 は GPT 単体と完全一致。
GPT の評価位置は neural_lm.bpc() と同じ（ctx 字の窓を半分ずつずらし、先頭窓は全部・以降は後半だけ）。

使い方: neural_lm.py --save checkpoints/xxx.pt で保存した重みを渡す。
  LM_CORPUS=aozora python gpt_cache.py checkpoints/aozora_iid.pt
"""
import json
import math
import os
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F

from neural_lm import TinyGPT
from sleep_growth import encode, load

torch.set_num_threads(int(os.environ.get("LM_THREADS", "6")))
LOG2 = math.log(2)
HS = [10, 30, 100, 300, 1000, 3000, 10000]
LAMBDAS = [0.0, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2]


@torch.no_grad()
def gpt_logp_per_position(model, ids, ctx):
    """各位置 t（ids[t] を当てる）の GPT 対数確率。数えない位置は nan。"""
    model.eval()
    half = ctx // 2
    out = np.full(len(ids), np.nan)
    starts = list(range(0, len(ids) - ctx - 1, half))
    for i in range(0, len(starts), 64):
        ss = starts[i:i + 64]
        x = torch.from_numpy(np.stack([ids[s:s + ctx] for s in ss]))
        y = torch.from_numpy(np.stack([ids[s + 1:s + ctx + 1] for s in ss]))
        lp = F.log_softmax(model(x), -1).gather(-1, y[..., None])[..., 0].numpy()
        for j, s in enumerate(ss):
            k = 0 if s == 0 else half
            out[s + 1 + k:s + 1 + ctx] = lp[j, k:]
    return out


def cache_probs(ids, Hs):
    """各位置 t で「それまでの痕跡」から見た ids[t] のキャッシュ確率 (nH, T)。痕跡が空なら nan。
    痕跡は字ごとに「最後に更新した時刻と値」だけ持つ遅延評価（毎字 V 個を減衰させない）。"""
    V = int(ids.max()) + 1
    T = len(ids)
    out = np.full((len(Hs), T), np.nan)
    for h, H in enumerate(Hs):
        dec = 0.5 ** (1.0 / H)
        last = np.zeros(V, dtype=np.int64)
        val = np.zeros(V)
        S = 0.0
        for t in range(T):
            c = ids[t]
            if S > 0:
                # 分子は時刻 t まで減衰させるので、分母（時刻 t−1 の合計）も1歩減衰させて揃える（全字で和が1）
                out[h, t] = val[c] * dec ** (t - last[c]) / (S * dec)
            # 時刻 t の字を足す（次の位置から見える）
            val[c] = val[c] * dec ** (t - last[c]) + 1.0
            last[c] = t
            S = S * dec + 1.0
    return out


def mixture_bpc(lg, pc, lambdas):
    """lg: (T,) GPT 対数確率（nan は数えない）、pc: (nH, T) キャッシュ確率 → (nH, nλ) の bits/char。"""
    m = ~np.isnan(lg)
    pg = np.exp(lg[m])
    pcm = pc[:, m]
    pcm = np.where(np.isnan(pcm), pg[None, :], pcm)   # 痕跡が空の位置は GPT のみ
    lam = np.asarray(lambdas)[None, :, None]
    p = (1 - lam) * pg[None, None, :] + lam * pcm[:, None, :]
    return -np.log(p).mean(-1) / LOG2, int(m.sum())


def main():
    pathlib.Path("results").mkdir(exist_ok=True)
    ck_path = sys.argv[1]
    corpus = os.environ.get("LM_CORPUS", "zenn")
    valid_chars = int(os.environ.get("LM_VALID_LIMIT", "300000"))
    ck = torch.load(ck_path, weights_only=False)
    itos = ck["itos"]
    stoi = {ch: i for i, ch in enumerate(itos)}
    arch = dict(ck["arch"])
    ctx = arch.pop("ctx")
    model = TinyGPT(len(itos), ctx=ctx, **arch)
    model.load_state_dict(ck["state"])
    texts = load()
    va = encode(texts["valid"], stoi)[:valid_chars]
    te = encode(texts["test"], stoi)

    lg_va = gpt_logp_per_position(model, va, ctx)
    r_va, n_va = mixture_bpc(lg_va, cache_probs(va, HS), LAMBDAS)
    hi, li = np.unravel_index(np.argmin(r_va), r_va.shape)
    H, lam = HS[hi], LAMBDAS[li]
    print(f"[{corpus}] valid（先頭 {len(va):,} 字）: GPT 単体 {r_va[0, 0]:.3f} → 最良 {r_va[hi, li]:.3f}"
          f"（λ={lam}, H={H}）", flush=True)
    lg_te = gpt_logp_per_position(model, te, ctx)
    r_te, n_te = mixture_bpc(lg_te, cache_probs(te, [H]), [0.0, lam])
    print(f"  test: GPT 単体 {r_te[0, 0]:.3f} → GPT＋キャッシュ {r_te[0, 1]:.3f}（差 {r_te[0, 0] - r_te[0, 1]:+.3f}）",
          flush=True)
    out = {"corpus": corpus, "checkpoint": pathlib.Path(ck_path).name, "gpt_best": ck["best"],
           "valid_chars": len(va), "valid_counted": n_va, "test_counted": n_te,
           "grid": {"Hs": HS, "lambdas": LAMBDAS, "valid": r_va.tolist()},
           "chosen": {"lambda": lam, "H": H},
           "valid": {"gpt": float(r_va[0, 0]), "gpt_cache": float(r_va[hi, li])},
           "test": {"gpt": float(r_te[0, 0]), "gpt_cache": float(r_te[0, 1])}}
    pathlib.Path(f"results/gpt_cache_{corpus}{os.environ.get('LM_TAG', '')}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
