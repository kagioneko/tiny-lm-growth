"""全部混ぜ — 自作の部品を線形に混ぜて LLM（青空 4.045）に届くか。

  P(c) = Σ_k w_k·P_k(c)、部品 k = GPT 48000回 / GPT 24000回 / 数え上げ n-gram / キャッシュ（短い・長い半減期）
LM_MIX_DYN=1 なら GPT は dyn_eval.py の動的評価版（results/mix_all_{corpus}_dyn.json）。
重み w は valid（先頭 LM_VALID_LIMIT 字）で EM（線形混合の対数尤度を最大化、単調に上がる）。test は決めた重みで1回だけ。
数える位置は gpt_cache.py と同じ（GPT の評価位置）。キャッシュが空の位置は GPT 48k で代用。
  LM_CORPUS=aozora python mix_all.py checkpoints/aozora_iid_48k.pt checkpoints/aozora_iid_24k.pt
"""
import json
import math
import os
import pathlib
import sys

import numpy as np
import torch

from gpt_cache import cache_probs, gpt_logp_per_position
from memory_lm import NgramBase
from neural_lm import TinyGPT
from sleep_growth import encode, load

LOG2 = math.log(2)
CACHE_HS = [300, 3000]
DYN = os.environ.get("LM_MIX_DYN") == "1"
DYN_PREFIX = os.environ.get("LM_MIX_PREFIX", "dyn")       # sleep_dyn にすると睡眠＋動的評価版を混ぜる
CORPUS = os.environ.get("LM_CORPUS", "aozora")
NAMES = [pathlib.Path(p).stem for p in sys.argv[1:]]


def load_gpt(path):
    ck = torch.load(path, weights_only=False)
    arch = dict(ck["arch"])
    ctx = arch.pop("ctx")
    m = TinyGPT(len(ck["itos"]), ctx=ctx, **arch)
    m.load_state_dict(ck["state"])
    return m, ctx, ck["itos"]


def ngram_probs(base, ids):
    out = np.empty(len(ids))
    out[0] = base.uni[ids[0]]
    L = ids.tolist()
    for t in range(1, len(L)):
        out[t] = base.dist(L[max(0, t - base.order):t])[L[t]]
    return out


def components(models, base, ids, split):
    if DYN:   # dyn_eval.py が保存した「読みながら学ぶ」GPT の対数確率を使う
        lgs = [np.load(f"results/{DYN_PREFIX}_{CORPUS}_{n}_{split}.npy") for n in NAMES]
    else:
        lgs = [gpt_logp_per_position(m, ids, ctx) for m, ctx in models]
    mask = ~np.isnan(lgs[0])
    cols = [np.exp(lg[mask]) for lg in lgs]
    cols.append(ngram_probs(base, ids)[mask])
    pc = cache_probs(ids, CACHE_HS)[:, mask]
    for row in pc:
        cols.append(np.where(np.isnan(row), cols[0], row))
    return np.stack(cols, 1)                    # (N, K)


def em(P, iters=200):
    w = np.full(P.shape[1], 1.0 / P.shape[1])
    for _ in range(iters):
        r = P * w
        r /= r.sum(1, keepdims=True)
        w = r.mean(0)
    return w


def bpc(P, w):
    return float(-np.log(P @ w).mean() / LOG2)


def main():
    pathlib.Path("results").mkdir(exist_ok=True)
    corpus = os.environ.get("LM_CORPUS", "aozora")
    vlim = int(os.environ.get("LM_VALID_LIMIT", "300000"))
    torch.set_num_threads(int(os.environ.get("LM_THREADS", "6")))
    models, itos = [], None
    for p in sys.argv[1:]:
        m, ctx, itos_p = load_gpt(p)
        assert itos is None or itos == itos_p
        itos = itos_p
        models.append((m, ctx))
    stoi = {ch: i for i, ch in enumerate(itos)}
    texts = load()
    tr = encode(texts["train"], stoi)
    va = encode(texts["valid"], stoi)[:vlim]
    te = encode(texts["test"], stoi)
    order = json.loads(pathlib.Path(f"results/memory_{corpus}.json").read_text())["order"]
    base = NgramBase(tr, len(itos), order)
    # order は文脈長（直前 order 字）なので n-gram の n = order+1
    names = [((DYN_PREFIX + " ") if DYN else "") + n for n in NAMES] + [f"{order + 1}-gram"] + [f"cache H={h}" for h in CACHE_HS]
    Pv = components(models, base, va, "valid")
    assert Pv.shape[1] == len(names), (Pv.shape, names)
    w = em(Pv)
    Pt = components(models, base, te, "test")
    one = {}
    for k, n in enumerate(names):
        v = float(-np.log(Pt[:, k]).mean() / LOG2)
        one[n] = v if math.isfinite(v) else None    # キャッシュ単体は確率0の字があり無限大 → null
    res = {"corpus": corpus, "names": names, "weights": dict(zip(names, w.tolist())),
           "valid": {"mix": bpc(Pv, w)}, "test": {"mix": bpc(Pt, w), "single": one}, "test_counted": len(Pt)}
    # 部品を1つずつ抜いた test（重みは valid で付け直し）
    res["test"]["drop_one"] = {}
    for k, n in enumerate(names):
        keep = [j for j in range(len(names)) if j != k]
        res["test"]["drop_one"][n] = bpc(Pt[:, keep], em(Pv[:, keep]))
    print(json.dumps(res, ensure_ascii=False, indent=1), flush=True)
    pathlib.Path(f"results/mix_all_{corpus}{('_' + DYN_PREFIX) if DYN else ''}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
