"""数え上げ n-gram のはしごを、教材の量を変えて測る（LM_CORPUS で教材を選ぶ）。

各量について ユニグラム・2〜5-gram（補間つき絶対値割引 D=0.75）の test bits/char。
語彙は「全量の学習用」から1回だけ作り（2回以上出た文字 + <unk>）、全ての量で共通にする
（量ごとに作り直すと、量が増えるほど珍しい字まで当てる問題になり、比較が不公平になるため）。valid は使わない。
"""
import json
import math
import os
import pathlib
import sys
from collections import Counter, defaultdict

import numpy as np

from sleep_growth import build_vocab, encode, load

LOG2 = math.log(2)


def ladder(train_ids, test_ids, V, max_order=4, D=0.75):
    uni = np.bincount(train_ids, minlength=V) + 1.0
    uni = uni / uni.sum()
    res = {"unigram": float(-np.log(uni[test_ids[1:]]).mean() / LOG2)}
    tr = train_ids.tolist()
    te = test_ids.tolist()
    tables = []
    for k in range(1, max_order + 1):
        c = defaultdict(Counter)
        for t in range(k, len(tr)):
            c[tuple(tr[t - k:t])][tr[t]] += 1
        tables.append({ctx: (cnt, sum(cnt.values()), len(cnt)) for ctx, cnt in c.items()})
        del c
        s = 0.0
        for t in range(1, len(te)):
            x = te[t]
            p = float(uni[x])
            for j in range(1, min(t, k) + 1):
                e = tables[j - 1].get(tuple(te[t - j:t]))
                if e is None:
                    break
                cnt, tot, types = e
                p = (max(cnt.get(x, 0) - D, 0) + D * types * p) / tot
            s -= math.log(p)
        res[f"{k + 1}gram"] = s / (len(te) - 1) / LOG2
        print(f"  {k + 1}-gram: {res[f'{k + 1}gram']:.3f}", flush=True)
    return res


def main():
    pathlib.Path("results").mkdir(exist_ok=True)
    sizes = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [70000, 300000, 1000000, 0]
    out = {}
    os.environ["LM_TRAIN_LIMIT"] = "0"
    itos, stoi = build_vocab(load()["train"])
    for n in sizes:
        os.environ["LM_TRAIN_LIMIT"] = str(n)
        texts = load()
        tr, te = encode(texts["train"], stoi), encode(texts["test"], stoi)
        print(f"教材 {len(tr):,} 字 / 語彙 {len(itos)}", flush=True)
        out[len(tr)] = {"vocab": len(itos), **ladder(tr, te, len(itos))}
    name = os.environ.get("LM_CORPUS", "zenn")
    pathlib.Path(f"results/ladder_{name}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
