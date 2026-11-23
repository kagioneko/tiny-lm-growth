"""制約系（化学反応LM・砂山LM）を教材で育てる — 「どこまで育つか」第7弾（DESIGN_next.md の D、最後の家族）。

原典: 化学＝原子の在庫で言える語が尽きる／砂山＝砂が溜まって崩れると語が出る。確率版（土台は memory_lm.py と同じ数え上げ n-gram）:
  D1 化学（字ごとの在庫）: 在庫 s_c∈[0,1]。字を出すと s_c ← s_c·(1−δ)、毎字 半減期 H で満タンへ回復。P ∝ P0·s^α
  D2 化学（共有の原子）  : 在庫を字ではなく「文字種」（ひらがな/カタカナ/漢字/英数字/その他）で共有。
                           漢字を続けて出すと漢字の原子が減る。P ∝ P0·s_{種(c)}^α
  D3 砂山               : 毎字、土台が予想した分だけ各字に砂が積もる（h += P0）。出た字の山は崩れて 0 になり、
                           崩れた砂は共起の近い字へ流れる（雪崩）。山は半減期 H で風化。
                           P ∝ P0·exp(κ·V·h̃)、h̃ = h·(1−減衰)（＝最近の予想の平均）。κ>0 なら「予想されたのにまだ出ない字が出やすい」
  D3+A                  : 砂山に第4弾の混ぜ合わせ型キャッシュを足す（C3 と同じ組み方）
α=0 / κ=0 は土台と完全一致（対照）。つまみは valid で格子探索、test は選んだ設定だけ。
予想（DESIGN）: D1 は免疫LMの形（出た字を出にくくする）なので土台より良くならない。D2 は種の交替を捉えれば少し効くかも。
"""
import argparse
import json
import math
import os
import pathlib

import numpy as np

from family_c import build_networks, pmi_matrix
from memory_lm import NgramBase, char_class
from sleep_growth import build_vocab, encode, load

LOG2 = math.log(2)
DELTA = 0.5


def evaluate(base, ids, cls, W, lamA, HA, Hs, alphas, kappas):
    V = base.V
    n_cls = int(cls.max()) + 1
    dec = np.array([0.5 ** (1.0 / h) for h in Hs])[:, None]
    stock = np.ones((len(Hs), V))
    atoms = np.ones((len(Hs), n_cls))
    sand = np.zeros((len(Hs), V))
    decA = 0.5 ** (1.0 / HA)
    tr = np.zeros(V)
    trS = 0.0
    al = np.asarray(alphas)[None, :, None]
    ka = np.asarray(kappas)[None, :, None]
    nll = {"base": 0.0, "A": 0.0, **{k: np.zeros((len(Hs), len(v))) for k, v in
                                       (("D1", alphas), ("D2", alphas), ("D3", kappas), ("D3A", kappas))}}
    ids = ids.tolist()
    order = base.order
    lp = np.log
    for t in range(1, len(ids)):
        c = ids[t]
        p = base.dist(ids[max(0, t - order):t])
        pc = p[c]
        nll["base"] -= math.log(pc)
        pcache = tr[c] / trS if trS > 0 else pc
        nll["A"] -= math.log((1 - lamA) * pc + lamA * pcache)
        # D1: 字ごとの在庫
        w = p[None, None, :] * stock[:, None, :] ** al                    # (nH, nα, V)
        nll["D1"] -= lp(w[:, :, c] / w.sum(-1))
        # D2: 文字種で共有の原子
        sa = atoms[:, cls]                                               # (nH, V)
        w = p[None, None, :] * sa[:, None, :] ** al
        nll["D2"] -= lp(w[:, :, c] / w.sum(-1))
        # D3: 砂山
        ht = sand * (1 - dec) * V                                        # (nH, V)
        w = p[None, None, :] * np.exp(ka * ht[:, None, :])
        psand = w[:, :, c] / w.sum(-1)
        nll["D3"] -= lp(psand)
        nll["D3A"] -= lp((1 - lamA) * psand + lamA * pcache)
        # 状態更新
        stock[:, c] *= (1 - DELTA)
        stock = 1 - (1 - stock) * dec                                    # 満タンへ回復
        atoms[:, cls[c]] *= (1 - DELTA)
        atoms = 1 - (1 - atoms) * dec
        sand = sand * dec + p[None, :]                                   # 予想した分だけ積もる
        fallen = sand[:, c:c + 1].copy()
        sand[:, c] = 0.0                                                 # 出た字の山が崩れる
        sand += fallen * W[c][None, :]                                   # 雪崩: 共起の近い字へ
        tr *= decA
        tr[c] += 1.0
        trS = trS * decA + 1.0
    n = len(ids) - 1
    return {k: v / n / LOG2 for k, v in nll.items()}


def main():
    pathlib.Path("results").mkdir(exist_ok=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", type=int, default=2)
    ap.add_argument("--valid-chars", type=int, default=0)
    ap.add_argument("--test-chars", type=int, default=0)
    a = ap.parse_args()
    corpus = os.environ.get("LM_CORPUS", "zenn")
    texts = load()
    itos, stoi = build_vocab(texts["train"])
    V = len(itos)
    tr, va, te = (encode(texts[k], stoi) for k in ("train", "valid", "test"))
    if a.valid_chars:
        va = va[:a.valid_chars]
    if a.test_chars:
        te = te[:a.test_chars]
    cls = np.array([char_class(ch) for ch in itos])
    mem = json.loads(pathlib.Path(f"results/memory_{corpus}.json").read_text())
    lamA, HA = mem["chosen"]["mixture_ref"]["lambda"], mem["chosen"]["mixture_ref"]["H"]
    base = NgramBase(tr, V, a.order)
    W, _ = build_networks(pmi_matrix(tr, V, 50))
    Hs = [3, 10, 30, 100]
    alphas = [0.0, 0.03, 0.1, 0.3, 1.0]
    kappas = [-0.3, -0.1, -0.03, 0.0, 0.03, 0.1, 0.3]
    r = evaluate(base, va, cls, W, lamA, HA, Hs, alphas, kappas)
    pick = {}
    for k, knob, vals in (("D1", "alpha", alphas), ("D2", "alpha", alphas), ("D3", "kappa", kappas),
                          ("D3A", "kappa", kappas)):
        i = np.unravel_index(np.argmin(r[k]), r[k].shape)
        pick[k] = {"H": Hs[i[0]], knob: vals[i[1]], "valid": float(r[k][i])}
    print(f"[{corpus}] valid（{len(va):,}字）: 土台 {r['base']:.3f} / A {r['A']:.3f}（λ={lamA}, H={HA}）")
    for k in pick:
        print(f"  {k} 最良 {pick[k]['valid']:.3f}  {pick[k]}", flush=True)
    rt = {}
    for k in pick:
        knob = "alpha" if k in ("D1", "D2") else "kappa"
        kw = {"alphas": [pick[k][knob]] if knob == "alpha" else [0.0],
              "kappas": [pick[k][knob]] if knob == "kappa" else [0.0]}
        rr = evaluate(base, te, cls, W, lamA, HA, [pick[k]["H"]], **kw)
        rt[k] = float(rr[k][0, 0])
        rt["base"], rt["A"] = rr["base"], rr["A"]
    print(f"  test: 土台 {rt['base']:.3f} / A {rt['A']:.3f} / D1 化学(字) {rt['D1']:.3f} / D2 化学(原子) {rt['D2']:.3f} / "
          f"D3 砂山 {rt['D3']:.3f} / D3+A {rt['D3A']:.3f}", flush=True)
    out = {"corpus": corpus, "order": a.order, "valid_chars": len(va), "test_chars": len(te),
           "A": {"lambda": lamA, "H": HA}, "delta": DELTA,
           "grid": {"Hs": Hs, "alphas": alphas, "kappas": kappas, **{k: r[k].tolist() for k in pick}},
           "valid": {"base": r["base"], "A": r["A"], **{k: pick[k]["valid"] for k in pick}},
           "chosen": pick, "test": rt}
    pathlib.Path(f"results/family_d_{corpus}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
