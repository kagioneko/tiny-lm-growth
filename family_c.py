"""集団・流行系（伝染LM・生態系LM）を教材で育てる — 「どこまで育つか」第6弾（DESIGN_next.md の C）。

A（記憶系）は「出た字そのもの」が出やすくなるキャッシュ。C は「**関連する字にも広がる**」キャッシュ。
ネットワーク（誰が誰に感染する／誰と誰が共生・競争する）は教材の共起から作る:
  共起 = 学習用本文で ±K 字以内に一緒に出た回数 → PMI = log(N·C_ij / (c_i·c_j))（自分自身は除く＝それは A の担当）
  伝染: 感染ネットワーク W = 正の PMI（PPMI）を行ごとに正規化。字 c が出ると W[c] の字が感染（半減期 H で回復）
  生態系: 相互作用 A = 符号つき PMI（+ 共生、− 競争）。いまの個体群 x（出た字、半減期 H）から各字への圧力 e = A·x/Σx

土台 P0 は memory_lm.py と同じ数え上げ n-gram。比べるもの（全部 valid で格子探索、test は選んだ設定だけ）:
  土台
  A  : 混ぜ合わせ型キャッシュ（第4弾の最良形）P = (1−λ)P0 + λ·痕跡分布
  C1 伝染（原典の形）: P ∝ P0·(1 + β·感染度)（感染度は平均1に正規化）           ← DESIGN の「土台 × (1 + 話題化ボーナス)」
  C2 伝染の上乗せ   : P = (1−λ−μ)P0 + λ·痕跡分布 + μ·感染分布  （λ,H は A の最良に固定、μ と感染の H を探す）
  C3 生態系         : P = (1−λ)·[P0·exp(γ·e) を正規化] + λ·痕跡分布  （γ と個体群の H を探す）
C2・C3 の「A からの改善」が、広がり（伝染）・共生と競争（生態系）の手柄。μ=0 / γ=0 は A と完全一致。
"""
import argparse
import json
import math
import os
import pathlib

import numpy as np

from memory_lm import NgramBase
from sleep_growth import build_vocab, encode, load

LOG2 = math.log(2)


def pmi_matrix(ids, V, K):
    """±K 字以内の共起から PMI（対角は 0）。共起0の組は −∞ の代わりに最小値で埋める。"""
    C = np.zeros(V * V, dtype=np.float64)
    for k in range(1, K + 1):
        C += np.bincount(ids[:-k] * V + ids[k:], minlength=V * V)
    C = C.reshape(V, V)
    C = C + C.T
    row = C.sum(1)
    N = row.sum()
    with np.errstate(divide="ignore"):
        pmi = np.log(C * N / np.maximum(np.outer(row, row), 1e-12))
    pmi[~np.isfinite(pmi)] = np.nan
    lo = np.nanmin(pmi)
    pmi = np.nan_to_num(pmi, nan=lo)
    np.fill_diagonal(pmi, 0.0)
    return pmi


def build_networks(pmi, clip=4.0):
    W = np.maximum(pmi, 0.0)
    W = W / np.maximum(W.sum(1, keepdims=True), 1e-12)        # 感染: 行和1（広がる先の分布）
    A = np.clip(pmi, -clip, clip) / clip                       # 生態系: [−1, 1]
    return W, A


def evaluate(base, ids, W, A, lamA, HA, H2s, betas, mus, gammas):
    """1回の走査で全格子を計算。返り値は bits/char。"""
    V = base.V
    decA = 0.5 ** (1.0 / HA)
    dec2 = np.array([0.5 ** (1.0 / h) for h in H2s])[:, None]
    tr = np.zeros(V)                       # A の痕跡（出た字そのもの）
    trS = 0.0
    sp = np.zeros((len(H2s), V))           # 感染度（W[c] の和、半減期 H2）
    spS = np.zeros((len(H2s), 1))
    popA = np.zeros((len(H2s), V))         # 個体群 x（出た字、半減期 H2）に A を掛けたもの x·A（線形なので逐次更新）
    popS = np.zeros((len(H2s), 1))
    betas = np.asarray(betas)[None, :]
    mus = np.asarray(mus)[None, :]
    gam = np.asarray(gammas)[None, :, None]
    nll = {"base": 0.0, "A": 0.0, "C1": np.zeros((len(H2s), betas.shape[1])),
           "C2": np.zeros((len(H2s), mus.shape[1])), "C3": np.zeros((len(H2s), gam.shape[1]))}
    ids = ids.tolist()
    order = base.order
    for t in range(1, len(ids)):
        c = ids[t]
        p = base.dist(ids[max(0, t - order):t])
        pc = p[c]
        nll["base"] -= math.log(pc)
        pcache = tr[c] / trS if trS > 0 else pc
        pA = (1 - lamA) * pc + lamA * pcache
        nll["A"] -= math.log(pA)
        if t > 1:
            s = sp / spS                                            # (nH, V) 感染分布
            # C1: P ∝ P0·(1+β·V·s)（V·s = 平均が1になるようにした感染度）
            num = pc * (1 + betas * V * s[:, c:c + 1])
            den = 1 + betas * V * (s @ p)[:, None]
            nll["C1"] -= np.log(num / den)
            # C2: 混ぜ合わせに感染分布を足す
            nll["C2"] -= np.log((1 - lamA - mus) * pc + lamA * pcache + mus * s[:, c:c + 1])
            # C3: 生態系の圧力 e = A·x/Σx で土台を傾ける
            e = popA / popS                                         # (nH, V) = x·A / Σx
            w = p[None, None, :] * np.exp(gam * e[:, None, :])       # (nH, nγ, V)
            peco = w[:, :, c] / w.sum(-1)
            nll["C3"] -= np.log((1 - lamA) * peco + lamA * pcache)
        else:
            for k in ("C1", "C2", "C3"):
                nll[k] -= math.log(pA if k != "C1" else pc)
        # 状態更新
        tr *= decA
        tr[c] += 1.0
        trS = trS * decA + 1.0
        sp *= dec2
        sp += W[c][None, :]
        spS = spS * dec2 + 1.0
        popA *= dec2
        popA += A[c][None, :]
        popS = popS * dec2 + 1.0
    n = len(ids) - 1
    return {k: (v / n / LOG2 if np.isscalar(v) else v / n / LOG2) for k, v in nll.items()}


def main():
    pathlib.Path("results").mkdir(exist_ok=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", type=int, default=2)
    ap.add_argument("--window", type=int, default=50, help="共起を数える ±K 字")
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
    mem = json.loads(pathlib.Path(f"results/memory_{corpus}.json").read_text())
    lamA, HA = mem["chosen"]["mixture_ref"]["lambda"], mem["chosen"]["mixture_ref"]["H"]
    base = NgramBase(tr, V, a.order)
    W, A = build_networks(pmi_matrix(tr, V, a.window))
    H2s = [30, 300, 3000]
    betas = [0.01, 0.03, 0.1, 0.3, 1.0]
    mus = [0.0, 0.01, 0.03, 0.05, 0.1]
    gammas = [0.0, 0.3, 1.0, 3.0]
    r = evaluate(base, va, W, A, lamA, HA, H2s, betas, mus, gammas)
    pick = {}
    for k, knob, vals in (("C1", "beta", betas), ("C2", "mu", mus), ("C3", "gamma", gammas)):
        i = np.unravel_index(np.argmin(r[k]), r[k].shape)
        pick[k] = {"H": H2s[i[0]], knob: vals[i[1]], "valid": float(r[k][i])}
    print(f"[{corpus}] valid（{len(va):,}字）: 土台 {r['base']:.3f} / A {r['A']:.3f}（λ={lamA}, H={HA}）")
    for k in ("C1", "C2", "C3"):
        print(f"  {k} 最良 {pick[k]['valid']:.3f}  {pick[k]}")
    rt = {}
    for k, knob in (("C1", "beta"), ("C2", "mu"), ("C3", "gamma")):
        kw = {"betas": [pick["C1"]["beta"]], "mus": [pick["C2"]["mu"]], "gammas": [pick["C3"]["gamma"]]}
        rr = evaluate(base, te, W, A, lamA, HA, [pick[k]["H"]], **kw)
        rt[k] = float(rr[k][0, 0])
        rt["base"], rt["A"] = rr["base"], rr["A"]
    print(f"  test: 土台 {rt['base']:.3f} / A {rt['A']:.3f} / C1 伝染(原典形) {rt['C1']:.3f} / "
          f"C2 伝染上乗せ {rt['C2']:.3f} / C3 生態系 {rt['C3']:.3f}", flush=True)
    out = {"corpus": corpus, "order": a.order, "window": a.window, "valid_chars": len(va), "test_chars": len(te),
           "A": {"lambda": lamA, "H": HA},
           "grid": {"H2s": H2s, "betas": betas, "mus": mus, "gammas": gammas,
                    **{k: r[k].tolist() for k in ("C1", "C2", "C3")}},
           "valid": {"base": r["base"], "A": r["A"], **{k: pick[k]["valid"] for k in pick}},
           "chosen": pick, "test": rt}
    pathlib.Path(f"results/family_c_{corpus}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
