"""記憶系（忘却LM・免疫LM）を教材で育てる — 「どこまで育つか」第4弾。

原典（Forgetting_LM v0.1 / Immune_LM v0.1）の仕組みはそのまま、遷移重みだけを教材の数え上げ n-gram に差し替える。
  土台   : P0(c | 文脈) = 補間つき絶対値割引 n-gram（D=0.75、ladder_scale.py と同じ）
  忘却LM : P(c) ∝ P0(c) × (1 + β × 痕跡_c)   痕跡は字が出ると +1、毎字 半減期 H で減衰（原典と同じ）
  免疫LM : P(c) ∝ P0(c) ÷ (1 + 強さ × 抗体_c)  抗体は字が出ると +1、同じ文字種の字に +0.3（原典の「同じ役割」を
           文字種=ひらがな/カタカナ/漢字/英数字/その他 に置き換え）、半減期 H で減衰
参考（原典の形ではない比較用）: 混ぜ合わせ型キャッシュ P = (1−λ)P0 + λ×痕跡/Σ痕跡（言語モデルで実績のある形）。
「原典の掛け算の形が弱いのか、文字単位という粒度が弱いのか」を切り分けるために測る。
学ぶつまみは (β, H) / (強さ, H) / (λ, H) だけ。valid（先頭 N 字）で格子探索し、選んだ設定の test を報告。
β=0 / 強さ=0 は土台と完全一致（対照）。
"""
import argparse
import json
import math
import os
import pathlib
from collections import Counter, defaultdict

import numpy as np

from sleep_growth import build_vocab, encode, load

LOG2 = math.log(2)
D = 0.75


class NgramBase:
    """補間つき絶対値割引 n-gram の「次の字の分布ベクトル」を返す。"""

    def __init__(self, train_ids, V, order):
        self.V, self.order = V, order
        uni = np.bincount(train_ids, minlength=V) + 1.0
        self.uni = uni / uni.sum()
        tr = train_ids.tolist()
        self.tables = []
        for k in range(1, order + 1):
            c = defaultdict(Counter)
            for t in range(k, len(tr)):
                c[tuple(tr[t - k:t])][tr[t]] += 1
            tab = {}
            for ctx, cnt in c.items():
                idx = np.fromiter(cnt.keys(), dtype=np.int64, count=len(cnt))
                val = np.fromiter(cnt.values(), dtype=np.float64, count=len(cnt))
                tab[ctx] = (idx, np.maximum(val - D, 0.0), float(val.sum()), len(cnt))
            self.tables.append(tab)
            del c

    def dist(self, hist):
        p = self.uni.copy()
        for k in range(1, min(len(hist), self.order) + 1):
            e = self.tables[k - 1].get(tuple(hist[-k:]))
            if e is None:
                break
            idx, disc, tot, types = e
            p *= D * types / tot
            p[idx] += disc / tot
        return p


def char_class(ch):
    o = ord(ch[0]) if ch else 0
    if 0x3040 <= o <= 0x309F:
        return 0  # ひらがな
    if 0x30A0 <= o <= 0x30FF:
        return 1  # カタカナ
    if 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF:
        return 2  # 漢字
    if ch.isascii() and ch.isalnum():
        return 3  # 英数字
    return 4      # その他（記号・<unk> など）


LAMBDAS = [0.01, 0.03, 0.1, 0.2, 0.3]


def evaluate(base, ids, betas, Hs, strengths, iHs, cls, cross=0.3, lambdas=LAMBDAS):
    """1回の走査で、全ての (β,H) と (強さ,H) の bits/char を同時に計算する。"""
    V = base.V
    fdec = np.array([0.5 ** (1.0 / h) for h in Hs])[:, None]
    idec = np.array([0.5 ** (1.0 / h) for h in iHs])[:, None]
    trace = np.zeros((len(Hs), V))
    anti = np.zeros((len(iHs), V))
    betas = np.asarray(betas)[None, :]                      # (1, nβ)
    strengths = np.asarray(strengths)[:, None, None]        # (nS, 1, 1)
    f_nll = np.zeros((len(Hs), betas.shape[1]))
    i_nll = np.zeros((strengths.shape[0], len(iHs)))
    lam = np.asarray(lambdas)[None, :]
    m_nll = np.zeros((len(Hs), lam.shape[1]))
    base_nll = 0.0
    n_cls = cls.max() + 1
    cls_onehot = np.eye(n_cls)[cls]                         # (V, n_cls)
    ids = ids.tolist()
    order = base.order
    for t in range(1, len(ids)):
        c = ids[t]
        p = base.dist(ids[max(0, t - order):t])
        pc = p[c]
        base_nll -= math.log(pc)
        # 忘却: P = p(1+β tr)/ (1 + β Σ p tr)
        dot = trace @ p                                     # (nH,)
        f_nll -= np.log(pc * (1 + betas * trace[:, c:c + 1])) - np.log(1 + betas * dot[:, None])
        # 参考: 混ぜ合わせ型キャッシュ（痕跡が空の最初の1字は土台のみ）
        tot = trace.sum(1, keepdims=True)
        pcache = np.where(tot > 0, trace[:, c:c + 1] / np.maximum(tot, 1e-12), pc)
        m_nll -= np.log((1 - lam) * pc + lam * pcache)
        # 免疫: P = p/(1+s a) / Σ p/(1+s a)
        denom = 1 + strengths * anti[None, :, :]            # (nS, nH, V)
        z = (p[None, None, :] / denom).sum(-1)              # (nS, nH)
        i_nll -= np.log(pc / denom[:, :, c]) - np.log(z)
        # 状態更新（原典どおり: 出た字に +1、減衰）
        trace *= fdec
        trace[:, c] += 1.0
        anti *= idec
        anti += cross * cls_onehot[:, cls[c]][None, :]
        anti[:, c] += 1.0 - cross
    n = len(ids) - 1
    return base_nll / n / LOG2, f_nll / n / LOG2, i_nll / n / LOG2, m_nll / n / LOG2


def main():
    pathlib.Path("results").mkdir(exist_ok=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", type=int, default=2, help="文脈の長さ（2=3-gram）")
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
    base = NgramBase(tr, V, a.order)
    Hs = [3, 10, 30, 100, 300, 1000, 3000]
    betas = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
    iHs = [3, 10, 30, 100]
    strengths = [0.03, 0.1, 0.3, 1.0, 3.0]

    b_va, f_va, i_va, m_va = evaluate(base, va, betas, Hs, strengths, iHs, cls)
    mi = np.unravel_index(np.argmin(m_va), m_va.shape)
    mH, ml = Hs[mi[0]], LAMBDAS[mi[1]]
    fi = np.unravel_index(np.argmin(f_va), f_va.shape)
    ii = np.unravel_index(np.argmin(i_va), i_va.shape)
    fH, fb = Hs[fi[0]], betas[fi[1]]
    iS, iH = strengths[ii[0]], iHs[ii[1]]
    print(f"[{corpus}] 土台 {a.order + 1}-gram  valid {b_va:.3f}")
    print(f"  忘却LM valid 最良 {f_va[fi]:.3f}（β={fb}, H={fH}）")
    print(f"  免疫LM valid 最良 {i_va[ii]:.3f}（強さ={iS}, H={iH}）／ 最弱設定でも {i_va.max():.3f}")
    print(f"  参考 混ぜ合わせ型 valid 最良 {m_va[mi]:.3f}（λ={ml}, H={mH}）")
    # test は選んだ設定だけ（ついでに同じ走査で全部出るが、報告は選んだ設定のみ）
    b_te, f_te, i_te, _ = evaluate(base, te, [fb], [fH], [iS], [iH], cls)
    _, _, _, m_te = evaluate(base, te, [fb], [mH], [iS], [iH], cls, lambdas=[ml])
    print(f"  test: 土台 {b_te:.3f} / 忘却LM {f_te[0, 0]:.3f} / 免疫LM {i_te[0, 0]:.3f} / 参考 混ぜ合わせ型 {m_te[0, 0]:.3f}")
    out = {"corpus": corpus, "order": a.order, "valid_chars": len(va), "test_chars": len(te),
           "grid": {"Hs": Hs, "betas": betas, "iHs": iHs, "strengths": strengths,
                    "forget_valid": f_va.tolist(), "immune_valid": i_va.tolist(),
                    "lambdas": LAMBDAS, "mixture_valid": m_va.tolist()},
           "valid": {"base": b_va, "forget": float(f_va[fi]), "immune": float(i_va[ii])},
           "chosen": {"forget": {"beta": fb, "H": fH}, "immune": {"strength": iS, "H": iH},
                      "mixture_ref": {"lambda": ml, "H": mH}},
           "test": {"base": b_te, "forget": float(f_te[0, 0]), "immune": float(i_te[0, 0]),
                    "mixture_ref": float(m_te[0, 0])}}
    pathlib.Path(f"results/memory_{corpus}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
