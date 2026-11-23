"""睡眠LM v0.3 — 長い文脈（n-gram）に睡眠ルールを載せる。

睡眠ルールは v0.2 と同じ（その覚醒で使った遷移 ×(1+s)、使わなかった遷移 ×(1−s)、行ごとに再正規化）。
変更点は「行」の単位だけ: 直前1文字 → 直前 k 文字（k=1..N）。

各文脈の行は疎に持つ:
  - 見たことのある次文字ごとの重み
  - 見たことのない次文字すべてで共有する「残り」の重み（1つにまとめても睡眠ルールは厳密に同じ:
    残りの文字は一度も使われていないので全員 ×(1−s) で、互いの比は変わらない）
予測: 文脈 k の行で見たことのある文字はその確率、見たことのない文字には「残り」の確率を
      1つ短い文脈 k−1 の予測で配り直す（バックオフ）。k=1 の行の「残り」は文字頻度（add-1 ユニグラム）で配る。
はしご用の比較対象として、同じバックオフ構造の数え上げ n-gram（補間つき絶対値割引）も実装する。
"""
import argparse
import json
import math
import pathlib
from collections import Counter, defaultdict

import numpy as np

from sleep_growth import UNK, build_vocab, encode, load

LOG2 = math.log(2)


def unigram_probs(train_ids, V):
    c = np.bincount(train_ids, minlength=V) + 1.0
    return c / c.sum()


# ---------------- 睡眠 n-gram ----------------
class SleepRow:
    __slots__ = ("w", "rest", "n_rest")

    def __init__(self, V):
        self.w = {}           # 次文字 → 対数重み
        self.rest = 0.0       # 見たことのない文字1つあたりの対数重み
        self.n_rest = V       # 見たことのない文字の数

    def logZ(self):
        vals = list(self.w.values())
        parts = vals + ([self.rest + math.log(self.n_rest)] if self.n_rest else [])
        m = max(parts)
        return m + math.log(sum(math.exp(p - m) for p in parts))

    def sleep(self, used, log_boost, log_decay):
        # 新しく使われた文字は「残り」から独立させる（その時点の重み = rest）
        for c in used:
            if c not in self.w:
                self.w[c] = self.rest
                self.n_rest -= 1
        for c in self.w:
            self.w[c] += log_boost if c in used else log_decay
        self.rest += log_decay
        z = self.logZ()
        for c in self.w:
            self.w[c] -= z
        self.rest -= z


class SleepNgram:
    def __init__(self, V, order, strength, uni):
        self.V, self.N = V, order
        self.lb, self.ld = math.log(1 + strength), math.log(1 - strength)
        self.rows = [dict() for _ in range(order + 1)]  # rows[k][context tuple] -> SleepRow（k=1..N）
        self.uni = uni

    def night(self, ids):
        """覚醒＝ids を読む → 睡眠＝各文脈の行に原典ルールを適用。"""
        used = [defaultdict(set) for _ in range(self.N + 1)]
        for t in range(1, len(ids)):
            nxt = int(ids[t])
            for k in range(1, self.N + 1):
                if t - k < 0:
                    break
                used[k][tuple(ids[t - k:t].tolist())].add(nxt)
        for k in range(1, self.N + 1):
            rows = self.rows[k]
            for ctx, s in used[k].items():
                row = rows.get(ctx)
                if row is None:
                    row = rows[ctx] = SleepRow(self.V)
                row.sleep(s, self.lb, self.ld)

    def prob(self, hist, c):
        """hist: 直前の文字列（tuple、長さ N まで）。バックオフで P(c | hist)。"""
        return self._p(hist, c, min(len(hist), self.N))

    def _p(self, hist, c, k):
        if k == 0:
            return float(self.uni[c])
        row = self.rows[k].get(hist[-k:])
        if row is None:
            return self._p(hist, c, k - 1)
        if c in row.w:
            return math.exp(row.w[c])
        # 残りの確率を、見たことのない文字に短い文脈の予測で配る
        rest_mass = math.exp(row.rest) * row.n_rest
        seen_lower = sum(self._p(hist, d, k - 1) for d in row.w)
        return rest_mass * self._p(hist, c, k - 1) / max(1e-12, 1.0 - seen_lower)


# ---------------- 数え上げ n-gram（補間つき絶対値割引） ----------------
class CountNgram:
    def __init__(self, V, order, uni, D=0.75):
        self.V, self.N, self.D, self.uni = V, order, D, uni
        self.c = [defaultdict(Counter) for _ in range(order + 1)]

    def fit(self, ids):
        for t in range(1, len(ids)):
            for k in range(1, self.N + 1):
                if t - k < 0:
                    break
                self.c[k][tuple(ids[t - k:t].tolist())][int(ids[t])] += 1

    def prob(self, hist, c):
        p = float(self.uni[c])
        for k in range(1, min(len(hist), self.N) + 1):
            cnt = self.c[k].get(hist[-k:])
            if not cnt:
                break
            tot = sum(cnt.values())
            p = (max(cnt.get(c, 0) - self.D, 0) + self.D * len(cnt) * p) / tot
        return p


def bpc(model, ids, N):
    s = 0.0
    for t in range(1, len(ids)):
        hist = tuple(ids[max(0, t - N):t].tolist())
        s -= math.log(max(model.prob(hist, int(ids[t])), 1e-300))
    return s / (len(ids) - 1) / LOG2


def grow(order, strength, wake, epochs, eval_every, data):
    tr, va, te, V, uni = data
    m = SleepNgram(V, order, strength, uni)
    curve, best = [], None
    night, read = 0, 0
    for ep in range(epochs):
        for start in range(0, len(tr) - 1, wake):
            chunk = tr[max(0, start - order):start + wake + 1]  # 文脈ぶん前を重ねる
            m.night(chunk)
            night += 1
            read += min(wake, len(tr) - 1 - start)
            if night % eval_every == 0:
                v = bpc(m, va, order)
                row = {"night": night, "chars_read": read, "valid_bpc": v}
                curve.append(row)
                if best is None or v < best["valid_bpc"]:
                    best = dict(row, test_bpc=bpc(m, te, order))
                elif v > best["valid_bpc"] + 0.5:  # 明らかに寝すぎに入ったら打ち切り（valid のみで判断）
                    return {"curve": curve, "best": best, "stopped_early": True}
    return {"curve": curve, "best": best, "stopped_early": False}


def main():
    pathlib.Path("results").mkdir(exist_ok=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--orders", default="1,2,3,4")
    ap.add_argument("--strengths", default="0.1,0.3")
    ap.add_argument("--wakes", default="5000,20000")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--out", default="results/sleep_ngram.json")
    a = ap.parse_args()
    texts = load()
    itos, stoi = build_vocab(texts["train"])
    V = len(itos)
    tr, va, te = (encode(texts[k], stoi) for k in ("train", "valid", "test"))
    uni = unigram_probs(tr, V)
    data = (tr, va, te, V, uni)

    ladder = {}
    for N in (1, 2, 3, 4, 5):
        cm = CountNgram(V, N, uni)
        cm.fit(tr)
        ladder[f"count_{N + 1}gram"] = {"valid": bpc(cm, va, N), "test": bpc(cm, te, N)}
        print(f"数え上げ {N + 1}-gram: valid {ladder[f'count_{N + 1}gram']['valid']:.3f}  "
              f"test {ladder[f'count_{N + 1}gram']['test']:.3f}", flush=True)

    runs = []
    for N in map(int, a.orders.split(",")):
        for s in map(float, a.strengths.split(",")):
            for w in map(int, a.wakes.split(",")):
                r = grow(N, s, w, a.epochs, max(1, 20000 // w), data)
                b = r["best"]
                runs.append({"context": N, "strength": s, "wake": w, **{k: b[k] for k in b},
                             "stopped_early": r["stopped_early"], "curve": r["curve"]})
                print(f"睡眠 文脈{N}字 s={s} 覚醒{w}: 最良 valid {b['valid_bpc']:.3f} "
                      f"test {b['test_bpc']:.3f}（{b['chars_read']:,}字読了時点）", flush=True)
    win = min(runs, key=lambda r: r["valid_bpc"])
    print("valid で選んだ最良:", {k: v for k, v in win.items() if k != "curve"})
    pathlib.Path(a.out).write_text(json.dumps({"ladder": ladder, "runs": runs}, ensure_ascii=False),
                                   encoding="utf-8")


if __name__ == "__main__":
    main()
