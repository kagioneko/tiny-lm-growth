"""睡眠LM 育成実験 v0.2 — 「どこまで育つか」第1弾

原典 Sleep LM v0.1との違いは1点だけ:
  覚醒フェーズで「自分で生成する」代わりに「コーパスを読む」。
睡眠ルールは原典のまま: その覚醒で使われた遷移 ×1.1 / 使われなかった遷移 ×0.9 → 行ごとに再正規化。
（使用回数の多寡ではなく、使われたか否かの二値。）

実装上の注意:
- 文字単位。語彙は train に2回以上出た文字 + <unk>。
- 重みは対数で持つ（×0.9 の繰り返しでアンダーフローして確率0→bits無限大になるのを防ぐ。
  数学的には原典と同じ更新）。初期値は一様（原典は擬似ランダムだが、コーパスと無関係な
  偏りを持ち込まないため一様にした）。
- その覚醒で一度も出てこなかった行は、全要素 ×0.9 → 再正規化で変化なし（原典どおり）。

物差し（はしご）: test 記事（育成中に一度も読まない）の bits/char。
  段0 一様分布 / 段1 ユニグラム（文字頻度）/ 段2 数え上げバイグラム（add-k、k は valid で選択）
  睡眠LMは構造上「直前1文字→次の1文字」なので、段2（数え上げバイグラム）が比較基準。
"""
import argparse
import json
import math
import pathlib

import numpy as np

DATA = pathlib.Path(__file__).resolve().parent / "data"
UNK = "<unk>"


def load():
    """教材の切り替え: 環境変数 LM_CORPUS=zenn（既定）/ aozora、LM_TRAIN_LIMIT=N（学習用を先頭N文字に）。"""
    import os
    corpus = os.environ.get("LM_CORPUS", "zenn")
    base = DATA if corpus == "zenn" else DATA / corpus
    out = {k: (base / f"{k}.txt").read_text(encoding="utf-8") for k in ("train", "valid", "test")}
    limit = int(os.environ.get("LM_TRAIN_LIMIT", "0"))
    if limit:
        out["train"] = out["train"][:limit]
    return out


def build_vocab(train, min_count=2):
    from collections import Counter
    c = Counter(train)
    chars = sorted(ch for ch, n in c.items() if n >= min_count)
    itos = [UNK] + chars
    return itos, {ch: i for i, ch in enumerate(itos)}


def encode(text, stoi):
    return np.array([stoi.get(ch, 0) for ch in text], dtype=np.int64)


def bpc_from_logp(logp_matrix, ids):
    """行=直前文字、列=次文字 の対数確率行列で ids を評価（最初の1文字は除く）。"""
    lp = logp_matrix[ids[:-1], ids[1:]]
    return float(-lp.mean() / math.log(2))


# ---- はしご（基準モデル） ----
def ladder(train_ids, valid_ids, test_ids, V):
    out = {"uniform": math.log2(V)}
    uni = np.bincount(train_ids, minlength=V) + 1.0
    uni_lp = np.log(uni / uni.sum())
    out["unigram"] = float(-uni_lp[test_ids[1:]].mean() / math.log(2))
    counts = np.zeros((V, V))
    np.add.at(counts, (train_ids[:-1], train_ids[1:]), 1)
    best = None
    for k in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
        lp = np.log((counts + k) / (counts + k).sum(axis=1, keepdims=True))
        v = bpc_from_logp(lp, valid_ids)
        if best is None or v < best[1]:
            best = (k, v, lp)
    out["bigram_count"] = bpc_from_logp(best[2], test_ids)
    out["bigram_k"] = best[0]
    return out, best[2]


# ---- 睡眠LM ----
class SleepReader:
    def __init__(self, V, strength=0.1):
        """strength=0.1 が原典（×1.1 / ×0.9）。×(1+s) / ×(1-s) として掃引する。"""
        self.V = V
        self.log_boost = math.log(1 + strength)
        self.log_decay = math.log(1 - strength)
        self.logW = np.full((V, V), -math.log(V))  # 一様・正規化済み

    def wake_and_sleep(self, chunk_ids):
        """覚醒: chunk を読んで使われた遷移を記録 → 睡眠: 原典ルールで更新。"""
        prev, nxt = chunk_ids[:-1], chunk_ids[1:]
        if len(prev) == 0:
            return
        used = np.unique(prev * self.V + nxt)
        rows = np.unique(prev)
        # 使われた行だけ更新（使われなかった行は ×0.9 → 再正規化で不変）
        self.logW[rows] += self.log_decay
        r, c = np.divmod(used, self.V)
        self.logW[r, c] += self.log_boost - self.log_decay
        sub = self.logW[rows]
        m = sub.max(axis=1, keepdims=True)
        self.logW[rows] = sub - (m + np.log(np.exp(sub - m).sum(axis=1, keepdims=True)))

    def bpc(self, ids):
        return bpc_from_logp(self.logW, ids)

    def sample(self, start_id, n, rng, temperature=1.0):
        out = [start_id]
        cur = start_id
        for _ in range(n):
            lp = self.logW[cur] / temperature
            p = np.exp(lp - lp.max())
            p[0] = 0.0  # <unk> は出さない
            p /= p.sum()
            cur = int(rng.choice(self.V, p=p))
            out.append(cur)
        return out


def run(wake_chars, epochs, eval_every, seed=0, strength=0.1):
    texts = load()
    itos, stoi = build_vocab(texts["train"])
    V = len(itos)
    tr, va, te = (encode(texts[k], stoi) for k in ("train", "valid", "test"))
    lad, bigram_lp = ladder(tr, va, te, V)
    model = SleepReader(V, strength)
    curve = [{"cycle": 0, "chars_read": 0, "valid_bpc": model.bpc(va), "test_bpc": model.bpc(te)}]
    best_logW = model.logW.copy()
    cycle = 0
    chars_read = 0
    for ep in range(epochs):
        for start in range(0, len(tr) - 1, wake_chars):
            # 覚醒の境目で遷移が途切れないよう、1文字重ねて切る
            chunk = tr[start:start + wake_chars + 1]
            model.wake_and_sleep(chunk)
            cycle += 1
            chars_read += len(chunk) - 1
            if cycle % eval_every == 0:
                curve.append({"cycle": cycle, "epoch": ep, "chars_read": chars_read,
                              "valid_bpc": model.bpc(va), "test_bpc": model.bpc(te)})
                if curve[-1]["valid_bpc"] < min(r["valid_bpc"] for r in curve[:-1]):
                    best_logW = model.logW.copy()
    if curve[-1]["cycle"] != cycle:
        curve.append({"cycle": cycle, "epoch": epochs - 1, "chars_read": chars_read,
                      "valid_bpc": model.bpc(va), "test_bpc": model.bpc(te)})
    best = min(curve, key=lambda r: r["valid_bpc"])
    rng = np.random.default_rng(seed)
    start = stoi.get("ね", 1)
    sample = "".join(itos[i] for i in model.sample(start, 120, rng))
    final_logW = model.logW
    model.logW = best_logW  # valid 最良時点のモデルからも出力させる
    sample_best = "".join(itos[i] for i in model.sample(start, 120, np.random.default_rng(seed)))
    model.logW = final_logW
    # 比較用: 数え上げバイグラムからのサンプル
    rng2 = np.random.default_rng(seed)
    p = np.exp(bigram_lp)
    cur, bi = start, [start]
    for _ in range(120):
        q = p[cur].copy(); q[0] = 0; q /= q.sum()
        cur = int(rng2.choice(V, p=q)); bi.append(cur)
    return {
        "config": {"wake_chars": wake_chars, "strength": strength, "epochs": epochs, "eval_every": eval_every, "seed": seed,
                   "vocab": V, "train_chars": len(tr), "valid_chars": len(va), "test_chars": len(te)},
        "ladder": lad,
        "curve": curve,
        "best_by_valid": best,
        "final": curve[-1],
        "sample_sleep": sample,
        "sample_sleep_best": sample_best,
        "sample_bigram": "".join(itos[i] for i in bi),
    }


def main():
    pathlib.Path("results").mkdir(exist_ok=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--wake-chars", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--strength", type=float, default=0.1, help="0.1=原典 (×1.1/×0.9)")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    r = run(a.wake_chars, a.epochs, a.eval_every, strength=a.strength)
    lad = r["ladder"]
    print(f"vocab={r['config']['vocab']}  train={r['config']['train_chars']:,}字")
    print(f"はしご(test bpc): 一様 {lad['uniform']:.3f} / ユニグラム {lad['unigram']:.3f} / "
          f"数え上げバイグラム {lad['bigram_count']:.3f} (k={lad['bigram_k']})")
    for row in r["curve"][:: max(1, len(r["curve"]) // 12)]:
        print(f"  cycle {row['cycle']:5d}  読んだ {row['chars_read']:7,}字  test {row['test_bpc']:.3f}")
    b, f = r["best_by_valid"], r["final"]
    print(f"最良(valid基準): cycle {b['cycle']} test {b['test_bpc']:.3f} / 最終: test {f['test_bpc']:.3f}")
    print("睡眠LM(最良時点)の出力:", r["sample_sleep_best"].replace("\n", "⏎"))
    print("睡眠LM(最終)の出力:", r["sample_sleep"].replace("\n", "⏎"))
    print("バイグラム出力:", r["sample_bigram"].replace("\n", "⏎"))
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
