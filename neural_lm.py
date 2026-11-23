"""ニューラル睡眠LM v1.0 — 小さな文字単位 GPT に「起きて読む・寝て整理する」を入れる。

語彙・分割・物差し（held-out の bits/char）は睡眠LM v0.2/v0.3 と同一。
学習のしかた（総更新回数はそろえる）:
  iid    : 普通の学習。学習用の全文からランダムに切り出して学習する（LLM の通常の学び方）
  stream : 起きているだけ。記事を先頭から順に読み、読んだ所だけで学習し、二度と戻らない
  sleep  : stream と同じ順で読みつつ、毎晩「これまで読んだ所」からランダムに思い出して学習する
           （海馬リプレイ風。覚醒1回 = wake_chars 字、睡眠1回 = 覚醒と同じ回数の更新）
評価: 文脈 ctx 字の窓を半分ずつずらし、各窓の後半（＝十分な文脈がある位置）だけを数える。
"""
import argparse
import json
import math
import pathlib
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sleep_growth import build_vocab, encode, load

import os as _os
torch.set_num_threads(int(_os.environ.get("LM_THREADS", "6")))
LOG2 = math.log(2)


class Block(nn.Module):
    def __init__(self, d, h, drop):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, dropout=drop, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d), nn.Dropout(drop))

    def forward(self, x, mask):
        a = self.ln1(x)
        x = x + self.attn(a, a, a, attn_mask=mask, need_weights=False)[0]
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, V, ctx=128, d=128, layers=4, heads=4, drop=0.2):
        super().__init__()
        self.ctx = ctx
        self.tok = nn.Embedding(V, d)
        self.pos = nn.Embedding(ctx, d)
        self.drop = nn.Dropout(drop)
        self.blocks = nn.ModuleList(Block(d, heads, drop) for _ in range(layers))
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, V, bias=False)
        self.head.weight = self.tok.weight  # 重み共有
        self.register_buffer("mask", torch.triu(torch.ones(ctx, ctx, dtype=torch.bool), 1))

    def forward(self, idx):
        T = idx.shape[1]
        x = self.drop(self.tok(idx) + self.pos(torch.arange(T)))
        for b in self.blocks:
            x = b(x, self.mask[:T, :T])
        return self.head(self.ln(x))


def batch_from(ids, lo, hi, B, ctx, rng):
    """ids[lo:hi] の範囲からランダムに B 本切り出す。"""
    hi = max(hi, lo + ctx + 2)
    starts = rng.integers(lo, hi - ctx - 1, size=B)
    x = torch.from_numpy(np.stack([ids[s:s + ctx] for s in starts]))
    y = torch.from_numpy(np.stack([ids[s + 1:s + ctx + 1] for s in starts]))
    return x, y


@torch.no_grad()
def bpc(model, ids, ctx):
    model.eval()
    half = ctx // 2
    tot, n = 0.0, 0
    starts = list(range(0, len(ids) - ctx - 1, half))
    for i in range(0, len(starts), 64):
        ss = starts[i:i + 64]
        x = torch.from_numpy(np.stack([ids[s:s + ctx] for s in ss]))
        y = torch.from_numpy(np.stack([ids[s + 1:s + ctx + 1] for s in ss]))
        lp = F.log_softmax(model(x), -1).gather(-1, y[..., None])[..., 0]
        for j, s in enumerate(ss):
            k = 0 if s == 0 else half  # 先頭の窓だけは全位置、以降は後半だけ
            tot -= lp[j, k:].sum().item()
            n += ctx - k
    model.train()
    return tot / n / LOG2


def train(mode, steps, wake_chars=5000, B=32, ctx=128, lr=3e-3, seed=0, eval_every=100, valid_limit=0, save=None, **arch):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    texts = load()
    itos, stoi = build_vocab(texts["train"])
    V = len(itos)
    tr, va, te = (encode(texts[k], stoi) for k in ("train", "valid", "test"))
    if valid_limit:
        va = va[:valid_limit]  # 途中評価（止めどき選び）用。test は最後に全文で測る
    model = TinyGPT(V, ctx=ctx, **arch)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    n_params = sum(p.numel() for p in model.parameters())

    # stream / sleep 用の読む順番: 先頭から wake_chars ずつ。全文を読み終えたら最初から読み直す。
    n_chunks = math.ceil(len(tr) / wake_chars)
    if mode == "iid":
        per_wake = None
    else:
        # 覚醒1回あたりの更新回数（総更新回数 steps を読み回数で割る）。sleep は半分を睡眠に回す。
        passes = max(1, steps // (n_chunks * 20))
        per_wake = max(1, steps // (n_chunks * passes))
    curve, best, best_state = [], None, None
    t0 = time.time()
    step = 0

    def do_step(lo, hi):
        nonlocal step
        x, y = batch_from(tr, lo, hi, B, ctx, rng)
        loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        step += 1
        if step % eval_every == 0 or step == steps:
            v = bpc(model, va, ctx)
            curve.append({"step": step, "valid_bpc": v, "sec": round(time.time() - t0, 1)})
            nonlocal best, best_state
            if best is None or v < best["valid_bpc"]:
                best = {"step": step, "valid_bpc": v}
                best_state = {k: t.clone() for k, t in model.state_dict().items()}

    if mode == "iid":
        while step < steps:
            do_step(0, len(tr))
    else:
        c = 0
        while step < steps:
            lo = (c % n_chunks) * wake_chars
            hi = min(len(tr), lo + wake_chars + ctx + 1)
            read_hi = hi if c >= n_chunks else hi  # 1周目は「今読んでいる所まで」しか知らない
            k_wake = per_wake if mode == "stream" else per_wake // 2
            for _ in range(k_wake):
                if step >= steps:
                    break
                do_step(lo, hi)                     # 覚醒: 今読んでいる所
            if mode == "sleep":
                seen_hi = read_hi if c < n_chunks else len(tr)
                for _ in range(per_wake - k_wake):
                    if step >= steps:
                        break
                    do_step(0, seen_hi)             # 睡眠: これまで読んだ所からリプレイ
            c += 1

    model.load_state_dict(best_state)
    if save:  # 最良時点の重みを保存（gpt_cache.py などで再利用）
        pathlib.Path(save).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": best_state, "itos": itos, "arch": {"ctx": ctx, **arch}, "best": dict(best)}, save)
    best["test_bpc"] = bpc(model, te, ctx)
    # 最良時点のモデルから文を出す（温度0.8、「ねこさん」から）
    model.eval()
    g = torch.Generator().manual_seed(seed)
    idx = torch.tensor([[stoi.get(ch, 0) for ch in "ねこさん"]])
    with torch.no_grad():
        for _ in range(200):
            logits = model(idx[:, -ctx:])[0, -1] / 0.8
            logits[0] = -1e9
            nxt = torch.multinomial(F.softmax(logits, -1), 1, generator=g)
            idx = torch.cat([idx, nxt[None]], 1)
    sample = "".join(itos[i] for i in idx[0].tolist())
    return {"mode": mode, "steps": steps, "wake_chars": wake_chars, "params": n_params,
            "arch": {"ctx": ctx, **arch}, "best": best, "curve": curve, "sample": sample,
            "sec": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="iid,stream,sleep")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--wake-chars", type=int, default=5000)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=128)
    ap.add_argument("--drop", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--valid-limit", type=int, default=0)
    ap.add_argument("--out", default="results/neural.json")
    ap.add_argument("--save", default=None, help="最良の重みの保存先（mode が1つのときだけ）")
    a = ap.parse_args()
    res = []
    for m in a.modes.split(","):
        r = train(m, a.steps, a.wake_chars, ctx=a.ctx, lr=a.lr, eval_every=a.eval_every,
                  valid_limit=a.valid_limit, save=a.save, d=a.d, layers=a.layers, heads=a.heads, drop=a.drop)
        res.append(r)
        b = r["best"]
        print(f"{m:6s} params={r['params']:,} 最良 step {b['step']}: valid {b['valid_bpc']:.3f} "
              f"test {b['test_bpc']:.3f}  ({r['sec']}s)", flush=True)
        print("   出力:", r["sample"].replace("\n", "⏎")[:200], flush=True)
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
