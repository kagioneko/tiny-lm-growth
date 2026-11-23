"""空間系（共鳴・力学系・E=mc²・相対論・量子トンネル）を教材で育てる — 「どこまで育つか」第5弾。

原典はどれも「語は空間上の点、状態（読み手）が近い語を出す」。物理シミュレーションは確率を出さないので、
同じ形の確率版を作り、語の位置・質量などを勾配法で教材から学ぶ（DESIGN_next.md の B）。

骨格（力学系）:  P(次=i | 前=j) ∝ exp( log m_i − |x_i − q_j|² )
  x_i: 字 i の位置（d 次元）、m_i: 質量、q_j = x_j + u_j: 読み手の位置（前の字の位置 + その字ごとの速度 u_j）
要素（1つずつ足して効果を切り分ける）:
  no_velocity : u=0（原典の素の重力に近い。距離が対称なので「を→動詞」と「動詞→を」を区別できない）
  no_mass     : m=1
  emc2        : 出すのにコスト c²·m_i（重いほど引き寄せるが、出しにくい）→ logit −= c²·m_i（c は学習）
  relativity  : 重い字の近くで時間が遅れて留まる → 同じ字への遷移に γ·log(局所重力) を足す（γ は学習）
  tunnel      : 確率 p で距離と無関係に跳ぶ → P = (1−p)·局所 + p·ユニグラム（p は学習）
  resonance   : 1次元・倍音つき。x は周波数の対数、倍音比 h∈{1/2,1,3/2,2,3} の重み w_h と幅 σ を学習、速度なし
  inertia ρ   : 読み手の位置が過去を引きずる q_t = ρ·q_{t−1} + (1−ρ)(x_{c_t}+u_{c_t})（ρ は学習）＝少し長い文脈
inertia なしはバイグラム（直前1字）なので、学習は2字組の数え上げ行列で全件一括（正確・速い）。
inertia ありは系列を窓で切って学習・評価する。
"""
import argparse
import json
import math
import os
import pathlib
import time

import numpy as np
import torch
import torch.nn.functional as F

from sleep_growth import build_vocab, encode, load

torch.set_num_threads(int(os.environ.get("LM_THREADS", "2")))
LOG2 = math.log(2)
HARMONICS = torch.log(torch.tensor([0.5, 1.0, 1.5, 2.0, 3.0]))


class SpaceLM(torch.nn.Module):
    def __init__(self, V, d, uni, variant=()):
        super().__init__()
        self.V, self.d, self.var = V, d, set(variant)
        g = torch.Generator().manual_seed(0)
        self.x = torch.nn.Parameter(torch.randn(V, d, generator=g) * 0.5)
        self.u = torch.nn.Parameter(torch.zeros(V, d)) if "no_velocity" not in self.var and "resonance" not in self.var else None
        self.logm = torch.nn.Parameter(torch.log(torch.as_tensor(uni, dtype=torch.float32))) if "no_mass" not in self.var else None
        self.logscale = torch.nn.Parameter(torch.zeros(()))        # 距離のものさし（τ）
        if "emc2" in self.var:
            self.logc2 = torch.nn.Parameter(torch.tensor(-3.0))
        if "relativity" in self.var:
            self.gamma = torch.nn.Parameter(torch.tensor(0.0))
        if "tunnel" in self.var:
            self.tunnel_logit = torch.nn.Parameter(torch.tensor(-3.0))
            self.register_buffer("uni_logp", torch.log(torch.as_tensor(uni, dtype=torch.float32)))
        if "resonance" in self.var:
            self.logw = torch.nn.Parameter(torch.zeros(len(HARMONICS)))
        if "inertia" in self.var:
            self.rho_logit = torch.nn.Parameter(torch.tensor(0.0))

    def query(self, j):
        q = self.x[j]
        if self.u is not None:
            q = q + self.u[j]
        return q

    def logits_from_query(self, q, prev=None):
        """q: (..., d) 読み手の位置 → (..., V) の対数確率。prev: 直前の字（相対論の「留まる」に使う）。"""
        s = torch.exp(self.logscale)
        if "resonance" in self.var:
            # 1次元: 駆動周波数 q と各字の倍音 x_i + log h の近さ（ガウス）の重み付き和
            diff = q[..., None, :] - (self.x[None, :, :] + HARMONICS.view(1, 1, -1))   # (..., V, H)
            score = torch.logsumexp(torch.log_softmax(self.logw, 0) - s * diff ** 2, -1)
        else:
            d2 = (q ** 2).sum(-1, keepdim=True) - 2 * q @ self.x.T + (self.x ** 2).sum(-1)
            score = -s * d2
        if self.logm is not None:
            score = score + self.logm
            if "emc2" in self.var:
                score = score - torch.exp(self.logc2) * torch.exp(self.logm)
        if "relativity" in self.var and prev is not None:
            # 局所重力 = Σ m_k exp(−s|x_k − x_prev|²) の対数を、同じ字に留まる遷移にだけ足す
            xp = self.x[prev]
            d2p = (xp ** 2).sum(-1, keepdim=True) - 2 * xp @ self.x.T + (self.x ** 2).sum(-1)
            lm = self.logm if self.logm is not None else torch.zeros(self.V)
            grav = torch.logsumexp(lm - s * d2p, -1)
            bonus = torch.zeros_like(score)
            bonus.scatter_(-1, prev.unsqueeze(-1), (self.gamma * grav).unsqueeze(-1))
            score = score + bonus
        logp = torch.log_softmax(score, -1)
        if "tunnel" in self.var:
            p = torch.sigmoid(self.tunnel_logit)
            logp = torch.logaddexp(torch.log1p(-p) + logp, torch.log(p) + self.uni_logp)
        return logp

    def bigram_logp(self):
        j = torch.arange(self.V)
        return self.logits_from_query(self.query(j), prev=j)        # (V, V)

    def seq_logp(self, ids):
        """inertia 用: ids (B, T) → 各位置の次の字の対数確率 (B, T−1)。"""
        rho = torch.sigmoid(self.rho_logit)
        B, T = ids.shape
        q = self.query(ids[:, 0])
        out = []
        for t in range(T - 1):
            if t > 0:
                q = rho * q + (1 - rho) * self.query(ids[:, t])
            lp = self.logits_from_query(q, prev=ids[:, t])
            out.append(lp.gather(-1, ids[:, t + 1:t + 2])[:, 0])
        return torch.stack(out, 1)


def bigram_counts(ids, V):
    c = np.zeros((V, V), dtype=np.float32)
    np.add.at(c, (ids[:-1], ids[1:]), 1)
    return torch.from_numpy(c)


def windows(ids, L):
    n = (len(ids) - 1) // (L - 1)
    return torch.from_numpy(np.stack([ids[k * (L - 1):k * (L - 1) + L] for k in range(n)]))


def fit(variant, d, data, steps=1500, lr=0.05, eval_every=50, L=64, B=64):
    tr, va, te, V, uni = data
    model = SpaceLM(V, d, uni, variant)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    inertia = "inertia" in model.var
    if not inertia:
        Ctr, Cva, Cte = (bigram_counts(x, V) for x in (tr, va, te))
        ntr, nva, nte = Ctr.sum(), Cva.sum(), Cte.sum()

        def ev(C, n):
            with torch.no_grad():
                return float(-(C * model.bigram_logp()).sum() / n / LOG2)
    else:
        Wtr, Wva, Wte = windows(tr, L), windows(va, L), windows(te, L)
        rng = np.random.default_rng(0)

        def ev(W, _):
            with torch.no_grad():
                tot, n = 0.0, 0
                for k in range(0, len(W), 256):
                    lp = model.seq_logp(W[k:k + 256])
                    tot -= float(lp.sum())
                    n += lp.numel()
                return tot / n / LOG2
    best, best_state, curve = None, None, []
    t0 = time.time()
    for step in range(1, steps + 1):
        if not inertia:
            loss = -(Ctr * model.bigram_logp()).sum() / ntr
        else:
            loss = -model.seq_logp(Wtr[rng.integers(0, len(Wtr), B)]).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % eval_every == 0 or step == steps:
            v = ev(Cva, nva) if not inertia else ev(Wva, None)
            curve.append((step, v))
            if best is None or v < best[1]:
                best = (step, v)
                best_state = {k: t.clone() for k, t in model.state_dict().items()}
            elif v > best[1] + 0.05 and step > best[0] + 5 * eval_every:
                break
    model.load_state_dict(best_state)
    test = ev(Cte, nte) if not inertia else ev(Wte, None)
    extra = {}
    if "emc2" in model.var:
        extra["c2"] = torch.exp(model.logc2).item()
    if "relativity" in model.var:
        extra["gamma"] = model.gamma.item()
    if "tunnel" in model.var:
        extra["tunnel_p"] = torch.sigmoid(model.tunnel_logit).item()
    if "inertia" in model.var:
        extra["rho"] = torch.sigmoid(model.rho_logit).item()
    return {"variant": sorted(model.var), "d": d, "best_step": best[0], "valid": best[1], "test": test,
            "params": sum(p.numel() for p in model.parameters()), "sec": round(time.time() - t0, 1),
            **extra, "curve": curve}


PLANS = {
    "dims": [((), d) for d in (1, 2, 4, 8, 16, 32, 64)],
    "family": [(("resonance", "no_velocity"), 1), (("no_velocity",), 2), (("no_mass",), 2), ((), 2),
               (("emc2",), 2), (("relativity",), 2), (("tunnel",), 2)],
    "inertia": [(("inertia",), 2), (("inertia",), 16)],
}


def main():
    pathlib.Path("results").mkdir(exist_ok=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="dims,family,inertia")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--quick", action="store_true", help="動作確認用（各 30 回だけ）")
    a = ap.parse_args()
    corpus = os.environ.get("LM_CORPUS", "zenn")
    texts = load()
    itos, stoi = build_vocab(texts["train"])
    V = len(itos)
    tr, va, te = (encode(texts[k], stoi) for k in ("train", "valid", "test"))
    if os.environ.get("LM_VALID_LIMIT"):
        va = va[:int(os.environ["LM_VALID_LIMIT"])]
    uni = np.bincount(tr, minlength=V) + 1.0
    uni = uni / uni.sum()
    data = (tr, va, te, V, uni)
    out_path = pathlib.Path(f"results/space_{corpus}.json")
    results = json.loads(out_path.read_text()) if out_path.exists() and not a.quick else []
    done = {(tuple(r["variant"]), r["d"]) for r in results}
    for plan in a.plan.split(","):
        for variant, d in PLANS[plan]:
            key = (tuple(sorted(variant)), d)
            if key in done:
                continue
            r = fit(variant, d, data, steps=30 if a.quick else a.steps, eval_every=10 if a.quick else 50)
            extra = {k: round(v, 4) for k, v in r.items() if k in ("c2", "gamma", "tunnel_p", "rho")}
            print(f"[{corpus}] {'+'.join(r['variant']) or 'base'} d={d}: valid {r['valid']:.3f} test {r['test']:.3f} "
                  f"(step {r['best_step']}, {r['params']:,} params, {r['sec']}s) {extra}", flush=True)
            if not a.quick:
                results.append(r)
                out_path.write_text(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
