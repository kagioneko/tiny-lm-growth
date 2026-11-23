"""睡眠＋動的評価 — 読みながら学ぶ GPT に「夜の読み返し」を足すと伸びるか。

昼: dyn_eval.py と同じ（128字窓を半分ずつずらし、①今の重みで採点 → ②その窓で Adam 1歩）。
夜: N 窓ごとに R 歩、読み返し学習（1歩 = ランダムに選んだ B 窓のバッチ）。読み返し元は2種類:
  past  : test のうち**もう採点が済んだ**部分（今日読んだ所の復習）。採点前の字は使わないので先読みなし
  train : 学習用教材（昔覚えた知識の復習＝読み進めるうちの忘れ防止）
R=0 は dyn_eval.py の動的評価と完全一致。(源, N, R) と学習率は valid（先頭 LM_VALID_LIMIT 字）で選び、test は選んだ設定で1回。
出力: results/sleep_dyn_{corpus}_{ckpt}.json（設定ごとに追記、落ちても済んだ設定は飛ばす）と test の各位置対数確率 .npy
  LM_CORPUS=aozora python sleep_dyn.py checkpoints/aozora_iid_48k.pt
"""
import json
import math
import os
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from mix_all import load_gpt
from sleep_growth import encode, load

LOG2 = math.log(2)
B = 4                     # 夜の1歩のバッチ（窓の数）
SEED = 0


def sleep_dyn_logp(model, ids, ctx, lr, night_every, night_steps, source, train_ids=None):
    """model は呼び出し側で初期の重みに戻しておくこと。返り値: 各位置の対数確率（数えない位置は nan）。"""
    model.eval()                                             # dropout なし（昼も夜も）
    opt = torch.optim.Adam(model.parameters(), lr=lr) if lr > 0 else None
    rng = np.random.default_rng(SEED)
    half = ctx // 2
    out = np.full(len(ids), np.nan)
    scored_upto = 0                                          # ids[:scored_upto+1] の正解は採点済み
    for w, s in enumerate(range(0, len(ids) - ctx - 1, half)):
        x = torch.from_numpy(ids[s:s + ctx])[None]
        y = torch.from_numpy(ids[s + 1:s + ctx + 1])[None]
        lp = F.log_softmax(model(x), -1).gather(-1, y[..., None])[0, :, 0]
        k = 0 if s == 0 else half
        out[s + 1 + k:s + 1 + ctx] = lp[k:].detach().numpy()
        scored_upto = s + ctx
        if opt is None:
            continue
        opt.zero_grad()
        (-lp.mean()).backward()
        opt.step()
        if night_steps and (w + 1) % night_every == 0:       # 夜
            src = ids if source == "past" else train_ids
            hi = (scored_upto - ctx) if source == "past" else (len(train_ids) - ctx - 1)
            if hi <= 0:
                continue
            for _ in range(night_steps):
                st = rng.integers(0, hi, size=B)             # past: 窓の正解 st+1..st+ctx ≤ scored_upto
                xb = torch.from_numpy(np.stack([src[a:a + ctx] for a in st]))
                yb = torch.from_numpy(np.stack([src[a + 1:a + ctx + 1] for a in st]))
                loss = F.cross_entropy(model(xb).reshape(-1, model_vocab(model)), yb.reshape(-1))
                opt.zero_grad()
                loss.backward()
                opt.step()
    return out


def model_vocab(model):
    return model.head.out_features if hasattr(model, "head") else model.lm_head.out_features


def bpc(lg):
    m = ~np.isnan(lg)
    return float(-lg[m].mean() / LOG2)


def key(cfg):
    return f"{cfg['source']}|N{cfg['N']}|R{cfg['R']}|lr{cfg['lr']:g}"


def main():
    pathlib.Path("results").mkdir(exist_ok=True)
    ck = sys.argv[1]
    corpus = os.environ.get("LM_CORPUS", "aozora")
    vlim = int(os.environ.get("LM_VALID_LIMIT", "300000"))
    torch.set_num_threads(int(os.environ.get("LM_THREADS", "6")))
    name = pathlib.Path(ck).stem
    lr = float(os.environ.get("LM_SLEEP_LR", json.loads(
        pathlib.Path(f"results/dyn_{corpus}_{name}.json").read_text())["chosen_lr"]))
    out_json = pathlib.Path(f"results/sleep_dyn_{corpus}_{name}.json")
    res = json.loads(out_json.read_text()) if out_json.exists() else {"corpus": corpus, "checkpoint": name, "B": B,
                                                                       "seed": SEED, "valid": {}}
    model, ctx, itos = load_gpt(ck)
    stoi = {ch: i for i, ch in enumerate(itos)}
    texts = load()
    tr = encode(texts["train"], stoi)
    va = encode(texts["valid"], stoi)[:vlim]
    te = encode(texts["test"], stoi)
    init = {k: v.clone() for k, v in model.state_dict().items()}
    grid = [{"source": "none", "N": 0, "R": 0, "lr": lr}]
    for source in ("past", "train"):
        for N in (32, 128):
            for R in (2, 8):
                grid.append({"source": source, "N": N, "R": R, "lr": lr})
    for N, R in ((16, 8), (32, 16), (16, 16)):          # 第2弾: train が端（N 小・R 大）で最良だったので広げる
        grid.append({"source": "train", "N": N, "R": R, "lr": lr})
    for cfg in grid:
        kk = key(cfg)
        if kk in res["valid"]:
            continue
        t0 = time.time()
        model.load_state_dict(init)
        lg = sleep_dyn_logp(model, va, ctx, cfg["lr"], cfg["N"], cfg["R"], cfg["source"], tr)
        res["valid"][kk] = {**cfg, "bpc": bpc(lg)}
        out_json.write_text(json.dumps(res, ensure_ascii=False, indent=1))
        print(f"[{name}] valid {kk}: {res['valid'][kk]['bpc']:.4f}  ({time.time() - t0:.0f}s)", flush=True)
    best = min(res["valid"].values(), key=lambda r: r["bpc"])
    res["chosen"] = best
    npys = [pathlib.Path(f"results/sleep_dyn_{corpus}_{name}_{sp}.npy") for sp in ("test", "valid")]
    if res.get("test", {}).get("key") != key(best) or not all(n.exists() for n in npys):   # 値があっても npy が無ければ作り直す
        model.load_state_dict(init)
        lg = sleep_dyn_logp(model, te, ctx, best["lr"], best["N"], best["R"], best["source"], tr)
        res["test"] = {"key": key(best), "bpc": bpc(lg)}
        np.save(f"results/sleep_dyn_{corpus}_{name}_test.npy", lg)
        model.load_state_dict(init)
        lgv = sleep_dyn_logp(model, va, ctx, best["lr"], best["N"], best["R"], best["source"], tr)
        np.save(f"results/sleep_dyn_{corpus}_{name}_valid.npy", lgv)
        out_json.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"[{name}] 選んだ設定 {res['test']['key']} → test {res['test']['bpc']:.4f}", flush=True)


if __name__ == "__main__":
    main()
