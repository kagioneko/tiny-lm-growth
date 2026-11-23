"""動的評価 — test を読みながら GPT を追加学習する（読み終えた部分だけで学ぶので先読みなし）。

窓の切り方と数える位置は gpt_cache.gpt_logp_per_position と同じ（ctx 字の窓を半分ずつずらす）。
各窓で: ①今の重みで数える位置を採点 → ②その窓全体の損失で Adam を1歩。次の窓が採点する位置は
②で学んだ位置より後ろだけ（窓 s の学習対象は s+1..s+ctx、次の窓 s+half が数えるのは s+ctx+1 以降）。
dropout は切る。学習率は valid（先頭 LM_VALID_LIMIT 字）で選び、test は選んだ値で1回。lr=0 は静的評価と一致。
出力: results/dyn_{corpus}_{ckpt名}.json と、各位置の対数確率 results/dyn_{corpus}_{ckpt名}_{valid,test}.npy（mix_all.py が使う）
  LM_CORPUS=aozora python dyn_eval.py checkpoints/aozora_iid_48k.pt
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
LRS = [float(x) for x in os.environ.get("LM_DYN_LRS", "1e-5,3e-5,1e-4,3e-4").split(",")]


def dyn_logp(model, ids, ctx, lr):
    """model は呼び出し側で初期の重みに戻しておくこと（この関数は重みを書き換える）。"""
    model.eval()                                 # dropout なし
    opt = torch.optim.Adam(model.parameters(), lr=lr) if lr > 0 else None
    half = ctx // 2
    out = np.full(len(ids), np.nan)
    for s in range(0, len(ids) - ctx - 1, half):
        x = torch.from_numpy(ids[s:s + ctx])[None]
        y = torch.from_numpy(ids[s + 1:s + ctx + 1])[None]
        logits = model(x)
        lp = F.log_softmax(logits, -1).gather(-1, y[..., None])[0, :, 0]
        k = 0 if s == 0 else half
        out[s + 1 + k:s + 1 + ctx] = lp[k:].detach().numpy()
        if opt is not None:
            loss = -lp.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return out


def bpc(lg):
    m = ~np.isnan(lg)
    return float(-lg[m].mean() / LOG2)


def main():
    pathlib.Path("results").mkdir(exist_ok=True)
    ck = sys.argv[1]
    corpus = os.environ.get("LM_CORPUS", "aozora")
    vlim = int(os.environ.get("LM_VALID_LIMIT", "300000"))
    torch.set_num_threads(int(os.environ.get("LM_THREADS", "6")))
    name = pathlib.Path(ck).stem
    out_json = pathlib.Path(f"results/dyn_{corpus}_{name}.json")
    res = json.loads(out_json.read_text()) if out_json.exists() else {"corpus": corpus, "checkpoint": name, "valid": {}}
    model, ctx, itos = load_gpt(ck)
    stoi = {ch: i for i, ch in enumerate(itos)}
    texts = load()
    va = encode(texts["valid"], stoi)[:vlim]
    te = encode(texts["test"], stoi)
    init = {k: v.clone() for k, v in model.state_dict().items()}
    best = None
    for lr in [0.0] + LRS:
        key = repr(lr)
        npy = pathlib.Path(f"results/dyn_{corpus}_{name}_valid_lr{key}.npy")
        if key not in res["valid"] or not npy.exists():     # 落ちても済んだ lr は飛ばす（値と npy の両方がある時だけ）
            t0 = time.time()
            model.load_state_dict(init)
            lg = dyn_logp(model, va, ctx, lr)
            res["valid"][key] = bpc(lg)
            np.save(f"results/dyn_{corpus}_{name}_valid_lr{key}.npy", lg)
            out_json.write_text(json.dumps(res, ensure_ascii=False, indent=1))
            print(f"[{name}] valid lr={lr:g}: {res['valid'][key]:.4f}  ({time.time() - t0:.0f}s)", flush=True)
        if best is None or res["valid"][key] < res["valid"][repr(best)]:   # lr=0（静的）も候補に含める
            best = lr
    res["chosen_lr"] = best
    if "test" not in res or res["test"].get("lr") != best or not pathlib.Path(f"results/dyn_{corpus}_{name}_test.npy").exists():
        model.load_state_dict(init)
        lg0 = dyn_logp(model, te, ctx, 0.0)
        model.load_state_dict(init)
        lg = dyn_logp(model, te, ctx, best)
        res["test"] = {"lr": best, "static": bpc(lg0), "dynamic": bpc(lg)}
        np.save(f"results/dyn_{corpus}_{name}_test.npy", lg)
        out_json.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    # mix_all 用に選んだ lr の valid を別名で
    src = pathlib.Path(f"results/dyn_{corpus}_{name}_valid_lr{repr(best)}.npy")
    np.save(f"results/dyn_{corpus}_{name}_valid.npy", np.load(src))
    print(f"[{name}] test: 静的 {res['test']['static']:.4f} → 動的 {res['test']['dynamic']:.4f}（lr={best:g}）", flush=True)


if __name__ == "__main__":
    main()
