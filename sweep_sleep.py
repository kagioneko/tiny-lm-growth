"""睡眠LM 育成の掃引: 覚醒の長さ × 睡眠の強さ。

選び方（リーク防止）: 各設定の中の「いつ止めるか」も、設定どうしの比較も valid bpc で決める。
test bpc は選ばれた点で報告するだけ。
"""
import csv
import json
import pathlib

from sleep_growth import run

OUT = pathlib.Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)

rows, curves = [], {}
for strength in (0.3, 0.1, 0.03, 0.01):
    for wake in (100, 500, 2000, 10000, 20000, 71202):
        epochs = 10 if wake < 10000 else 60
        ev = max(1, 2000 // wake)
        r = run(wake, epochs=epochs, eval_every=ev, strength=strength)
        b, f = r["best_by_valid"], r["final"]
        rows.append({"strength": strength, "wake_chars": wake, "epochs": epochs,
                     "best_cycle": b["cycle"], "best_chars_read": b["chars_read"],
                     "best_valid_bpc": round(b["valid_bpc"], 3), "test_bpc_at_best": round(b["test_bpc"], 3),
                     "final_test_bpc": round(f["test_bpc"], 3)})
        curves[f"s{strength}_w{wake}"] = r["curve"]
        print(rows[-1], flush=True)
lad = r["ladder"]
win = min(rows, key=lambda x: x["best_valid_bpc"])
print("ladder:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in lad.items()})
print("valid で選んだ最良設定:", win)
with open(OUT / "sleep_sweep.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)
(OUT / "sleep_sweep_curves.json").write_text(
    json.dumps({"ladder": lad, "winner": win, "curves": curves}, ensure_ascii=False), encoding="utf-8")
