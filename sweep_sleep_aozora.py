"""純粋な睡眠LM（v0.2 ルール）を青空文庫で。設定の選択は valid、test は報告のみ。"""
import json, os, pathlib
os.environ["LM_CORPUS"] = "aozora"
from sleep_growth import run

pathlib.Path("results").mkdir(exist_ok=True)

rows = []
for s in (0.3, 0.1, 0.03):
    for wake in (20000, 100000, 400000):
        r = run(wake, epochs=6 if wake < 400000 else 20, eval_every=max(1, 200000 // wake), strength=s)
        b = r["best_by_valid"]
        rows.append({"strength": s, "wake": wake, "best_chars_read": b["chars_read"],
                     "valid": round(b["valid_bpc"], 3), "test": round(b["test_bpc"], 3),
                     "final_test": round(r["final"]["test_bpc"], 3), "sample_best": r["sample_sleep_best"][:80]})
        print(rows[-1], flush=True)
win = min(rows, key=lambda x: x["valid"])
print("valid で選んだ最良:", win)
pathlib.Path("results/sleep_aozora.json").write_text(json.dumps({"runs": rows, "winner": win}, ensure_ascii=False, indent=1))
