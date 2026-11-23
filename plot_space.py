"""空間系: 次元のはしご（語を何次元に置けば言葉になるか）。軸ラベルは英語。"""
import json, math, pathlib, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = pathlib.Path(__file__).resolve().parent / "results"
corpus = sys.argv[1] if len(sys.argv) > 1 else "zenn"
rs = json.loads((R / f"space_{corpus}.json").read_text())
base = sorted({r["d"]: r for r in rs if r["variant"] == []}.values(), key=lambda r: r["d"])
inert = sorted([r for r in rs if r["variant"] == ["inertia"]], key=lambda r: r["d"])
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot([r["d"] for r in base], [r["test"] for r in base], "o-", color="#16a37a", label="space LM (position + velocity + mass)")
ax.plot([r["d"] for r in inert], [r["test"] for r in inert], "s--", color="#7c5cff", label="+ inertia (longer context)")
tun = [r for r in rs if r["variant"] == ["tunnel"]]
if tun:
    ax.plot([2], [tun[0]["test"]], "^", color="#e08a00", ms=10, label="+ tunnel (2D)")
refs = {"zenn": [("Sleep LM (pure rule)", 6.093), ("count bigram (baseline, 1-char context)", 5.512), ("tiny GPT 256-dim", 5.313)],
        "aozora": [("Sleep LM (pure rule)", 7.034), ("count bigram", 5.766), ("tiny GPT 256-dim (12k steps)", 4.379)]}[corpus]
for name, y in refs:
    ax.axhline(y, color="gray", ls=":", lw=1)
    ax.text(1, y + 0.04, name, color="gray", fontsize=8)
ax.set_xscale("log", base=2)
ax.set_xlabel("dimensions of the space words live in")
ax.set_ylabel("held-out bits per char (lower = better)")
ax.set_title(f"How many dimensions does a word need? ({corpus})")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(R / f"space_dims_{corpus}.png", dpi=120)
print("saved")
