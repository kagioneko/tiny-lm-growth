"""睡眠LM 成長曲線の図（軸ラベルは英語: 日本語フォント非依存にするため）。"""
import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = pathlib.Path(__file__).resolve().parent / "results"
d = json.loads((R / "sleep_sweep_curves.json").read_text(encoding="utf-8"))
lad = d["ladder"]
fig, ax = plt.subplots(figsize=(8, 5))
for key, label in [("s0.1_w500", "original rule (x1.1/x0.9), wake=500"),
                   ("s0.3_w20000", "best by valid: strength 0.3, wake=20000"),
                   ("s0.01_w10000", "gentle: strength 0.01, wake=10000")]:
    c = d["curves"][key]
    ax.plot([r["chars_read"] for r in c], [r["test_bpc"] for r in c], marker=".", label=label)
for name, y, ls in [("uniform", lad["uniform"], ":"), ("unigram", lad["unigram"], "--"),
                    ("count bigram (baseline)", lad["bigram_count"], "-.")]:
    ax.axhline(y, color="gray", ls=ls, lw=1)
    ax.text(5e3, y + 0.15, f"{name} {y:.2f}", color="gray", fontsize=8)
ax.set_xscale("log"); ax.set_ylim(5, 16)
ax.set_xlabel("characters read (log)"); ax.set_ylabel("held-out bits per char (lower = better)")
ax.set_title("Sleep LM growth on Zenn articles: grows, then over-sleeps")
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(R / "sleep_growth.png", dpi=120)
print("saved", R / "sleep_growth.png")
