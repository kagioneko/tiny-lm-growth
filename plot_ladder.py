"""最終はしごの図: 教材ごとに、各段（一様→睡眠LM→数え上げ→小さいGPT→本物のLLM）の test bits/char。
結果 JSON がまだ無い段は飛ばす。軸ラベルは英語（日本語フォント非依存）。
"""
import json
import math
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = pathlib.Path(__file__).resolve().parent / "results"


def jload(name):
    p = R / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def best_neural(name):
    d = jload(name)
    if not d:
        return None
    r = min(d, key=lambda r: r["best"]["valid_bpc"])          # 選ぶのは valid、描くのはその test
    return (r["best"]["valid_bpc"], r["best"]["test_bpc"])


rungs = {"zenn": [], "aozora": []}
# Zenn
sweep = jload("sleep_sweep_curves.json")
ngram = jload("sleep_ngram.json")
rungs["zenn"].append(("uniform", math.log2(1025)))
if sweep:
    rungs["zenn"].append(("unigram", sweep["ladder"]["unigram"]))
if ngram:
    best_sleep = min(ngram["runs"], key=lambda r: r["valid_bpc"])["test_bpc"]
    rungs["zenn"].append(("Sleep LM\n(pure rule)", best_sleep))
    best_ng = min(ngram["ladder"].values(), key=lambda v: v["valid"])      # valid で選んだ n の test
    rungs["zenn"].append(("count n-gram\n(best)", best_ng["test"]))
z = best_neural("neural_zenn.json")
z = z[1] if z else None
if z:
    rungs["zenn"].append(("tiny GPT\n0.9M", z))
# Aozora
lad = jload("ladder_aozora.json")
if lad:
    full = lad[max(lad, key=int)]
    rungs["aozora"].append(("uniform", math.log2(full["vocab"])))
    rungs["aozora"].append(("unigram", full["unigram"]))
    sa = jload("sleep_aozora.json")
    if sa:
        rungs["aozora"].append(("Sleep LM\n(pure rule)", sa["winner"]["test"]))
    lv = jload("ladder_aozora_valid.json")                                  # valid で選んだ n の test
    n_best = min((k for k in lv if k.endswith("gram") and k != "unigram" and k in full), key=lambda k: lv[k])
    rungs["aozora"].append(("count n-gram\n(best)", full[n_best]))
a = min(x for x in (best_neural("neural_aozora_iid.json"), best_neural("neural_aozora_iid_12k.json"),
                     best_neural("neural_aozora_iid_24k.json"),
                     best_neural("neural_aozora_iid_48k.json")) if x)[1]   # valid 最良の版の test
rungs["aozora"].append(("tiny GPT\n5.8M", a))
for name in ("zenn", "aozora"):
    gc = jload(f"gpt_cache_{name}_48k.json") or jload(f"gpt_cache_{name}.json")
    if gc:
        rungs[name].append(("tiny GPT\n+cache", gc["test"]["gpt_cache"]))
    mx = jload(f"mix_all_{name}.json")
    if mx:
        rungs[name].append(("mix\n(all parts)", mx["test"]["mix"]))
    md = jload(f"mix_all_{name}_dyn.json")
    if md:
        rungs[name].append(("mix +\ndynamic\neval", md["test"]["mix"]))
    ms = jload(f"mix_all_{name}_sleep_dyn.json")
    if ms:
        rungs[name].append(("mix +\ndyn eval\n+ sleep", ms["test"]["mix"]))
llm = jload("llm_Qwen2.5-0.5B.json")
if llm:
    rungs["zenn"].append(("real LLM\n(Qwen 0.5B)", llm["zenn"]["bpc"]))
    rungs["aozora"].append(("real LLM\n(Qwen 0.5B)", llm["aozora"]["bpc"]))

fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True)
for ax, (name, rs) in zip(axes, rungs.items()):
    labels, vals = zip(*rs)
    colors = ["#bbb"] * len(rs)
    for i, l in enumerate(labels):
        if "Sleep" in l:
            colors[i] = "#7c5cff"
        elif "sleep" in l and "mix" in l:
            colors[i] = "#7c5cff"
        elif "dynamic" in l:
            colors[i] = "#c2185b"
        elif "mix" in l:
            colors[i] = "#0b6e50"
        elif "cache" in l:
            colors[i] = "#0b6e50"
        elif "GPT" in l:
            colors[i] = "#16a37a"
        elif "LLM" in l:
            colors[i] = "#e08a00"
    ax.bar(range(len(rs)), vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.1, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_xticks(range(len(rs)), [l.replace("count n-gram", "n-gram") for l in labels], fontsize=8)
    ax.set_title({"zenn": "Zenn articles (71K chars of training text)",
                  "aozora": "Aozora Bunko novels (4.2M chars)"}[name])
axes[0].set_ylabel("held-out bits per char (lower = better)")
fig.suptitle("How far each tiny LM climbs the ladder")
fig.tight_layout()
fig.savefig(R / "ladder.png", dpi=120)
print("saved", R / "ladder.png")
