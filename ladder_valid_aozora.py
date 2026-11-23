"""青空文庫の数え上げ n-gram のはしごを valid で測る（plot_ladder の n-gram 段で n を valid で選ぶため）。"""
import json, os, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
os.environ["LM_CORPUS"] = "aozora"
from ladder_scale import ladder
from sleep_growth import build_vocab, encode, load
t = load(); itos, stoi = build_vocab(t["train"])
r = ladder(encode(t["train"], stoi), encode(t["valid"], stoi), len(itos), max_order=5)
pathlib.Path("results/ladder_aozora_valid.json").write_text(json.dumps({"note": "ladder_scale.ladder を valid 全文に（全量の train）。plot_ladder の n-gram 段の選択用", **r}, ensure_ascii=False, indent=1))
print(r)
