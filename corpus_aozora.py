"""大きい教材: 青空文庫（夏目漱石・宮沢賢治・芥川竜之介、新字新仮名、著作権なし の 335 作品）。

取得は data/aozora/raw/<作品ID>.txt（1行目にメタ JSON、以降は本文。公式インデックスCSVから選定）。
ここではルビ・注記・冒頭の記号説明・末尾の底本情報を落として本文だけにし、作品単位で分割する
（作品IDを並べて 10本に1本を test、10本に1本を valid）。出力は data/aozora/{train,valid,test}.txt。
--limit で学習用を先頭から N 文字に切り詰めた版も作れる（教材の量を変えて成長を測るため）。
"""
import argparse
import json
import pathlib
import re

D = pathlib.Path(__file__).resolve().parent / "data" / "aozora"


def clean(text):
    lines = text.split("\n")
    # 冒頭: 題名・著者 → 「----」で囲まれた記号説明 → 本文
    seps = [i for i, ln in enumerate(lines[:80]) if re.fullmatch(r"-{10,}\r?", ln)]
    if len(seps) >= 2:
        lines = lines[seps[1] + 1:]
    body = "\n".join(lines)
    body = re.split(r"\n底本[：:]", body)[0]           # 末尾の底本情報
    body = re.sub(r"《[^》]*》", "", body)              # ルビ
    body = body.replace("｜", "")                       # ルビの範囲指定
    body = re.sub(r"［＃[^］]*］", "", body)            # 注記
    body = body.replace("\r", "").replace("　", " ")
    body = re.sub(r"\n{2,}", "\n", body)
    return body.strip() + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="学習用を先頭から N 文字に（0=全部）")
    a = ap.parse_args()
    works = sorted(D.glob("raw/*.txt"), key=lambda p: int(p.stem))
    split = {"train": [], "valid": [], "test": []}
    for i, p in enumerate(works):
        meta_line, _, text = p.read_text(encoding="utf-8").partition("\n")
        key = "test" if i % 10 == 9 else "valid" if i % 10 == 8 else "train"
        split[key].append(clean(text))
    out = {}
    for k, parts in split.items():
        s = "".join(parts)
        if k == "train" and a.limit:
            s = s[:a.limit]
        out[k] = s
        suffix = f"_{a.limit}" if (k == "train" and a.limit) else ""
        (D / f"{k}{suffix}.txt").write_text(s, encoding="utf-8")
        print(f"{k}{suffix}: {len(parts)} 作品 / {len(s):,} 文字")


if __name__ == "__main__":
    main()
