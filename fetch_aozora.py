"""青空文庫の教材を取得する（corpus_aozora.py の前段）。

aozora_selection.json（作品ID・作者・題名・公式のテキスト zip の URL。335作品、著作権の切れた新字新仮名のみ）を読み、
各 zip の本文（Shift_JIS）を UTF-8・改行 \\n にして data/aozora/raw/<作品ID>.txt に保存する（1行目は {"author","title"} の JSON）。
青空文庫のサーバーに負担をかけないよう、1作品ごとに --interval 秒（既定1秒）待つ。取得済みの作品は飛ばす。
  python fetch_aozora.py && python corpus_aozora.py
"""
import argparse
import io
import json
import pathlib
import time
import urllib.request
import zipfile

D = pathlib.Path(__file__).resolve().parent / "data" / "aozora"
UA = "tiny-lm-growth (research script)"


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    name = next(n for n in z.namelist() if n.lower().endswith(".txt"))
    return z.read(name).decode("cp932").replace("\r\n", "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=0, help="先頭 N 作品だけ（動作確認用）")
    a = ap.parse_args()
    sel = json.loads((pathlib.Path(__file__).resolve().parent / "aozora_selection.json").read_text(encoding="utf-8"))
    if a.limit:
        sel = sel[:a.limit]
    (D / "raw").mkdir(parents=True, exist_ok=True)
    for i, (wid, (author, title, url)) in enumerate(sel):
        out = D / "raw" / f"{wid}.txt"
        if out.exists():
            continue
        meta = json.dumps({"author": author, "title": title}, ensure_ascii=False)
        out.write_text(meta + "\n" + fetch(url), encoding="utf-8")
        print(f"[{i + 1}/{len(sel)}] {author}「{title}」", flush=True)
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
