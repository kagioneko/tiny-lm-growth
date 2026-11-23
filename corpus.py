"""育成用コーパス: ねこさんの Zenn 記事（ローカルの ~/zenn/articles/*.md）。

- frontmatter・コードブロック・HTMLタグ・URL・Markdown記号を落として本文だけにする
- 記事単位で分割（ファイル名順に並べ、5本に1本を test、さらに残りの10本に1本を valid）
  → test は育成中に一度も読まない記事
- 文字単位で扱う（日本語の単語分割器に依存しないため。LLMの bits/char と同じ物差し）
- 出力は data/（.gitignore 済み。記事本文はリポジトリに入れない）
"""
import json
import os
import pathlib
import re
import sys

SRC = pathlib.Path(os.environ.get("ZENN_ARTICLES", pathlib.Path.home() / "zenn" / "articles"))
OUT = pathlib.Path(__file__).resolve().parent / "data"


def clean(md: str) -> str:
    md = re.sub(r"\A---\n.*?\n---\n", "", md, flags=re.S)           # frontmatter
    md = re.sub(r"```.*?```", "\n", md, flags=re.S)                  # コードブロック
    md = re.sub(r":::.*?:::", "\n", md, flags=re.S)                  # Zenn のメッセージ枠
    md = re.sub(r"`[^`\n]*`", "", md)                                # インラインコード
    md = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", md)                     # 画像
    md = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", md)                 # リンク → 文字だけ
    md = re.sub(r"https?://\S+", "", md)                             # 生URL
    md = re.sub(r"<[^>]+>", "", md)                                  # HTMLタグ
    md = re.sub(r"^\s*(#{1,6}|[-*+]|\d+\.|>|\|)\s*", "", md, flags=re.M)  # 見出し・箇条書き・引用・表
    md = re.sub(r"[*_~|]", "", md)                                   # 強調・表の罫線
    md = re.sub(r"[ \t　]+", " ", md)
    lines = [ln.strip() for ln in md.splitlines()]
    lines = [ln for ln in lines if ln and not re.fullmatch(r"[-:= ]+", ln)]
    return "\n".join(lines) + "\n"


def main():
    files = sorted(SRC.glob("*.md"))
    if not files:
        sys.exit(f"記事が見つかりません: {SRC}")
    split = {"train": [], "valid": [], "test": []}
    for i, f in enumerate(files):
        if i % 5 == 4:
            split["test"].append(f)
        elif i % 10 == 3:
            split["valid"].append(f)
        else:
            split["train"].append(f)
    OUT.mkdir(exist_ok=True)
    meta = {}
    for name, fs in split.items():
        text = "".join(clean(f.read_text(encoding="utf-8")) for f in fs)
        (OUT / f"{name}.txt").write_text(text, encoding="utf-8")
        meta[name] = {"articles": len(fs), "chars": len(text), "files": [f.name for f in fs]}
        print(f"{name}: {len(fs)} 記事 / {len(text):,} 文字")
    (OUT / "split.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
