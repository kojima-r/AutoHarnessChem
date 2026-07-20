"""report_user.md → report_user.html 変換（依存パッケージ不要）。

`markdown` パッケージがあればそれを使い、無ければ正規表現ベースの
簡易コンバータで見出し・リスト・画像・リンク・コードブロック・表を変換する。
sandbox 内で実行する前提（カレントディレクトリ = workspace）。
"""
from __future__ import annotations

import html
import re
from pathlib import Path

INPUT_MD = "report_user.md"
OUTPUT_HTML = "report_user.html"

_CSS = """
body { font-family: sans-serif; max-width: 860px; margin: 2rem auto; padding: 0 1rem;
       line-height: 1.7; color: #222; }
h1, h2, h3 { border-bottom: 1px solid #ddd; padding-bottom: .2em; }
img { max-width: 100%; height: auto; border: 1px solid #eee; }
code { background: #f4f4f4; padding: .1em .3em; border-radius: 3px; }
pre code { display: block; padding: .8em; overflow-x: auto; }
table { border-collapse: collapse; }
th, td { border: 1px solid #ccc; padding: .3em .6em; }
blockquote { color: #555; border-left: 4px solid #ccc; margin-left: 0; padding-left: 1em; }
"""


def convert_with_package(text: str) -> str | None:
    try:
        import markdown
    except ImportError:
        return None
    return markdown.markdown(text, extensions=["tables", "fenced_code"])


def convert_minimal(text: str) -> str:
    """必要十分な簡易 Markdown → HTML 変換。"""
    out: list[str] = []
    lines = text.splitlines()
    i = 0
    in_list = False

    def close_list():
        nonlocal in_list
        if in_list:
            out.append("</ul>")
            in_list = False

    while i < len(lines):
        line = lines[i]

        if line.startswith("```"):  # fenced code block
            close_list()
            block: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                block.append(lines[i])
                i += 1
            out.append("<pre><code>" + html.escape("\n".join(block)) + "</code></pre>")
            i += 1
            continue

        if "|" in line and i + 1 < len(lines) and re.match(r"^\s*\|?[\s:|-]+\|?\s*$", lines[i + 1]):
            close_list()  # table
            header = [c.strip() for c in line.strip().strip("|").split("|")]
            out.append("<table><tr>" + "".join(f"<th>{_inline(c)}</th>" for c in header) + "</tr>")
            i += 2
            while i < len(lines) and "|" in lines[i]:
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                out.append("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in cells) + "</tr>")
                i += 1
            out.append("</table>")
            continue

        heading = re.match(r"^(#{1,6})\s+(.*)$", line)
        if heading:
            close_list()
            level = len(heading.group(1))
            out.append(f"<h{level}>{_inline(heading.group(2))}</h{level}>")
        elif re.match(r"^\s*[-*]\s+", line):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append("<li>" + _inline(re.sub(r"^\s*[-*]\s+", "", line)) + "</li>")
        elif line.startswith(">"):
            close_list()
            out.append("<blockquote>" + _inline(line.lstrip("> ")) + "</blockquote>")
        elif not line.strip():
            close_list()
        else:
            close_list()
            out.append("<p>" + _inline(line) + "</p>")
        i += 1

    close_list()
    return "\n".join(out)


def _inline(text: str) -> str:
    text = html.escape(text, quote=False)
    text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r'<img src="\2" alt="\1">', text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    return text


def main() -> None:
    text = Path(INPUT_MD).read_text(encoding="utf-8")
    body = convert_with_package(text) or convert_minimal(text)
    title_match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    title = html.escape(title_match.group(1)) if title_match else "Report"
    document = (
        "<!DOCTYPE html>\n"
        '<html lang="ja">\n<head>\n<meta charset="utf-8">\n'
        f"<title>{title}</title>\n<style>{_CSS}</style>\n</head>\n<body>\n"
        f"{body}\n</body>\n</html>\n"
    )
    Path(OUTPUT_HTML).write_text(document, encoding="utf-8")
    print(f"wrote {OUTPUT_HTML} ({len(document)} bytes)")


if __name__ == "__main__":
    main()
