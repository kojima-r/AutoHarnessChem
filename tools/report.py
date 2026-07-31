"""ユーザ向け HTML レポートの生成（Markdown → HTML + 構造式の可視化）。

SmilesDrawer 2.x（`app/web/vendor/smiles-drawer.min.js`、MIT）を **インライン埋め込み**
するので、生成された `report_user.html` はネットワークなしでも構造が描画される。

Markdown 側の記法:

  ```smiles
  CCO                             エタノール
  CC(=O)Nc1ccc(O)cc1              パラセタモール
  CC(=O)OC(C)=O.Nc1ccc(O)cc1>>CC(=O)Nc1ccc(O)cc1   アセチル化（`>` を含めば反応式）
  ```

  文中では `` `smiles:CCO` `` と書くと小さな構造チップになる。

さらに `auto_structures=True`（既定）なら、workspace の CSV（orbital_features.csv /
tddft_spectrum.csv / reactiont5_predictions.csv / retrosynthesis_routes.csv 等）から
SMILES 列を拾って「構造一覧」セクションを自動で追加する。
"""
from __future__ import annotations

import csv
import html
import re
from pathlib import Path

from schemas import Artifact, ToolResult

VENDOR_JS = Path(__file__).resolve().parent.parent / "app" / "web" / "vendor" / "smiles-drawer.min.js"
STATIC_JS_URL = "/static/smiles-drawer.min.js"

# SMILES 列とみなす CSV 列名（小文字化して部分一致）。SMILES でない値が混じっても
# is_drawable_smiles で弾かれるため、やや広めに取る
SMILES_COLUMN_HINTS = ("smiles", "precursor", "prediction", "candidate", "product",
                       "reactant", "target", "input", "reaction")
# 構造として描画してよい文字だけを許可する（属性値に空白は入れない）
_SMILES_SAFE = re.compile(r"^[A-Za-z0-9@+\-\[\]\(\)=#$:/\\%.*>~{}]+$")
MAX_AUTO_STRUCTURES = 24

_CSS = """
body { font-family: system-ui, "Hiragino Sans", "Noto Sans JP", sans-serif;
       max-width: 900px; margin: 2rem auto; padding: 0 1rem;
       line-height: 1.7; color: #1c2733; }
h1, h2, h3 { border-bottom: 1px solid #ddd; padding-bottom: .2em; }
img { max-width: 100%; height: auto; border: 1px solid #eee; }
code { background: #f4f4f4; padding: .1em .3em; border-radius: 3px; }
pre code { display: block; padding: .8em; overflow-x: auto; }
table { border-collapse: collapse; }
th, td { border: 1px solid #ccc; padding: .3em .6em; }
blockquote { color: #555; border-left: 4px solid #ccc; margin-left: 0; padding-left: 1em; }
.structures { display: flex; flex-wrap: wrap; gap: .8rem; margin: 1rem 0; padding: 0;
              list-style: none; }
.structures li { border: 1px solid #dfe3e8; border-radius: 8px; padding: .5rem;
                 background: #fff; max-width: 100%; }
.structures svg { display: block; width: 260px; height: 200px; }
.structures li.reaction svg { width: 520px; }
.structures .label { font-size: .8rem; font-weight: 600; margin-top: .2rem; }
.structures .smiles { font-family: ui-monospace, Menlo, Consolas, monospace;
                      font-size: .7rem; color: #66727e; word-break: break-all;
                      max-width: 520px; }
.smiles-chip { display: inline-flex; align-items: center; gap: .25rem;
               vertical-align: middle; border: 1px solid #dfe3e8; border-radius: 6px;
               padding: .1rem .3rem; background: #fff; }
.smiles-chip svg { width: 86px; height: 56px; display: block; }
.smiles-chip code { background: none; font-size: .72rem; }
.structure-error { color: #b91c1c; font-size: .72rem; }
"""

_RENDER_JS = """
document.addEventListener("DOMContentLoaded", function () {
  if (typeof SmiDrawer === "undefined") {   // ライブラリが無ければ SMILES 文字列のまま残す
    document.querySelectorAll("[data-smiles]").forEach(function (el) { el.remove(); });
    return;
  }
  var drawer = new SmiDrawer({ padding: 4.0 }, { padding: 4.0 });
  document.querySelectorAll("svg[data-smiles]").forEach(function (el) {
    drawer.draw(el.getAttribute("data-smiles"), el, "light", null, function (error) {
      el.insertAdjacentHTML("afterend",
        '<div class="structure-error">構造を描画できませんでした（SMILES を確認してください）</div>');
      el.remove();
    });
  });
});
"""


# ---------------------------------------------------------------------------
# Markdown → HTML
# ---------------------------------------------------------------------------

def convert_markdown(text: str) -> str:
    """必要十分な簡易 Markdown → HTML 変換（```smiles ブロックを構造グリッドにする）。"""
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

        if line.startswith("```"):
            close_list()
            language = line[3:].strip().lower()
            block: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                block.append(lines[i])
                i += 1
            i += 1
            if language in ("smiles", "smi", "reaction"):
                out.append(render_structure_grid(_parse_smiles_block(block)))
            else:
                out.append("<pre><code>" + html.escape("\n".join(block)) + "</code></pre>")
            continue

        if "|" in line and i + 1 < len(lines) and re.match(r"^\s*\|?[\s:|-]+\|?\s*$",
                                                           lines[i + 1]):
            close_list()  # table
            header = [c.strip() for c in line.strip().strip("|").split("|")]
            out.append("<table><tr>" + "".join(f"<th>{_inline(c)}</th>" for c in header)
                       + "</tr>")
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
    # `smiles:<SMILES>` は構造チップに、それ以外の `code` は通常のコード表記に
    text = re.sub(r"`smiles:([^`\s]+)`", lambda m: render_structure_chip(m.group(1)), text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    return text


# ---------------------------------------------------------------------------
# 構造式の描画（SmilesDrawer 用のマークアップ生成）
# ---------------------------------------------------------------------------

def is_drawable_smiles(smiles: str) -> bool:
    """描画を試す価値があるか（構文の厳密な検証はブラウザ側の parser に任せる）。"""
    smiles = (smiles or "").strip()
    if not (1 <= len(smiles) <= 400) or not _SMILES_SAFE.match(smiles):
        return False
    if not re.search(r"[A-Za-z]", smiles):
        return False
    # 拡張子付きのファイル名や数値だけの列を弾く
    if re.search(r"\.(csv|json|png|md|html|xyz|log|db|xlsx|pptx)$", smiles, re.IGNORECASE):
        return False
    return True


def _parse_smiles_block(lines: list[str]) -> list[tuple[str, str]]:
    """```smiles ブロックを (smiles, ラベル) の列にする（1行目の空白以降がラベル）。"""
    entries: list[tuple[str, str]] = []
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        smiles, _, label = line.partition(" ")
        entries.append((smiles.strip(), label.strip()))
    return entries


def render_structure_grid(entries: list[tuple[str, str]]) -> str:
    """(smiles, label) の列を構造カードのグリッドにする。"""
    items = []
    for smiles, label in entries:
        if not is_drawable_smiles(smiles):
            # 描画できない文字列はそのまま見せる（黙って消さない）
            items.append(f'<li><div class="label">{html.escape(label)}</div>'
                         f'<div class="smiles">{html.escape(smiles)}</div></li>')
            continue
        kind = "reaction" if ">" in smiles else "molecule"
        items.append(
            f'<li class="{kind}">'
            f'<svg data-smiles="{html.escape(smiles, quote=True)}"></svg>'
            + (f'<div class="label">{html.escape(label)}</div>' if label else "")
            + f'<div class="smiles">{html.escape(smiles)}</div></li>'
        )
    if not items:
        return ""
    return '<ul class="structures">' + "".join(items) + "</ul>"


def count_structures(html_fragment: str) -> int:
    """HTML 断片に含まれる描画対象（data-smiles 属性）の数。"""
    return len(re.findall(r'data-smiles="', html_fragment))


def render_structure_chip(smiles: str) -> str:
    """文中に埋める小さな構造チップ。"""
    if not is_drawable_smiles(smiles):
        return f"<code>{html.escape(smiles)}</code>"
    return ('<span class="smiles-chip">'
            f'<svg data-smiles="{html.escape(smiles, quote=True)}"></svg>'
            f"<code>{html.escape(smiles)}</code></span>")


# ---------------------------------------------------------------------------
# workspace の CSV から SMILES を拾う（レポートに構造一覧を自動追加する）
# ---------------------------------------------------------------------------

def collect_csv_smiles(workspace: Path,
                       limit: int = MAX_AUTO_STRUCTURES) -> list[tuple[str, str]]:
    """workspace 直下の CSV から (smiles, ラベル) を重複なく集める。"""
    found: dict[str, str] = {}
    for path in sorted(Path(workspace).glob("*.csv")):
        try:
            with path.open(encoding="utf-8", newline="") as fp:
                rows = list(csv.DictReader(fp))
        except (OSError, UnicodeDecodeError, csv.Error):
            continue
        if not rows:
            continue
        columns = [c for c in (rows[0].keys() or ())
                   if c and any(hint in c.lower() for hint in SMILES_COLUMN_HINTS)]
        for row in rows:
            for column in columns:
                for candidate in _split_cell(row.get(column) or ""):
                    if candidate in found or not is_drawable_smiles(candidate):
                        continue
                    found[candidate] = f"{path.name}: {column}"
                    if len(found) >= limit:
                        return list(found.items())
    return list(found.items())


def _split_cell(value: str) -> list[str]:
    """1セルに複数 SMILES が入る形式（`|` 区切りの候補、`REACTANT:...` 形式）を分解する。"""
    value = (value or "").strip()
    if not value:
        return []
    parts: list[str] = []
    for chunk in value.split("|"):
        chunk = chunk.strip()
        if not chunk:
            continue
        # ReactionT5 の入力形式は接頭辞を落として成分ごとに見る
        if re.search(r"(REACTANT|REAGENT|PRODUCT):", chunk):
            for piece in re.split(r"(?:REACTANT|REAGENT|PRODUCT):", chunk):
                piece = piece.strip()
                if piece:
                    parts.append(piece)
        else:
            parts.append(chunk)
    return [p for p in parts if p]


# ---------------------------------------------------------------------------
# HTML 文書の組み立て
# ---------------------------------------------------------------------------

def library_script(inline_library: bool = True) -> str:
    """SmilesDrawer を読み込む <script> タグ。

    inline_library=True なら同梱ファイルを埋め込み、オフラインでも描画できる
    自己完結 HTML にする。False なら Web UI 経由の配信 URL を参照する。
    """
    if inline_library and VENDOR_JS.exists():
        return "<script>\n" + VENDOR_JS.read_text(encoding="utf-8") + "\n</script>"
    return f'<script src="{STATIC_JS_URL}"></script>'


def build_html(markdown_text: str, *, title: str | None = None,
               auto_structures: list[tuple[str, str]] | None = None,
               inline_library: bool = True) -> str:
    """Markdown 本文（+ 自動抽出した構造）から完成した HTML 文書を返す。"""
    body = convert_markdown(markdown_text)
    if auto_structures:
        body += ("\n<h2>構造一覧（成果物 CSV から自動抽出）</h2>\n"
                 + render_structure_grid(auto_structures))
    if title is None:
        heading = re.search(r"^#\s+(.+)$", markdown_text, re.MULTILINE)
        title = heading.group(1).strip() if heading else "Report"
    return (
        "<!DOCTYPE html>\n"
        '<html lang="ja">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{html.escape(title)}</title>\n"
        f"<style>{_CSS}</style>\n"
        f"{library_script(inline_library)}\n"
        f"<script>{_RENDER_JS}</script>\n"
        "</head>\n<body>\n"
        f"{body}\n</body>\n</html>\n"
    )


# ---------------------------------------------------------------------------
# ツール: render_report_html
# ---------------------------------------------------------------------------

def render_report_html(
    workspace: Path,
    markdown_path: str = "report_user.md",
    output_html: str = "report_user.html",
    auto_structures: bool = True,
    inline_library: bool = True,
) -> ToolResult:
    """report_user.md を構造式付きの HTML レポートへ変換する。"""
    workspace = Path(workspace)
    source = workspace / markdown_path
    if not source.is_file():
        return ToolResult(
            status="failed",
            summary=f"Markdown が見つかりません: {markdown_path}"
                    "（先に報告本文を workspace 直下へ保存してください）",
            retryable=False, error_type="input_not_found",
        )
    text = source.read_text(encoding="utf-8")
    structures = collect_csv_smiles(workspace) if auto_structures else []
    n_inline = count_structures(convert_markdown(text))
    document = build_html(text, auto_structures=structures,
                          inline_library=inline_library)
    target = workspace / output_html
    target.write_text(document, encoding="utf-8")

    library = "インライン埋め込み" if inline_library and VENDOR_JS.exists() else STATIC_JS_URL
    return ToolResult(
        status="success",
        summary=(f"{output_html} を生成しました（{target.stat().st_size} bytes、"
                 f"本文中の構造 {n_inline} 件 + CSV から自動抽出 {len(structures)} 件、"
                 f"SmilesDrawer: {library}）"),
        data={"output_html": str(target), "markdown_path": str(source),
              "n_structures_in_body": n_inline,
              "auto_structures": [s for s, _ in structures],
              "inline_library": bool(inline_library and VENDOR_JS.exists())},
        artifacts=[Artifact(path=str(target), mime="text/html",
                            bytes=target.stat().st_size, kind="report")],
    )
