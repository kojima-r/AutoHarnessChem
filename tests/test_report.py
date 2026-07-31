"""HTML レポート生成（Markdown → HTML + SmilesDrawer による構造描画）のテスト。"""
import re
from pathlib import Path

from tools import report

MARKDOWN = """# ベンゼンの HOMO/LUMO

## 結論

HOMO = -7.58 eV、LUMO = 7.28 eV（`smiles:c1ccccc1` を HF/STO-3G で計算）。

## 構造

```smiles
c1ccccc1                                         ベンゼン
CC(=O)OC(C)=O.Nc1ccc(O)cc1>>CC(=O)Nc1ccc(O)cc1   アセチル化
```

## 成果物

- `orbital_features.csv` — 1分子1行の軌道エネルギー
- ![散布図](true_vs_pred.png)

| 分子 | HOMO (eV) |
|---|---|
| ベンゼン | -7.58 |
"""


def test_smiles_fence_becomes_structure_grid():
    html = report.convert_markdown(MARKDOWN)
    assert 'data-smiles="c1ccccc1"' in html
    # 反応式（> を含む）は reaction クラスで描画する
    assert 'class="reaction"' in html
    assert 'data-smiles="CC(=O)OC(C)=O.Nc1ccc(O)cc1&gt;&gt;CC(=O)Nc1ccc(O)cc1"' in html
    # ラベルと SMILES 文字列も残す（JS が無くても内容が読める）
    assert "ベンゼン" in html and "アセチル化" in html
    # 通常の Markdown 変換は従来どおり
    assert "<h1>" in html and "<table>" in html
    assert '<img src="true_vs_pred.png"' in html
    assert "<code>orbital_features.csv</code>" in html


def test_inline_smiles_chip():
    html = report.convert_markdown("HOMO は `smiles:c1ccccc1` で計算した。")
    assert 'class="smiles-chip"' in html
    assert 'data-smiles="c1ccccc1"' in html
    # 普通のコード表記は構造にしない
    assert "data-smiles" not in report.convert_markdown("`report_user.md` を保存した。")


def test_non_drawable_entries_are_kept_as_text():
    # ファイル名やプレーンな文章を書いてしまっても、消さずにそのまま見せる
    html = report.convert_markdown("```smiles\nreport_user.md 説明文\n```")
    assert "data-smiles" not in html
    assert "report_user.md" in html and "説明文" in html


def test_is_drawable_smiles_rejects_filenames_and_prose():
    assert report.is_drawable_smiles("CC(=O)Nc1ccc(O)cc1")
    assert report.is_drawable_smiles("CCO.O>>CCOC")
    assert not report.is_drawable_smiles("orbital_features.csv")
    assert not report.is_drawable_smiles("HOMO / LUMO")      # 空白は不可
    assert not report.is_drawable_smiles("123")              # 元素記号がない
    assert not report.is_drawable_smiles("")


def test_escaping_is_applied_to_attributes():
    html = report.render_structure_grid([('C"><script>alert(1)</script>', "x")])
    assert "<script>" not in html
    assert "&lt;script&gt;" in html or "&quot;" in html


def test_collect_csv_smiles_from_workspace(tmp_path):
    (tmp_path / "orbital_features.csv").write_text(
        "smiles,homo_ev,lumo_ev\nc1ccccc1,-7.58,7.28\nCCO,-11.0,5.0\n", encoding="utf-8")
    (tmp_path / "reactiont5_predictions.csv").write_text(
        "input,prediction,candidates\n"
        "REACTANT:CC(=O)OC(C)=O.Nc1ccc(O)cc1REAGENT:,CC(=O)Nc1ccc(O)cc1,"
        "CC(=O)Nc1ccc(O)cc1|CCO\n", encoding="utf-8")
    (tmp_path / "cv_metrics.csv").write_text("r2,rmse\n0.4,0.1\n", encoding="utf-8")

    found = dict(report.collect_csv_smiles(tmp_path))
    assert "c1ccccc1" in found and "CCO" in found
    assert "CC(=O)Nc1ccc(O)cc1" in found          # prediction 列から
    # ReactionT5 の input 列は REACTANT:/REAGENT:/PRODUCT: を外して成分ごとに見る
    assert "CC(=O)OC(C)=O.Nc1ccc(O)cc1" in found
    assert all("r2" != s for s in found)          # 数値列は拾わない
    assert found["c1ccccc1"].startswith("orbital_features.csv")


def test_collect_csv_smiles_is_capped(tmp_path):
    rows = "\n".join(f"C{'C' * i}O" for i in range(1, 40))
    (tmp_path / "many.csv").write_text("smiles\n" + rows + "\n", encoding="utf-8")
    assert len(report.collect_csv_smiles(tmp_path)) == report.MAX_AUTO_STRUCTURES


def test_build_html_inlines_the_library():
    html = report.build_html(MARKDOWN)
    assert report.VENDOR_JS.exists(), "SmilesDrawer が app/web/vendor に同梱されていること"
    assert "window.SmiDrawer" in html            # ライブラリ本体が埋め込まれている
    assert report.STATIC_JS_URL not in html      # CDN/配信URLに依存しない
    assert html.startswith("<!DOCTYPE html>")
    assert "<title>ベンゼンの HOMO/LUMO</title>" in html
    assert "new SmiDrawer" in html               # 描画スクリプト


def test_build_html_can_reference_served_library():
    html = report.build_html("# x", inline_library=False)
    assert f'<script src="{report.STATIC_JS_URL}"></script>' in html
    assert "window.SmiDrawer" not in html


def test_render_report_html_tool(tmp_path):
    (tmp_path / "report_user.md").write_text(MARKDOWN, encoding="utf-8")
    (tmp_path / "orbital_features.csv").write_text(
        "smiles,homo_ev,lumo_ev\nc1ccccc1,-7.58,7.28\n", encoding="utf-8")

    result = report.render_report_html(tmp_path)
    assert result.status == "success"
    assert result.data["n_structures_in_body"] == 3     # 反応1 + 分子1 + チップ1
    assert result.data["auto_structures"] == ["c1ccccc1"]
    assert result.data["inline_library"] is True
    assert result.artifacts[0].path.endswith("report_user.html")
    assert result.artifacts[0].kind == "report"

    html = (tmp_path / "report_user.html").read_text(encoding="utf-8")
    assert "構造一覧（成果物 CSV から自動抽出）" in html
    assert html.count('data-smiles="c1ccccc1"') == 3    # 本文2箇所 + 自動抽出1箇所


def test_render_report_html_without_markdown(tmp_path):
    result = report.render_report_html(tmp_path)
    assert result.status == "failed" and result.error_type == "input_not_found"


def test_render_report_html_can_skip_auto_structures(tmp_path):
    (tmp_path / "report_user.md").write_text("# t\n", encoding="utf-8")
    (tmp_path / "orbital_features.csv").write_text("smiles\nCCO\n", encoding="utf-8")
    result = report.render_report_html(tmp_path, auto_structures=False)
    assert result.data["auto_structures"] == []
    # 埋め込んだライブラリ自体は data-smiles を含むので、描画対象の要素で判定する
    assert "<svg data-smiles=" not in (tmp_path / "report_user.html").read_text(encoding="utf-8")


def test_registered_as_tool(tmp_path):
    from harness.policy import PolicyGate
    from schemas import SandboxConfig
    from tools.registry import build_default_registry

    registry = build_default_registry(tmp_path, SandboxConfig(type="local"), PolicyGate())
    assert "render_report_html" in registry.names()

    (tmp_path / "report_user.md").write_text("# t\n\n`smiles:CCO`\n", encoding="utf-8")
    result = registry.call("render_report_html")
    assert result.status == "success"
    assert 'data-smiles="CCO"' in (tmp_path / "report_user.html").read_text(encoding="utf-8")


def test_vendored_library_is_smiles_drawer_2():
    source = report.VENDOR_JS.read_text(encoding="utf-8")
    assert "window.SmiDrawer" in source and "data-smiles" in source
    # 反応式描画（ReactionDrawer）を含む 2.x 系であること
    assert "ReactionDrawer" in source
    license_file = report.VENDOR_JS.parent / "LICENSE-smiles-drawer.md"
    assert "MIT License" in license_file.read_text(encoding="utf-8")


def test_web_ui_uses_the_vendored_library():
    index = (Path(report.VENDOR_JS).parent.parent / "index.html").read_text(encoding="utf-8")
    assert f'<script src="{report.STATIC_JS_URL}"></script>' in index
    assert "SmiDrawer" in index and "data-smiles" in index
    # プロンプト補助（構造プレビュー・テンプレート）の要素があること
    for element in ("smiPreview", "smiInsert", "presets", "templates", "reqStructures",
                    "structureCard"):
        assert f'id="{element}"' in index, element


def test_result_reporting_skill_documents_the_syntax():
    skill = (Path(__file__).resolve().parent.parent / "skills" / "result-reporting"
             / "SKILL.md").read_text(encoding="utf-8")
    assert "render_report_html" in skill
    assert "```smiles" in skill and "smiles:" in skill


def test_no_stale_render_html_script():
    """旧 scripts/render_html.py は render_report_html ツールに置き換わっている。"""
    scripts_dir = (Path(__file__).resolve().parent.parent / "skills" / "result-reporting"
                   / "scripts")
    assert not (scripts_dir / "render_html.py").exists()


def test_markdown_table_rows_with_pipes_and_smiles():
    """表の中の `smiles:` チップも描画対象になる。"""
    html = report.convert_markdown(
        "| 分子 | 構造 |\n|---|---|\n| ベンゼン | `smiles:c1ccccc1` |\n")
    assert "<table>" in html and re.search(r'<td>.*data-smiles="c1ccccc1"', html)
