"""ブラウザ側の構造式描画を実際に走らせて確認する（jsdom があるときだけ実行）。

静的な文字列検査（tests/test_report.py）では「SmilesDrawer が本当に描けるか」
「プロンプト内の SMILES を壊さずに拾えるか」は分からないため、node + jsdom で
DOM を動かして検証する。有効化するには:

    npm install --no-save jsdom      # リポジトリ直下（node_modules/ は git 管理外）

jsdom が見つからない環境では skip される（pytest 全体は node に依存しない）。
"""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from tools.report import render_report_html

ROOT = Path(__file__).resolve().parent.parent
CHECKER = Path(__file__).parent / "js" / "check_structures.js"
INDEX_HTML = ROOT / "app" / "web" / "index.html"

MOLECULE = "CC(=O)Nc1ccc(O)cc1"                                  # パラセタモール
REQUEST = f"パラセタモール (CC(=O)Nc1ccc(O)cc1) の逆合成経路を提案してください"
REACTION = "CC(=O)OC(C)=O.Nc1ccc(O)cc1>>CC(=O)Nc1ccc(O)cc1"      # アセチル化

REPORT_MD = f"""# 構造可視化の確認

## 結論

`smiles:{MOLECULE}` の逆合成は 1 段のアセチル化。

## 構造

```smiles
{MOLECULE}   パラセタモール
{REACTION}   アセチル化
c1ccccc1     ベンゼン
```
"""

RETRO_CSV = (
    "target_smiles,route_rank,solved,n_steps,n_precursors,score,precursors,search_time_s\n"
    f"{MOLECULE},1,True,1,2,0.99,CC(=O)OC(C)=O.Nc1ccc(O)cc1,2.8\n"
)
ORBITAL_CSV = "smiles,homo_ev,lumo_ev\nc1ccccc1,-7.58,7.28\nCCO,-11.0,5.0\n"


def _node_with_jsdom() -> str | None:
    node = shutil.which("node")
    if node is None:
        return None
    probe = subprocess.run([node, "-e", "require.resolve('jsdom')"],
                           cwd=ROOT, capture_output=True, text=True)
    return node if probe.returncode == 0 else None


@pytest.fixture(scope="module")
def checked(tmp_path_factory):
    node = _node_with_jsdom()
    if node is None:
        pytest.skip("node + jsdom が無い（`npm install --no-save jsdom` で有効化）")

    workspace = tmp_path_factory.mktemp("ws")
    (workspace / "report_user.md").write_text(REPORT_MD, encoding="utf-8")
    (workspace / "retrosynthesis_routes.csv").write_text(RETRO_CSV, encoding="utf-8")
    (workspace / "orbital_features.csv").write_text(ORBITAL_CSV, encoding="utf-8")
    assert render_report_html(workspace).status == "success"

    run_id = "run-jsdom-check"
    artifacts = [{"path": f"/ws/{run_id}/{name}", "mime": mime, "bytes": 100}
                 for name, mime in (("retrosynthesis_routes.csv", "text/csv"),
                                    ("orbital_features.csv", "text/csv"),
                                    ("report_user.html", "text/html"))]
    fixtures = {
        "run_id": run_id,
        "molecule": MOLECULE,
        "reaction": REACTION,
        "template_label": "逆合成",
        "preset_label": "カフェイン",
        "providers": {"available": {"deepagents": True, "claude": False, "openai": False},
                      "routing": {}, "default": "deepagents"},
        "run_summary": {"run_id": run_id, "status": "succeeded", "provider": "claude",
                        "task_type": "retrosynthesis_planning", "description": "check"},
        "run_detail": {"run_id": run_id, "status": "succeeded", "provider": "claude",
                       "artifacts": artifacts,
                       "report_md": "# Run report: " + run_id,
                       "report": {"attempts": 1, "artifacts": artifacts,
                                  "final_message": "完了しました",
                                  "task": {"description": REQUEST,
                                           "task_type": "retrosynthesis_planning",
                                           "secondary_task_types": [],
                                           "inputs": {"files": ["data_smi.csv"]},
                                           "expected_outputs": ["retrosynthesis_routes.json"]},
                                  "verification": {"passed": True,
                                                   "requirements_satisfied": ["ok"],
                                                   "requirements_missing": [],
                                                   "scientific_warnings": []}}},
        "events": [{"run_id": run_id, "event_type": "tool_call", "actor": "claude",
                    "payload": {"tool": "plan_retrosynthesis",
                                "arguments": {"targets": [MOLECULE]}},
                    "timestamp": "2026-07-25T12:00:00+00:00"},
                   # 失敗イベント（stderr の末尾 + エラーログの場所を trace に載せる）
                   {"run_id": run_id, "event_type": "tool_result", "actor": "claude",
                    "payload": {"tool": "predict_reaction_t5", "status": "failed",
                                "summary": "CUDA のメモリ確保に失敗しました。",
                                "error_type": "out_of_memory",
                                "stderr_tail": "line1\ntorch.AcceleratorError: "
                                               "CUDA error: out of memory",
                                "error_log": "tool_errors.jsonl"},
                    "timestamp": "2026-07-25T12:00:05+00:00"}],
        "csvs": {"retrosynthesis_routes.csv": RETRO_CSV,
                 "orbital_features.csv": ORBITAL_CSV},
    }
    fixtures_path = workspace / "fixtures.json"
    fixtures_path.write_text(json.dumps(fixtures, ensure_ascii=False), encoding="utf-8")

    completed = subprocess.run(
        [node, str(CHECKER), str(INDEX_HTML), str(workspace / "report_user.html"),
         str(fixtures_path)],
        cwd=ROOT, capture_output=True, text=True, timeout=180)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return json.loads(completed.stdout)


# --- HTML レポート ----------------------------------------------------------

def test_report_draws_every_structure(checked):
    report = checked["report"]
    assert report["library_loaded"], "埋め込んだ SmilesDrawer が読み込めていない"
    assert report["total"] >= 4                       # 本文3件 + CSV 自動抽出
    assert report["drawn"] == report["total"]         # すべて描画された
    assert report["render_errors"] == 0
    # 結合線と原子ラベルが実際に生成されている（空の svg ではない）
    assert report["bonds"] > 20 and report["atom_labels"] > 5
    assert report["jsdom_errors"] == []


def test_report_draws_reaction_smiles(checked):
    report = checked["report"]
    assert report["reactions"] >= 1
    assert report["reactions_drawn"] == report["reactions"]


# --- Web UI（プロンプト補助） ------------------------------------------------

def test_ui_previews_molecule_and_reaction(checked):
    ui = checked["ui"]
    assert ui["molecule"]["drawn"] > 0 and "OK" in ui["molecule"]["status"]
    assert "分子" in ui["molecule"]["status"]
    assert ui["reaction"]["elements"] > 10
    assert "反応式" in ui["reaction"]["status"]
    assert ui["jsdom_errors"] == []


def test_ui_template_fills_prompt_and_task_type(checked):
    template = checked["ui"]["template"]
    assert MOLECULE in template["request"] and "逆合成" in template["request"]
    assert template["task_type"] == "retrosynthesis_planning"
    assert "retrosynthesis_routes.json" in template["expect"]


def test_ui_detects_smiles_in_prompt_without_breaking_parentheses(checked):
    """`CC(=O)Nc1ccc(O)cc1` を括弧で分断せず 1 件として検出すること。"""
    assert checked["ui"]["template"]["detected"] == [MOLECULE]


def test_ui_preset_sets_valid_smiles(checked):
    preset = checked["ui"]["preset"]
    assert preset["smiles"].startswith("Cn1cnc2")      # カフェイン
    assert "OK" in preset["status"]


def test_ui_structure_card_renders_reactions_from_csv(checked):
    card = checked["ui"]["card"]
    assert card["hidden"] is False
    assert card["count"] >= 3
    assert card["drawn"] == card["count"]
    # 逆合成 CSV は「前駆体 >> 目標分子」の反応式として描かれる
    assert card["reactions"] >= 1
    assert any("経路" in label for label in card["labels"])


def test_ui_detail_shows_request_and_user_report_by_default(checked):
    """ラン詳細では「ユーザからの入力」と「ユーザ向け報告」が最初から見えている。"""
    layout = checked["ui"]["layout"]
    assert "ユーザからの入力" in layout["headings"]
    assert "ユーザ向け報告" in layout["headings"]
    assert REQUEST in layout["request"]
    assert "retrosynthesis_planning" in layout["meta"]      # task_type などの付随情報
    assert "data_smi.csv" in layout["meta"]                  # 入力ファイル
    assert layout["reportFrame"] is True                     # report_user.html を埋め込み表示


def test_ui_detail_other_sections_are_collapsed_by_default(checked):
    """検証結果・構造・trace・成果物・report.md は既定で閉じたトグルになっている。"""
    sections = {s["key"]: s for s in checked["ui"]["layout"]["sections"]}
    assert {"verification", "structures", "trace", "artifacts", "report_md"} <= set(sections)
    assert [key for key, s in sections.items() if s["open"]] == []
    # 構造は中身があるときだけ表示される（描画自体は閉じたままでも行われる）
    assert sections["structures"]["hidden"] is False


def test_ui_detail_toggle_state_survives_rerender(checked):
    """開いたセクションはポーリングによる再描画で閉じない。"""
    assert checked["ui"]["layout"]["reopened"] == ["trace"]


def test_ui_trace_shows_stderr_tail_and_links_the_error_log(checked):
    """失敗イベントは stderr の末尾を表示し、全文のログへリンクすること。"""
    layout = checked["ui"]["layout"]
    row = next((r for r in layout["traceRows"] if "predict_reaction_t5" in r), "")
    assert "out_of_memory" in row
    assert "stderr: torch.AcceleratorError: CUDA error: out of memory" in row
    assert any(link.endswith("artifacts/tool_errors.jsonl") for link in layout["traceLinks"])
