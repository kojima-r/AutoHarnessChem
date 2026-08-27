"""ChemEval 評価スクリプトのテスト（データのダウンロード・SDK・rdkit を要求しない）。

検証するのは
  - カタログ（tasks.yaml）の整合性と filename の正規化
  - エージェント出力からの答えの取り出し
  - 各 metric の採点（正解 → 満点 / 誤答 → 0 点）
  - スタブ controller での run → 採点 → レポートの一巡
"""
import asyncio
import json
from pathlib import Path

import pytest

from benchmarks_chemeval import dataset, report, runner, score
from benchmarks_chemeval.catalog import LEVELS, load_catalog, normalize_key
from benchmarks_chemeval.extract import ANSWER_FILE, collect_answer, extract_from_text
from benchmarks_chemeval.metrics import METRICS, METRIC_INFO, score_item
from harness.config import load_config
from schemas import RunReport, TaskSpec, VerificationResult


# --- カタログ -----------------------------------------------------------

def test_catalog_is_consistent():
    """id / key の重複が無く、metric・level・task_type が実装と対応している。"""
    from schemas.models import TaskType

    catalog = load_catalog()
    assert len(catalog) == 107                                  # text 53 + multimodal 54
    assert len(catalog.select(splits=["text"])) == 53
    assert len(catalog.select(splits=["multimodal"])) == 54
    valid_task_types = set(TaskType.__args__)
    for task in catalog:
        assert task.split in ("text", "multimodal")
        assert task.metric in METRICS, f"{task.id}: 未実装の metric {task.metric}"
        assert task.metric in METRIC_INFO
        assert task.level in LEVELS, f"{task.id}: 未知の level"
        assert task.ahc_task_type in valid_task_types, f"{task.id}: 未知の task_type"
        assert task.answer_type in ("text", "number", "list", "dict")


@pytest.mark.parametrize("filename,expected_key,expected_shot", [
    ("BBBP_test.json", "BBBP_test", 0),
    ("3shot_BBBP_test_3shot.json", "BBBP_test", 3),
    ("SMILES转IUPAC", "SMILES转IUPAC", 0),
    ("3shot_SMILES转IUPAC", "SMILES转IUPAC", 3),
    ("2.文献理解\\1.信息抽取\\10.催化类型抽取_自建\\催化类型抽取_test.json",
     "2.文献理解/1.信息抽取/10.催化类型抽取_自建/催化类型抽取_test", 0),
])
def test_normalize_key(filename, expected_key, expected_shot):
    assert normalize_key(filename) == (expected_key, expected_shot)


def test_catalog_lookup_distinguishes_same_leaf_name():
    """极性_test は分類用と回帰用で別タスク（葉の名前だけでは決められない）。"""
    catalog = load_catalog()
    classification = ("3.分子理解\\3.分子性质预测\\1.5分子性质分类预测极性_自建\\极性_test.json")
    regression = ("3.分子理解\\3.分子性质预测\\2.5分子性质回归预测极性_自建\\极性_test.json")
    assert catalog.lookup(classification)[0].id == "property_polarity_classification"
    assert catalog.lookup(regression)[0].id == "property_polarity_regression"
    assert catalog.lookup("見たことのないタスク.json")[0] is None


# --- 答えの取り出し ------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ('{"answer": "C"}', "C"),
    ('考えました。\n```json\n{"answer": "-4.88"}\n```', "-4.88"),
    ('前置き {"answer": "Yes"} 後置き', "Yes"),
    ('{"answer": "A"} のあと {"answer": "B"}', "B"),      # 後方を優先
    ("answer: Carbocation", "Carbocation"),
])
def test_extract_from_text(text, expected):
    assert extract_from_text(text) == (expected, True)


def test_collect_answer_prefers_answer_file(tmp_path):
    (tmp_path / ANSWER_FILE).write_text('{"answer": ["CCO", "CCN"]}', encoding="utf-8")
    result = collect_answer(tmp_path, '{"answer": "無視される"}', "list")
    assert result["answer"] == ["CCO", "CCN"]
    assert result["answer_source"] == ANSWER_FILE


def test_collect_answer_falls_back_to_final_message(tmp_path):
    result = collect_answer(tmp_path, "説明だけで JSON が無い", "text")
    assert result["answer_source"] == "final_message_text"
    result = collect_answer(tmp_path, "", "text")
    assert result["answer"] is None and result["answer_source"] == "none"


def test_collect_answer_joins_list_for_text_tasks(tmp_path):
    (tmp_path / ANSWER_FILE).write_text('{"answer": ["NMR", "MS"]}', encoding="utf-8")
    assert collect_answer(tmp_path, "", "text")["answer"] == "NMR, MS"


# --- 採点 ---------------------------------------------------------------

@pytest.mark.parametrize("metric,answer,target,expected", [
    ("choice", "最終的に C", "C", 1.0),
    ("choice", "A", "C", 0.0),
    ("choice", "ACD", "ACD", 1.0),
    ("true_false", "Incorrect", "Incorrect", 1.0),
    ("true_false", "Correct", "Incorrect", 0.0),
    ("yes_no", "Yes", "Yes", 1.0),
    ("yes_no", "no", "Yes", 0.0),
    ("contains", "This is Organic Chemistry", "Organic Chemistry", 1.0),
    ("contains", "Materials", "Organic Chemistry", 0.0),
    ("entity_f1", "EBNA, nucleotide", "EBNA, nucleotide", 1.0),
    ("entity_f1", "EBNA", "EBNA, nucleotide", pytest.approx(2 / 3)),
    ("relation_f1", "(a,b)(c,d)", "(a,b)(c,d)", 1.0),
    ("reagent_f1", "Ic1ccc2ncccc2c1", "Ic1ccc2ncccc2c1", 1.0),
    ("range_overlap", "180-370 kJ/mol", "180-370 kJ/mol", 1.0),
    ("range_overlap", "0-10 kJ/mol", "180-370 kJ/mol", 0.0),
    ("formula", "C5H8", "H8C5", 1.0),
    ("iupac", "Ethanol", "ethanol", 1.0),
    ("smiles", "CCO", "CCO", 1.0),          # rdkit なし → 文字列一致で採点
    ("text_exact", "$C_{10}H_{10}$", "$C_{10} H_{10}$", 1.0),
    ("text_exact", r"$H_2 + Cl_2 \rightarrow 2HCl$", r"$H_2+Cl_2 \to 2HCl$", 1.0),
    ("text_exact", "$CH_4$", "$C_{10}H_{10}$", 0.0),
])
def test_score_item_basic(metric, answer, target, expected):
    assert score_item(metric, answer, target)["score"] == expected


def test_reaction_smiles_scores_role_wise_without_rdkit():
    """rdkit なしでも役割（反応物 > 条件 > 生成物）ごとの集合 F1 で部分点を出す。"""
    gold = "CCO.CC(=O)O>Cc1ccccc1>CCOC(C)=O"
    assert score_item("reaction_smiles", gold, gold)["metrics"]["exact_match"] == 1.0
    partial = score_item("reaction_smiles", "CCO>Cc1ccccc1>CCOC(C)=O", gold)
    assert 0.0 < partial["score"] < 1.0
    assert partial["metrics"]["exact_match"] == 0.0


def test_multimodal_task_key_comes_from_file_path(tmp_path):
    """multimodal split は filename が空（"nan"）で、file_path がタスク名になる。"""
    path = tmp_path / "multimodal.jsonl"
    path.write_text(json.dumps({
        "index": 0, "filename": "nan", "file_path": "2D分子识别_60.jsonl",
        "query": "画像の分子の SMILES を出せ <ImageHere>", "target": "CSc1ncc(Br)c(N)n1",
        "image": "00000_Molecule_1.png", "img_path": "images/Molecule_1.png",
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    items = dataset.load_items("multimodal", path=path, auto_prepare=False)
    assert len(items) == 1
    assert items[0].task_id == "mm_structure_to_smiles"
    assert items[0].metric == "smiles" and items[0].image == "00000_Molecule_1.png"
    # 画像つきの問題ではプロンプトに画像への言及が入る
    assert "00000_Molecule_1.png" in runner.build_request(items[0])


def test_regression_reports_errors_not_score():
    result = score_item("regression", "-4.0", "-4.883")
    assert result["score"] is None
    assert result["metrics"]["abs_error"] == pytest.approx(0.883)
    # 範囲で答えた場合は中央値を使う（公式の平均処理と同じ）
    assert score_item("regression", "100-120 K", "110K")["metrics"]["abs_error"] == 0.0


def test_sider_counts_matching_labels():
    gold = "{'Eye disorders': 'Yes', 'Cardiac disorders': 'No'}"
    result = score_item("sider", {"Eye disorders": "Yes", "Cardiac disorders": "Yes"}, gold)
    assert result["metrics"]["label_accuracy"] == 0.5
    assert result["metrics"]["strict_match"] == 0.0


def test_judge_metric_is_excluded_without_judge():
    without = score_item("judge", "なにか答えた", "参照解答")
    assert without["score"] is None and without["answered"] is True
    with_judge = score_item("judge", "なにか答えた", "参照解答",
                            {"judge": {"score": 0.8, "reason": "おおむね正しい"}})
    assert with_judge["score"] == 0.8


def test_missing_answer_scores_zero_but_marks_unanswered():
    result = score_item("yes_no", None, "Yes")
    assert result["score"] == 0.0 and result["answered"] is False


# --- run → 採点 → レポートの一巡 ----------------------------------------

FIXTURE_ROWS = [
    # (filename, query, target)
    ("选择任务.json", "Q1 選択問題", "C"),
    ("BBBP_test.json", "Q2 BBBP", "Yes"),
    ("ESOL_test.json", "Q3 ESOL", "-4.883"),
    ("简答任务.json", "Q4 短答", "参照解答の本文"),
]


def _fixture_jsonl(tmp_path: Path) -> Path:
    path = tmp_path / "text.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for index, (filename, query, target) in enumerate(FIXTURE_ROWS):
            fh.write(json.dumps({"index": index, "filename": filename,
                                 "query": query, "target": target},
                                ensure_ascii=False) + "\n")
    return path


class StubController:
    """answer ファイルを書くだけの controller（SDK を使わない）。

    answers は task_id → 答え。None を渡すと答えファイルを作らない（無回答の再現）。
    """

    def __init__(self, config, answers):
        self.config = config
        self.answers = answers
        self.seen = []

    async def run(self, request, provider=None, task_type=None, expected_outputs=None,
                  copy_inputs=None, run_id=None, **kwargs):
        self.seen.append({"request": request, "task_type": task_type,
                          "expected_outputs": expected_outputs, "provider": provider,
                          "run_id": run_id})
        workspace = self.config.paths.workspaces / run_id
        workspace.mkdir(parents=True, exist_ok=True)
        task_id = next(k for k in self.answers if f"-{k}-" in run_id)
        answer = self.answers[task_id]
        if answer is not None:
            (workspace / ANSWER_FILE).write_text(
                json.dumps({"answer": answer}, ensure_ascii=False), encoding="utf-8")
        task = TaskSpec(description=request, expected_outputs=expected_outputs or [])
        return RunReport(
            run_id=run_id, task=task, provider=provider, passed=answer is not None,
            attempts=1, verification=VerificationResult(passed=answer is not None),
            final_message="stub run")


def test_end_to_end_with_stub_controller(tmp_path):
    config = load_config()
    config.paths.workspaces = tmp_path / "workspaces"
    config.paths.traces = tmp_path / "traces"

    items = dataset.load_items(path=_fixture_jsonl(tmp_path), limit_per_task=1,
                              auto_prepare=False)
    assert [item.task_id for item in items] == [
        "objective_choice", "subjective_short_answer", "property_bbbp", "property_esol"]

    # 選択=正解 / 短答=自由記述 / BBBP=誤答 / ESOL=近い数値
    controller = StubController(config, {
        "objective_choice": "C", "subjective_short_answer": "それらしい説明",
        "property_bbbp": "No", "property_esol": "-4.0"})
    records_path = tmp_path / "results" / "records.jsonl"
    runner_config = runner.RunnerConfig(label="test", providers=("deepagents",),
                                        concurrency=2)
    produced = asyncio.run(
        runner.run_items(config, items, runner_config, records_path, controller=controller))
    assert len(produced) == 4
    # ChemEval の query 本文と答えの保存先指示がプロンプトに入っている
    assert all(ANSWER_FILE in call["request"] for call in controller.seen)
    assert any("Q1 選択問題" in call["request"] for call in controller.seen)
    assert all(call["expected_outputs"] == [ANSWER_FILE] for call in controller.seen)

    scored = asyncio.run(score.score_records(runner.read_records(records_path),
                                             config=None, judge=None, use_chem=False))
    by_task = {r["task_id"]: r for r in scored}
    assert by_task["objective_choice"]["score"] == 1.0
    assert by_task["property_bbbp"]["score"] == 0.0
    assert by_task["property_esol"]["score"] is None            # 回帰は RMSE で集計
    assert by_task["property_esol"]["metrics"]["abs_error"] == pytest.approx(0.883)
    assert by_task["subjective_short_answer"]["score"] is None  # judge 未実施

    summary = report.write_report(scored, tmp_path / "results", "test")
    overall = summary["providers"]["deepagents"]["overall"]
    assert overall["n"] == 4
    assert overall["macro_task_score"] == 0.5                   # 選択 1.0 / BBBP 0.0
    assert (tmp_path / "results" / "report.md").exists()
    assert (tmp_path / "results" / "metrics.json").exists()
    markdown = (tmp_path / "results" / "report.md").read_text(encoding="utf-8")
    assert "property_esol" in markdown and "rmse=" in markdown


def test_resume_skips_recorded_items(tmp_path):
    config = load_config()
    config.paths.workspaces = tmp_path / "workspaces"
    config.paths.traces = tmp_path / "traces"
    items = dataset.load_items(path=_fixture_jsonl(tmp_path), limit_per_task=1,
                              auto_prepare=False)
    records_path = tmp_path / "results" / "records.jsonl"
    runner_config = runner.RunnerConfig(label="test", providers=("deepagents",))
    answers = {"objective_choice": "C", "subjective_short_answer": "説明",
               "property_bbbp": "Yes", "property_esol": "-4.9"}

    first = StubController(config, answers)
    asyncio.run(runner.run_items(config, items[:2], runner_config, records_path,
                                 controller=first))
    second = StubController(config, answers)
    asyncio.run(runner.run_items(config, items, runner_config, records_path,
                                 controller=second))
    assert len(second.seen) == 2                     # 既に記録済みの 2 件は飛ばす
    assert len(runner.read_records(records_path)) == 4
