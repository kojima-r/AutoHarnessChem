"""ChemEval 評価スクリプトのテスト（データのダウンロード・SDK・rdkit を要求しない）。

検証するのは
  - カタログ（tasks.yaml）の整合性と filename の正規化
  - エージェント出力からの答えの取り出し
  - 各 metric の採点（正解 → 満点 / 誤答 → 0 点）
  - スタブ controller での run → 採点 → レポートの一巡
"""
import asyncio
import json
import sys
import types
from unittest import mock
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


# --- 素の LLM ベースライン（bare モード）--------------------------------

class StubBareModel:
    """1 ターン問い合わせのスタブ（SDK を要求しない）。"""

    def __init__(self, answers: dict[str, str]):
        self.answers = answers
        self.reported_model = "stub-model-1"
        self.asked: list[str] = []

    async def ask(self, prompt: str) -> str:
        self.asked.append(prompt)
        for key, text in self.answers.items():
            if key in prompt:
                return text
        return ""


def test_bare_mode_answers_without_harness(tmp_path):
    from benchmarks_chemeval import bare

    items = dataset.load_items(path=_fixture_jsonl(tmp_path), limit_per_task=1,
                              auto_prepare=False)
    asker = StubBareModel({"Q1 選択問題": '{"answer": "C"}',
                           "BBBP": '{"answer": "No"}'})
    records_path = tmp_path / "bare" / "records.jsonl"
    runner_config = runner.RunnerConfig(label="bare-test", providers=("claude",),
                                        concurrency=2)
    produced = asyncio.run(
        bare.run_items_bare(items, runner_config, records_path, asker=asker))

    assert len(produced) == len(items)
    by_task = {r["task_id"]: r for r in produced}
    # 問題文は原文のまま渡す（答えファイルの指示を足さない = 論文と同条件）
    assert all(ANSWER_FILE not in prompt for prompt in asker.asked)
    assert by_task["objective_choice"]["answer"] == "C"
    assert by_task["objective_choice"]["mode"] == "bare"
    assert by_task["objective_choice"]["model"] == "stub-model-1"
    # harness を通していないので Verifier の結果は持たない
    assert by_task["objective_choice"]["harness_passed"] is None
    # 採点は harness 側と同じ経路
    scored = asyncio.run(score.score_records(produced, config=None, judge=None,
                                             use_chem=False))
    assert {r["task_id"]: r["score"] for r in scored}["objective_choice"] == 1.0
    # 再実行すると記録済みは飛ばす（枠をまたいだ再開ができる）
    again = asyncio.run(
        bare.run_items_bare(items, runner_config, records_path, asker=asker))
    assert again == []


def test_bare_mode_disables_tools_and_rejects_tool_use():
    """素の LLM ベースラインでツールを使わせない。

    `allowed_tools=[]` は「許可リスト未指定」の意味でツールは無効にならず、
    初回の bare 実行ではモデルが Bash から selfies ライブラリを呼んで答えていた
    （= 素の LLM ではない）。ツールセット自体を空にする `tools=[]` が必要。
    """
    from benchmarks_chemeval import bare

    captured = {}

    class FakeOptions:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class ToolUseBlock:
        name = "Bash"

    class AssistantMessage:
        content = [ToolUseBlock()]

    async def fake_query(prompt, options):
        for message in [AssistantMessage()]:
            yield message

    module = types.ModuleType("claude_agent_sdk")
    module.ClaudeAgentOptions = FakeOptions
    module.query = fake_query
    with mock.patch.dict(sys.modules, {"claude_agent_sdk": module}):
        asker = bare.BareModel(model="claude-opus-5")
        with pytest.raises(bare.ToolUseDetected):
            asyncio.run(asker.ask("SMILES を SELFIES に変換してください"))
    # ツールセットを空にして渡していること
    assert captured["tools"] == []
    assert captured["max_turns"] >= 2


def test_bare_mode_treats_usage_limit_message_as_failure(tmp_path):
    """枠切れの本文を答えとして記録しない（記録すると再実行できなくなる）。

    SDK は枠切れを例外ではなく `You've hit your session limit · resets ...` という
    応答本文で返す。初回の bare 実行ではこれを答えとして採用してしまい、
    2,210 問中 1,346 問が「回答済みだが 0 点」として確定していた。
    """
    from benchmarks_chemeval import bare

    items = dataset.load_items(path=_fixture_jsonl(tmp_path), limit_per_task=1,
                              auto_prepare=False)
    asker = StubBareModel({"": "You've hit your session limit · resets 4:50am (Asia/Tokyo)"})
    # スタブは素通しなので、実装側の判定を通すために ask をラップする
    raw_ask = asker.ask

    async def ask(prompt):
        text = await raw_ask(prompt)
        if bare._LIMIT_MESSAGE.search(text):
            raise bare.UsageLimitReached(text)
        return text

    asker.ask = ask
    records_path = tmp_path / "limit" / "records.jsonl"
    runner_config = runner.RunnerConfig(label="limit-test", providers=("claude",),
                                        concurrency=2)
    produced = asyncio.run(
        bare.run_items_bare(items, runner_config, records_path, asker=asker))
    assert produced and all(r["answer"] is None for r in produced)
    assert all("UsageLimitReached" in (r["error"] or "") for r in produced)
    # answer が無いので purge の対象になり、枠が戻れば解き直せる
    assert all(r["answer_source"] == "none" for r in produced)


def test_synthetic_model_names_are_ignored():
    """SDK の `<synthetic>` を実行モデルとして記録しない。

    bare の初回実行で 2,210 件のうち 1,349 件がこれで `<synthetic>` になり、
    「どのモデルの成績か」が言えなくなった。
    """
    from adapters.base import real_model_name

    assert real_model_name("claude-opus-5") == "claude-opus-5"
    assert real_model_name("<synthetic>") is None
    assert real_model_name("") is None
    assert real_model_name(None) is None


def test_compare_column_uses_same_field_and_subset():
    """比較列は ahc 側と同じ指標で取り、母集団を揃える。"""
    from benchmarks_chemeval import baselines as bl

    doc = {"systems": ["A"],
           "tasks": [{"task_id": "objective_choice", "paper_task": "MCTask",
                      "paper_metric": "Accuracy", "comparable": True,
                      "ours_field": "score", "values": {"A": 50.0}},
                     {"task_id": "iupac_to_smiles", "paper_task": "IUPAC2SMILES",
                      "paper_metric": "Tanimoto (valid)", "comparable": True,
                      "ours_field": "tanimoto", "values": {"A": 30.0}}],
           "aggregates": []}
    ours = {"objective_choice": {"score": 0.9, "metrics": {}},
            "iupac_to_smiles": {"score": 0.6, "metrics": {"tanimoto": 0.8}}}
    # 相手側は 1 タスクだけ持つ（母集団が揃うか）
    other = {"objective_choice": {"score": 0.5, "metrics": {}}}
    out = bl.compare(ours, doc, other_tasks=other, other_label="bare")
    column = out["compare"]
    assert column["n_tasks"] == 1
    assert column["macro"] == 0.5
    assert column["ours_macro_same_subset"] == 0.9      # 全体平均(0.85)ではない
    assert column["tasks"]["iupac_to_smiles"] is None


# --- 文献値（論文 Table 1）------------------------------------------------

def test_baselines_are_consistent_with_catalog():
    """baselines.yaml の task_id・手法・指標の整合性を検査する。"""
    from benchmarks_chemeval import baselines as bl

    doc = bl.load_baselines()
    known = {task.id for task in load_catalog()}
    systems = doc["systems"]
    assert len(systems) == len(set(systems)) == 13
    for entry in doc["tasks"]:
        assert entry["task_id"] in known, entry["task_id"]
        # 比較可能な行は 13 手法すべての値を持ち、ahc 側の比較先が決まっていること
        if entry["comparable"]:
            assert set(entry["values"]) == set(systems), entry["task_id"]
            assert all(v is not None for v in entry["values"].values()), entry["task_id"]
            assert entry.get("ours_field"), entry["task_id"]
    for agg in doc["aggregates"]:
        assert all(t in known for t in agg["task_ids"]), agg["paper_task"]


def test_baseline_comparison_scales_and_picks_best():
    """0..100 の文献値を 0..1 に直し、最高値・Δ を正しく出す。"""
    from benchmarks_chemeval import baselines as bl

    doc = {"systems": ["A", "B"],
           "tasks": [{"task_id": "objective_choice", "paper_task": "MCTask",
                      "paper_metric": "Accuracy", "comparable": True,
                      "ours_field": "score", "values": {"A": 40.0, "B": 80.0}},
                     {"task_id": "iupac_to_smiles", "paper_task": "IUPAC2SMILES",
                      "paper_metric": "Tanimoto (valid)", "comparable": True,
                      "ours_field": "tanimoto", "values": {"A": 50.0, "B": 30.0}},
                     {"task_id": "smiles_to_formula", "paper_task": "SMILES2MF",
                      "paper_metric": "L2", "comparable": False,
                      "values": {"A": 0.5, "B": 0.6}}],
           "aggregates": []}
    tasks = {"objective_choice": {"score": 0.95, "metrics": {}},
             "iupac_to_smiles": {"score": 0.62, "metrics": {"tanimoto": 0.964}},
             "smiles_to_formula": {"score": 0.98, "metrics": {}}}
    rows = bl.task_rows(tasks, doc)
    assert rows["objective_choice"]["best"] == 0.8
    assert rows["objective_choice"]["best_system"] == "B"
    assert rows["objective_choice"]["delta"] == pytest.approx(0.15)
    # score ではなく tanimoto と比べる
    assert rows["iupac_to_smiles"]["ours"] == 0.964
    assert rows["iupac_to_smiles"]["delta"] == pytest.approx(0.464)
    # 指標が違う行は比較しない（生の値のまま残す）
    assert rows["smiles_to_formula"]["best"] is None
    assert rows["smiles_to_formula"]["values"] == {"A": 0.5, "B": 0.6}
    macro = bl.system_macro(rows, doc)
    assert macro["n_tasks"] == 2
    assert macro["systems"]["B"] == pytest.approx(0.55)
    assert macro["ours"] == pytest.approx((0.95 + 0.964) / 2)


def test_parse_judgement_recovers_score_from_truncated_output():
    """reason の途中で出力が切れても、judge が付けた score は回収する。

    実際のフル評価で 9 件がこれで未採点になった（JSON が閉じないため
    ブロック抽出が失敗する）。score は先頭に出ているので拾える。
    """
    from benchmarks_chemeval.judge import parse_judgement

    assert parse_judgement('{"score": 7, "reason": "おおむね正しい"}')["score"] == 0.7
    truncated = '{"score": 8, "reason": "参照解答の主要要素をすべて含むが、要求された形式では'
    assert parse_judgement(truncated)["score"] == 0.8
    assert parse_judgement("採点できません") is None


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
