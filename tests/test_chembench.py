"""ChemBench 評価スクリプトのテスト（SDK・ネットワーク・重い依存を要求しない）。

検証するのは
  - カタログ（topics.yaml）の整合性
  - 公式 report からの出題の復元（選択肢の文字 → 得点の対応）
  - エージェント出力からの答えの取り出し
  - 採点（公式の all_correct 規則との一致を**公開 report で実測**）
  - 文献値の突き合わせ（同一問題への絞り込み）
  - スタブ controller / スタブ LLM での run → 採点 → レポートの一巡
"""
import asyncio
import json

import pytest

from benchmarks_chembench import baselines, dataset, report, runner, score, validate
from benchmarks_chembench.catalog import (METRIC_KINDS, TOPICS, load_catalog,
                                          topic_id)
from benchmarks_chembench.extract import ANSWER_FILE, collect_answer, extract_from_text
from benchmarks_chembench.metrics import (classification_scores, parse_number,
                                          prepare_mcq_answer, score_item)
from harness.config import load_config
from schemas import RunReport, TaskSpec, VerificationResult

# 公式クローンが無い環境ではデータ依存のテストを飛ばす
_HAS_CLONE = (dataset.REPORTS_DIR / dataset.DEFAULT_REFERENCE_MODEL).exists()
needs_clone = pytest.mark.skipif(
    not _HAS_CLONE, reason="benchmarks_chembench/chembench のクローンが無い")


# --- カタログ -----------------------------------------------------------

def test_catalog_is_consistent():
    """トピック定義が実装（TaskType）と対応していて、重複が無い。"""
    from schemas.models import TaskType

    catalog = load_catalog()
    assert len(catalog) == 9
    valid_task_types = set(TaskType.__args__)
    for topic in catalog:
        assert topic.id in TOPICS
        assert topic.ahc_task_type in valid_task_types, f"{topic.id}: 未知の task_type"
        assert topic.name and topic.description
    assert len({t.id for t in catalog}) == 9


def test_topic_id_normalizes_csv_labels():
    assert topic_id("Chemical Preference") == "chemical_preference"
    assert topic_id("Toxicity and Safety") == "toxicity_and_safety"


@needs_clone
def test_question_meta_covers_every_published_question():
    """公式の分類 CSV が公開 report の全問を覆っている（推測で補わない）。"""
    catalog = load_catalog()
    names = {path.stem for path in
             dataset._question_report_files(dataset.DEFAULT_REFERENCE_MODEL)}
    assert len(names) > 2000
    assert not {n for n in names if catalog.meta(n) is None}


# --- 出題の復元 ---------------------------------------------------------

_MCQ_PROMPT = """The following is a multiple choice question about chemistry.
Please answer by responding with the letter of the correct answer.

Question: Which gas is isoelectronic with the carbide ion?

Options:
A. Carbon monoxide
B. Nitric oxide
C. Dilute H2SO4 with Pt electrodes

You MUST include the letter(s) of the correct answer (separated by comma if there are many) within the following tags: [ANSWER] and [/ANSWER].
For example, '[ANSWER]<answer>[/ANSWER]'."""


def test_parse_question_and_options():
    assert dataset.parse_question(_MCQ_PROMPT) == (
        "Which gas is isoelectronic with the carbide ion?")
    options = dataset.parse_options(_MCQ_PROMPT)
    assert options == {"A": "Carbon monoxide", "B": "Nitric oxide",
                       "C": "Dilute H2SO4 with Pt electrodes"}


def test_resolve_score_map_strips_latex_like_the_official_pipeline():
    """正解側は生の LaTeX、プロンプト側は後処理済み。剥がしてから突き合わせる。"""
    options = dataset.parse_options(_MCQ_PROMPT)
    targets = {"Carbon monoxide": 1, "Nitric oxide": 0,
               r"Dilute \ce{H2SO4} with \ce{Pt} electrodes": 0}
    assert dataset.resolve_score_map(options, targets) == {"A": 1.0, "B": 0.0, "C": 0.0}


def test_resolve_score_map_refuses_ambiguous_mapping():
    """対応が取れないときは黙って 0 点にせず None を返す（誤採点を防ぐ）。"""
    options = {"A": "Carbon monoxide", "B": "Nitric oxide"}
    assert dataset.resolve_score_map(options, {"Something else": 1}) is None
    # 正解が 1 つも無い問題は hamming が計算できないので落とす
    assert dataset.resolve_score_map(
        options, {"Carbon monoxide": 0.5, "Nitric oxide": 0}) is None


def test_post_process_matches_official_regexes():
    assert dataset.post_process(r"aq. \ce{NaCl} and $\Delta G$") == "aq. NaCl and \\Delta G"
    assert dataset.post_process("[START_SMILES]CCO[END_SMILES]") == "CCO"
    assert dataset.post_process(r"\pu{10 t}") == "10 t"


# --- 答えの取り出し ------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("よく考えると [ANSWER]C[/ANSWER] です", "C"),
    ("[ANSWER]A,C[/ANSWER]", "A,C"),
    ("[ANSWER]42292.49[/ANSWER]", "42292.49"),
    ("まず [ANSWER]A[/ANSWER] …訂正して [ANSWER]B[/ANSWER]", "B"),   # 後方を優先
])
def test_extract_from_text(text, expected):
    assert extract_from_text(text) == (expected, True)


def test_collect_answer_prefers_answer_file(tmp_path):
    (tmp_path / ANSWER_FILE).write_text('{"answer": "B"}', encoding="utf-8")
    result = collect_answer(tmp_path, "[ANSWER]C[/ANSWER]", "mcq")
    assert result["answer"] == "B"
    assert result["answer_source"] == ANSWER_FILE


def test_collect_answer_falls_back_to_answer_tag_then_text(tmp_path):
    result = collect_answer(tmp_path, "結論は [ANSWER]D[/ANSWER]", "mcq")
    assert (result["answer"], result["answer_source"]) == ("D", "answer_tag")
    result = collect_answer(tmp_path, "タグを書き忘れた説明だけ", "mcq")
    assert result["answer_source"] == "final_message_text"
    result = collect_answer(tmp_path, "", "mcq")
    assert result["answer"] is None and result["answer_source"] == "none"


def test_collect_answer_joins_list_for_mcq(tmp_path):
    (tmp_path / ANSWER_FILE).write_text('{"answer": ["A", "C"]}', encoding="utf-8")
    assert collect_answer(tmp_path, "", "mcq")["answer"] == "A, C"


# --- 採点 ---------------------------------------------------------------

def test_mcq_requires_exact_set_match():
    """選択肢問題は部分点なし（hamming == 0 のみ 1 点）。"""
    record = {"metric_kind": "mcq", "score_map": {"A": 1, "B": 0, "C": 1, "D": 0}}
    assert score_item(record, "[ANSWER]A,C[/ANSWER]")["score"] == 1.0
    partial = score_item(record, "[ANSWER]A[/ANSWER]")
    assert partial["score"] == 0.0                     # 取りこぼしは 0 点
    assert partial["metrics"]["f1"] == pytest.approx(2 / 3)   # 出来は f1 に出る
    assert score_item(record, "[ANSWER]A,B,C[/ANSWER]")["score"] == 0.0   # 余計も 0 点


def test_numeric_uses_one_percent_of_the_target():
    record = {"metric_kind": "numeric", "target": 42000.0, "tolerance": 420.0}
    assert score_item(record, "[ANSWER]42100[/ANSWER]")["score"] == 1.0
    assert score_item(record, "[ANSWER]42292.49[/ANSWER]")["score"] == 1.0   # 差 292 < 420
    assert score_item(record, "[ANSWER]43000[/ANSWER]")["score"] == 0.0      # 差 1000 > 420
    assert score_item(record, "[ANSWER]4.21e4[/ANSWER]")["score"] == 1.0


def test_missing_answer_scores_zero_but_marks_unanswered():
    record = {"metric_kind": "mcq", "score_map": {"A": 1, "B": 0}}
    result = score_item(record, None)
    assert result["score"] == 0.0 and result["answered"] is False


@pytest.mark.parametrize("text,expected", [
    ("42292.49", 42292.49),
    ("1.5 * 10^-3", 0.0015),
    ("3e5", 300000.0),
    ("5 kJ", 5.0),          # NUM_REGEX が数値トークンだけを切り出す
])
def test_parse_number(text, expected):
    assert parse_number(text) == pytest.approx(expected)


def test_prepare_mcq_answer_matches_official_behaviour():
    """公式の正規表現は選択肢の**本文**で答えると拾えない（= 0 点）。

    本実装は公式の LLM 抽出フォールバックを持たないぶん**わずかに厳しい**。
    この差は ahc 側に不利に働くので、比較としては安全側。
    """
    assert prepare_mcq_answer("[ANSWER]A,B[/ANSWER]") == "['A,B']"
    assert not prepare_mcq_answer("The point group is D3d.\n\n[ANSWER]D3d[/ANSWER]")


def test_classification_scores_matches_official_formula():
    scores = classification_scores({"A": 1, "B": 1, "C": 0}, ["A", "C"])
    assert scores["hamming"] == pytest.approx(1.0)      # (取りこぼし1 + 余計1) / 正解2
    assert scores["precision"] == pytest.approx(0.5)
    assert scores["recall"] == pytest.approx(0.5)


@needs_clone
def test_scoring_agrees_with_official_reports():
    """公式が記録した正誤と本実装の判定が一致する（比較の前提）。

    ランダムベースラインは refusal が無く全問比較できるので完全一致を要求する。
    """
    result = validate.validate_model("random_baseline")
    assert result["compared"] > 2700
    assert result["agreement"] == 1.0
    # 実モデルでも 99.9% 以上（残りは公式側の LLM 抽出フォールバック由来）
    gpt = validate.validate_model("gpt-4o")
    assert gpt["agreement"] >= 0.999
    assert gpt["numeric_agreement"] == 1.0


# --- 文献値 -------------------------------------------------------------

@needs_clone
def test_discover_models_finds_published_reports():
    models = baselines.discover_models()
    assert len(models) >= 30
    assert "gpt-4o" in models and "random_baseline" in models
    # ツールを使う構成は ahc と同じ土俵なので必ず種別が付いている
    assert models["gpt-4o-react"]["kind"] == "agent"
    assert models["gpt-4o"]["kind"] == "llm"
    # 人間は参加者の平均として擬似モデル化する
    assert models["humans-notool"]["kind"] == "human"
    assert models["humans-notool"]["n_participants"] >= 10
    assert all(model["scores"] for model in models.values())


def test_compare_restricts_literature_values_to_the_same_questions():
    """文献値は ahc が解いた問題だけを母集団にする（同一問題での比較）。"""
    models = {"fake": {"name": "Fake", "kind": "llm", "kind_label": "汎用LLM",
                       "tools": False, "source": "aggregate",
                       "scores": {"q1": 1.0, "q2": 0.0, "q3": 1.0}}}
    records = [{"question_name": "q1", "score": 1.0, "topic": "general_chemistry"},
               {"question_name": "q2", "score": 1.0, "topic": "general_chemistry"}]
    compared = baselines.compare(records, models=models)
    assert compared["n_questions"] == 2
    assert compared["ours"] == 1.0
    system = compared["systems"]["fake"]
    assert system["accuracy"] == 0.5          # q1/q2 だけ（q3 は母集団外）
    assert system["n"] == 2
    assert system["accuracy_full"] == 0.6667      # 全体は参考値として残す


def test_compare_reports_bare_column_on_the_shared_subset():
    models = {"fake": {"name": "Fake", "kind": "llm", "kind_label": "汎用LLM",
                       "tools": False, "source": "aggregate",
                       "scores": {"q1": 1.0, "q2": 1.0}}}
    records = [{"question_name": "q1", "score": 1.0, "topic": "t"},
               {"question_name": "q2", "score": 0.0, "topic": "t"}]
    # 素の LLM 側は q1 しか解けていない → n を分けて両方の平均を出す
    other = [{"question_name": "q1", "score": 0.0, "topic": "t"}]
    compared = baselines.compare(records, other_records=other, other_label="bare",
                                 models=models)
    assert compared["other"]["n"] == 1
    assert compared["other"]["accuracy"] == 0.0
    assert compared["other"]["ours_same_subset"] == 1.0    # 同じ 1 問での ahc の値


# --- run → 採点 → レポート ----------------------------------------------

def _items() -> list:
    """クローンを読まずに作る最小の item 2 件（選択肢 1 + 数値 1）。"""
    return [
        dataset.ChemBenchItem(
            question_name="demo-mcq", topic="general_chemistry",
            topic_name="General Chemistry", requires="Knowledge",
            difficulty="difficulty-basic", metric_kind="mcq",
            ahc_task_type="generic", prompt=_MCQ_PROMPT,
            question=dataset.parse_question(_MCQ_PROMPT),
            options=dataset.parse_options(_MCQ_PROMPT),
            score_map={"A": 1.0, "B": 0.0, "C": 0.0}, in_human_subset=True),
        dataset.ChemBenchItem(
            question_name="demo-num", topic="physical_chemistry",
            topic_name="Physical Chemistry", requires="Calculation",
            difficulty="", metric_kind="numeric", ahc_task_type="generic",
            prompt="Question: moles?\n\nYou MUST include [ANSWER] and [/ANSWER].",
            question="moles?", target=42000.0, tolerance=420.0),
    ]


class StubController:
    """answer ファイルを書くだけの controller（SDK を使わない）。"""

    def __init__(self, config, answers):
        self.config = config
        self.answers = answers
        self.seen = []

    async def run(self, request, provider=None, task_type=None, expected_outputs=None,
                  run_id=None, **kwargs):
        self.seen.append({"request": request, "task_type": task_type,
                          "expected_outputs": expected_outputs, "provider": provider,
                          "run_id": run_id})
        workspace = self.config.paths.workspaces / run_id
        workspace.mkdir(parents=True, exist_ok=True)
        name = next(k for k in self.answers if k in run_id)
        answer = self.answers[name]
        if answer is not None:
            (workspace / ANSWER_FILE).write_text(
                json.dumps({"answer": answer}), encoding="utf-8")
        task = TaskSpec(description=request, expected_outputs=expected_outputs or [])
        return RunReport(run_id=run_id, task=task, provider=provider,
                         passed=answer is not None, attempts=1,
                         verification=VerificationResult(passed=answer is not None),
                         final_message="stub run")


class StubBareModel:
    """1 往復の素の LLM のかわり（プロンプトをそのまま記録する）。"""

    def __init__(self, answers):
        self.answers = answers
        self.asked = []
        self.reported_model = "stub-model-1"

    async def ask(self, prompt):
        self.asked.append(prompt)
        for needle, answer in self.answers.items():
            if needle in prompt:
                return answer
        return ""


def test_end_to_end_with_stub_controller(tmp_path):
    config = load_config()
    config.paths.workspaces = tmp_path / "workspaces"
    config.paths.traces = tmp_path / "traces"

    items = _items()
    controller = StubController(config, {"demo-mcq": "A", "demo-num": "42100"})
    records_path = tmp_path / "results" / "records.jsonl"
    runner_config = runner.RunnerConfig(label="test", providers=("deepagents",),
                                        concurrency=2)
    produced = asyncio.run(
        runner.run_items(config, items, runner_config, records_path,
                         controller=controller))
    assert len(produced) == 2
    # ChemBench のプロンプト原文と答えの保存先指示が両方入っている
    assert all(ANSWER_FILE in call["request"] for call in controller.seen)
    assert any("isoelectronic" in call["request"] for call in controller.seen)
    assert all(call["expected_outputs"] == [ANSWER_FILE] for call in controller.seen)

    scored = score.score_records(runner.read_records(records_path))
    by_name = {r["question_name"]: r for r in scored}
    assert by_name["demo-mcq"]["score"] == 1.0
    assert by_name["demo-num"]["score"] == 1.0
    assert by_name["demo-num"]["metrics"]["mae"] == pytest.approx(100.0)

    summary = report.write_report(scored, tmp_path / "results", "test")
    overall = summary["providers"]["deepagents"]["overall"]
    assert overall["n"] == 2 and overall["score"] == 1.0
    assert set(summary["providers"]["deepagents"]["topics"]) == {
        "general_chemistry", "physical_chemistry"}
    assert (tmp_path / "results" / "report.md").exists()
    markdown = (tmp_path / "results" / "report.md").read_text(encoding="utf-8")
    assert "トピック別" in markdown and "General Chemistry" in markdown


def test_bare_mode_sends_the_prompt_verbatim(tmp_path):
    from benchmarks_chembench import bare

    items = _items()
    asker = StubBareModel({"isoelectronic": "[ANSWER]A[/ANSWER]",
                           "moles?": "[ANSWER]41000[/ANSWER]"})
    records_path = tmp_path / "bare" / "records.jsonl"
    runner_config = runner.RunnerConfig(label="bare-test", providers=("claude",),
                                        concurrency=2)
    produced = asyncio.run(
        bare.run_items_bare(items, runner_config, records_path, asker=asker))

    assert len(produced) == 2
    # 文献と同条件にするため、答えファイルの指示は足さない
    assert all(ANSWER_FILE not in prompt for prompt in asker.asked)
    by_name = {r["question_name"]: r for r in produced}
    assert by_name["demo-mcq"]["answer"] == "A"
    assert by_name["demo-mcq"]["mode"] == "bare"
    assert by_name["demo-mcq"]["model"] == "stub-model-1"
    # harness を通していないので Verifier の結果は持たない
    assert by_name["demo-mcq"]["harness_passed"] is None
    # 採点は harness 側と同じ経路
    scored = score.score_records(produced)
    assert {r["question_name"]: r["score"] for r in scored} == {
        "demo-mcq": 1.0, "demo-num": 0.0}
    # 再実行すると記録済みは飛ばす（枠をまたいだ再開ができる）
    assert asyncio.run(
        bare.run_items_bare(items, runner_config, records_path, asker=asker)) == []


def test_bare_mode_disables_tools_and_rejects_tool_use():
    """素の LLM ベースラインでツールを使わせない。

    `allowed_tools=[]` は「許可リスト未指定」の意味でツールは無効にならないため、
    ツールセット自体を空にする `tools=[]` が必要。
    """
    import sys
    import types
    from unittest import mock

    from benchmarks_chembench import bare

    captured = {}

    class FakeOptions:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class ToolUseBlock:
        name = "Bash"

    class AssistantMessage:
        model = "stub"
        content = [ToolUseBlock()]

    async def fake_query(prompt=None, options=None):
        yield AssistantMessage()

    module = types.ModuleType("claude_agent_sdk")
    module.ClaudeAgentOptions = FakeOptions
    module.query = fake_query
    with mock.patch.dict(sys.modules, {"claude_agent_sdk": module}):
        with pytest.raises(bare.ToolUseDetected):
            asyncio.run(bare.BareModel().ask("質問"))
    assert captured["tools"] == []
    assert captured["allowed_tools"] == []


def test_bare_mode_treats_usage_limit_message_as_failure():
    """枠切れの本文を答えとして記録しない（再開できなくなるため）。"""
    import sys
    import types
    from unittest import mock

    from benchmarks_chembench import bare

    class TextBlock:
        text = "You've hit your session limit · resets 4:50am (Asia/Tokyo)"

    class AssistantMessage:
        model = "stub"
        content = [TextBlock()]

    async def fake_query(prompt=None, options=None):
        yield AssistantMessage()

    module = types.ModuleType("claude_agent_sdk")
    module.ClaudeAgentOptions = type("O", (), {"__init__": lambda self, **k: None})
    module.query = fake_query
    with mock.patch.dict(sys.modules, {"claude_agent_sdk": module}):
        with pytest.raises(bare.UsageLimitReached):
            asyncio.run(bare.BareModel().ask("質問"))


def test_report_includes_three_way_comparison(tmp_path):
    """AHC / 素のLLM / 文献 の 3 列がレポートに揃う。"""
    models = {"fake": {"name": "Fake LLM", "kind": "llm", "kind_label": "汎用LLM",
                       "tools": False, "source": "aggregate",
                       "scores": {"demo-mcq": 0.0, "demo-num": 1.0}},
              "fake-agent": {"name": "Fake Agent", "kind": "agent",
                             "kind_label": "ツール利用エージェント", "tools": True,
                             "source": "aggregate",
                             "scores": {"demo-mcq": 1.0, "demo-num": 1.0}}}
    ahc = score.score_records([
        {"question_name": "demo-mcq", "topic": "general_chemistry",
         "metric_kind": "mcq", "score_map": {"A": 1, "B": 0}, "answer": "A",
         "provider": "claude", "requires": "Knowledge", "model": "claude-opus-5",
         "harness_passed": True, "attempts": 1, "elapsed_sec": 10.0},
        {"question_name": "demo-num", "topic": "physical_chemistry",
         "metric_kind": "numeric", "target": 100.0, "tolerance": 1.0, "answer": "100",
         "provider": "claude", "requires": "Calculation", "model": "claude-opus-5",
         "harness_passed": True, "attempts": 1, "elapsed_sec": 20.0}])
    bare_records = score.score_records([
        {"question_name": "demo-mcq", "topic": "general_chemistry",
         "metric_kind": "mcq", "score_map": {"A": 1, "B": 0}, "answer": "B",
         "provider": "claude", "mode": "bare", "model": "claude-opus-5"},
        {"question_name": "demo-num", "topic": "physical_chemistry",
         "metric_kind": "numeric", "target": 100.0, "tolerance": 1.0, "answer": "50",
         "provider": "claude", "mode": "bare", "model": "claude-opus-5"}])

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(baselines, "discover_models", lambda: models)
        summary = report.write_report(ahc, tmp_path, "test",
                                      compare_records=bare_records,
                                      compare_label="test-bare")
    block = summary["providers"]["claude"]
    assert block["baselines"]["ours"] == 1.0
    assert block["baselines"]["other"]["accuracy"] == 0.0
    markdown = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "素のLLM" in markdown and "Fake Agent" in markdown
    # ツールを使う手法との比較（同条件）が本文に出る
    assert "ツールを使うエージェント" in markdown
    # metrics.json に問題ごとの文献値を丸ごと残さない（肥大を防ぐ）
    saved = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    system = saved["providers"]["claude"]["baselines"]["systems"]["fake"]
    assert "scores" not in system


def test_metric_kinds_are_the_two_official_ones():
    assert set(METRIC_KINDS) == {"mcq", "numeric"}


def test_bare_report_does_not_claim_to_be_ahc(tmp_path):
    """bare モード単独のレポートを「ahc の成績」と読ませない。

    列見出しだけ直しても本文が嘘になるので、1.1 / 1.2 の主語も差し替わること。
    """
    records = score.score_records([
        {"question_name": "demo-mcq", "topic": "general_chemistry",
         "metric_kind": "mcq", "score_map": {"A": 1, "B": 0}, "answer": "A",
         "provider": "claude", "mode": "bare", "model": "claude-opus-5",
         "in_human_subset": True}])
    models = {"fake": {"name": "Fake LLM", "kind": "llm", "kind_label": "汎用LLM",
                       "tools": False, "source": "aggregate",
                       "scores": {"demo-mcq": 0.0}}}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(baselines, "discover_models", lambda: models)
        report.write_report(records, tmp_path, "bare-test")
    markdown = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "素の LLM（ツールなし・1 往復）に解かせて" in markdown
    assert "ahc の成績ではない" in markdown
    assert "**AutoHarnessChem（ahc）に解かせて**採点した" not in markdown
    # 順位表と human subset の主語も bare になる（ツールありと書かない）
    assert "**素のLLM（claude-opus-5、この run）**" in markdown
    assert "AHC（この run）" not in markdown


def test_bare_mode_labels_the_sdk_it_actually_used():
    """bare は claude-agent-sdk 直叩きなので config の provider を名乗らせない。

    名乗らせると「deepagents で測った素の LLM」というレポートになり、
    どの SDK の成績か読み手が追えなくなる。
    """
    from benchmarks_chembench import evaluate

    parser = evaluate.build_parser()
    args = parser.parse_args(["run", "--bare", "--dry-run", "--limit-per-topic", "1"])
    assert args.bare is True and args.providers is None

    captured = {}

    class FakeRuntime:
        provider = "deepagents"

    class FakeConfig:
        runtime = FakeRuntime()

    def fake_load_items(_args):
        return [_items()[0]]

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(evaluate, "_load_selected_items", fake_load_items)
        patch.setattr(evaluate, "print",
                      lambda *a, **k: captured.setdefault("lines", []).append(" ".join(map(str, a))),
                      raising=False)
        assert evaluate.cmd_run(args, FakeConfig()) == 0
    assert any("providers=('claude',)" in line for line in captured["lines"])


# --- 長期実行（枠をまたぐ再開） ------------------------------------------

def _write_records(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                    encoding="utf-8")


def test_purge_returns_unanswered_records_to_the_queue(tmp_path):
    """答えが無い record を残したまま再開すると永久に飛ばされる（枠切れの罠）。"""
    path = tmp_path / "records.jsonl"
    _write_records(path, [
        {"question_name": "ok", "provider": "claude", "answer": "A"},
        {"question_name": "limit", "provider": "claude", "answer": None,
         "error": "session limit"},
        {"question_name": "timeout", "provider": "claude", "answer": None,
         "error": "timeout"},
    ])
    stats = runner.purge_failed(path, max_retries=3)
    assert stats == {"total": 3, "purged": 2, "given_up": 0, "kept": 1,
                     "not_attempted": 0}
    # 答えのある record だけが残り、再開時に飛ばされるのはそれだけになる
    assert runner.load_done(path) == {("claude", "ok")}
    counts = json.loads((tmp_path / runner.PURGE_COUNTS_FILE).read_text(encoding="utf-8"))
    assert counts["retries"] == {"limit": 1, "timeout": 1}


def test_purge_gives_up_after_max_retries(tmp_path):
    """同じ問題を無限に回さない。上限に達したら確定失敗として残して前へ進む。"""
    path = tmp_path / "records.jsonl"
    failing = {"question_name": "hard", "provider": "claude", "answer": None,
               "error": "timeout"}
    for expected_purged in (1, 1):
        _write_records(path, [failing])
        assert runner.purge_failed(path, max_retries=2)["purged"] == expected_purged
    # 3 回目は諦めて残す（0 点として集計に入る）
    _write_records(path, [failing])
    stats = runner.purge_failed(path, max_retries=2)
    assert stats["purged"] == 0 and stats["given_up"] == 1
    kept = runner.read_records(path)
    assert kept[0]["purge_given_up"] == 2
    # 残ったので再開時は「実行済み」として飛ばされる
    assert runner.load_done(path) == {("claude", "hard")}


def test_purge_is_a_noop_when_everything_answered(tmp_path):
    path = tmp_path / "records.jsonl"
    rows = [{"question_name": "a", "provider": "claude", "answer": "A"}]
    _write_records(path, rows)
    before = path.read_text(encoding="utf-8")
    stats = runner.purge_failed(path, max_retries=3)
    assert stats["purged"] == 0
    assert path.read_text(encoding="utf-8") == before      # 書き換えない
    assert not (tmp_path / runner.PURGE_COUNTS_FILE).exists()


# --- 母集団（文献値の公表条件と揃っているか） ----------------------------

_FAKE_FULL = {
    # 難しいトピックほど問題数が多い、という ChemBench の性質を模した母集団
    "chemical_preference": ["p%d" % i for i in range(6)],
    "general_chemistry": ["g%d" % i for i in range(2)],
}


def _models_covering(names):
    return {"ref": {"name": "Ref", "kind": "llm", "kind_label": "汎用LLM",
                    "tools": False, "source": "aggregate",
                    "scores": {n: 1.0 for n in names}}}


def test_coverage_reweights_equal_per_topic_sampling(monkeypatch):
    """均等抽出の単純平均は公表値と比較できない。構成比で加重した値も出す。

    ChemBench は難しいトピックほど問題数が多いので、トピックから均等に取ると
    正答率が実際より高く出る。ここを黙っていると公表値と取り違えられる。
    """
    full = {t: len(v) for t, v in _FAKE_FULL.items()}
    monkeypatch.setattr(baselines, "full_benchmark_topics", lambda models=None: full)
    # 各トピック 2 問ずつ（均等抽出）。難しいトピックは 0 点、簡単なほうは満点
    records = (
        [{"question_name": f"p{i}", "topic": "chemical_preference", "score": 0.0}
         for i in range(2)]
        + [{"question_name": f"g{i}", "topic": "general_chemistry", "score": 1.0}
           for i in range(2)])
    cover = baselines.coverage(records, models={})
    assert cover["n_run"] == 4 and cover["n_full"] == 8
    assert cover["is_full"] is False
    assert cover["mix_matches_full"] is False          # 0.5/0.5 対 0.75/0.25
    # 単純平均は 0.5 だが、構成比で加重すると 0.25（= 難しいトピックが 3/4）
    assert cover["weighted_score"] == 0.25
    assert cover["topics"]["chemical_preference"]["share_full"] == 0.75
    assert cover["topics"]["chemical_preference"]["share_run"] == 0.5


def test_coverage_detects_a_full_run(monkeypatch):
    """全件なら構成比は自明に一致し、公表値と直接比較できる。"""
    full = {t: len(v) for t, v in _FAKE_FULL.items()}
    monkeypatch.setattr(baselines, "full_benchmark_topics", lambda models=None: full)
    records = [{"question_name": n, "topic": t, "score": 1.0}
               for t, names in _FAKE_FULL.items() for n in names]
    cover = baselines.coverage(records, models={})
    assert cover["is_full"] is True and cover["coverage"] == 1.0
    assert cover["mix_matches_full"] is True
    assert cover["weighted_score"] == 1.0


@needs_clone
def test_full_benchmark_topics_uses_the_published_question_set():
    """一部の report は改訂前の 2,854 問を含む。母集団は 2,788 問側を使う。"""
    full = baselines.full_benchmark_topics()
    assert sum(full.values()) == 2788
    assert full["chemical_preference"] == 1001
    assert full["toxicity_and_safety"] == 675


def test_report_warns_when_the_mix_does_not_match(tmp_path, monkeypatch):
    full = {"chemical_preference": 6, "general_chemistry": 2}
    monkeypatch.setattr(baselines, "full_benchmark_topics", lambda models=None: full)
    monkeypatch.setattr(baselines, "discover_models", lambda: {})
    records = score.score_records([
        {"question_name": "p0", "topic": "chemical_preference", "metric_kind": "mcq",
         "score_map": {"A": 1, "B": 0}, "answer": "B", "provider": "claude",
         "model": "claude-opus-5"},
        {"question_name": "g0", "topic": "general_chemistry", "metric_kind": "mcq",
         "score_map": {"A": 1, "B": 0}, "answer": "A", "provider": "claude",
         "model": "claude-opus-5"}])
    report.write_report(records, tmp_path, "mix-test")
    markdown = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "公表値とは比較不可" in markdown
    assert "構成比で加重した推定値" in markdown
    assert "--limit-per-topic 0" in markdown


def test_ahc_wrapper_knows_every_subcommand():
    """`ahc chembench <sub>` のサブコマンド一覧を委譲先から取っていること。

    app/cli.py に名前を列挙していると、サブコマンドを足したときに取りこぼして
    「未指定」と誤判定され `topics` が余計に渡る（purge を足したとき実際に壊れた）。
    """
    from unittest import mock

    from app import cli
    from benchmarks_chembench import evaluate

    known = set()
    for action in evaluate.build_parser()._subparsers._group_actions:
        known |= set(action.choices or {})
    assert {"topics", "validate", "verify-hf", "run", "purge", "score"} <= known

    for sub in sorted(known):
        args = mock.Mock(config=None, chembench_args=[sub, "--help"])
        with mock.patch.object(cli, "_cmd_chembench", wraps=cli._cmd_chembench):
            with mock.patch("benchmarks_chembench.evaluate.main",
                            side_effect=lambda argv: argv) as spy:
                forwarded = cli._cmd_chembench(args)
        # サブコマンドを渡しているのだから topics を足してはいけない
        assert forwarded == [sub, "--help"], f"{sub}: {forwarded}"
        assert spy.called


def _topic_item(name, topic):
    return dataset.ChemBenchItem(
        question_name=name, topic=topic, topic_name=topic, requires="", difficulty="",
        metric_kind="mcq", ahc_task_type="generic", prompt="p", question="q",
        options={"A": "a", "B": "b"}, score_map={"A": 1.0, "B": 0.0})


def test_balance_pending_corrects_an_existing_skew():
    """再開時は「前回やり残したトピック」を優先して構成比を取り戻す。

    目標の構成比を未実行分だけから取ると、前回の偏りがそのまま目標に化ける。
    """
    # 全体は big 8 問 / small 2 問。big は 6 問終わっていて small は未着手
    pending = ([_topic_item(f"b{i}", "big") for i in range(2)]
               + [_topic_item(f"s{i}", "small") for i in range(2)])
    ordered = runner.balance_pending(pending, {"big": 6})
    # big は既に 6/6 で目標（8 問中 6.4 問相当）を満たしているので small が先に来る
    assert [i.topic for i in ordered][:2] == ["small", "small"]


def test_balance_pending_keeps_the_mix_from_scratch():
    """初回（実行済みゼロ）は構成比どおりに交互に出す。"""
    pending = ([_topic_item(f"b{i}", "big") for i in range(6)]
               + [_topic_item(f"s{i}", "small") for i in range(2)])
    ordered = runner.balance_pending(pending, {})
    assert len(ordered) == 8
    assert sum(1 for i in ordered[:4] if i.topic == "big") == 3   # 6:2 の比を保つ
    # 全問がちょうど 1 回ずつ現れる（取りこぼし・重複が無い）
    assert sorted(i.question_name for i in ordered) == sorted(
        i.question_name for i in pending)


def test_balance_pending_handles_empty_and_single_topic():
    assert runner.balance_pending([], {"big": 3}) == []
    single = [_topic_item("a", "only"), _topic_item("b", "only")]
    assert [i.question_name for i in runner.balance_pending(single, {})] == ["a", "b"]


@needs_clone
def test_full_run_order_is_representative_at_any_prefix():
    """全件 run はどこで切っても母集団の縮小版になる（途中経過を公表値と比べるため）。"""
    import collections

    items = dataset.load_items(limit_per_topic=None)
    assert len(items) == 2788
    full = collections.Counter(i.topic for i in items)
    for prefix in (100, 500, 1500):
        seen = collections.Counter(i.topic for i in items[:prefix])
        assert len(seen) == 9, f"先頭 {prefix} 問に出ないトピックがある"
        worst = max(abs(seen[t] / prefix - full[t] / len(items)) for t in full)
        assert worst < 0.02, f"先頭 {prefix} 問の構成比のずれが大きい: {worst}"


def test_coverage_distinguishes_equal_sampling_from_a_partial_run(monkeypatch):
    """偏りの原因が「均等抽出」か「全件 run の途中」かで読み手の対処が違う。

    前者は設定を変える必要があり、後者は実行を進めれば解消する。
    """
    full = {"big": 6, "small": 2}
    monkeypatch.setattr(baselines, "full_benchmark_topics", lambda models=None: full)

    equal = [{"question_name": f"{t}{i}", "topic": t, "score": 1.0}
             for t in full for i in range(1)]
    assert baselines.coverage(equal, models={})["equal_sampling"] is True

    # 片方のトピックを全部終えた「途中」の状態は均等抽出ではない
    partial = ([{"question_name": f"big{i}", "topic": "big", "score": 1.0}
                for i in range(6)]
               + [{"question_name": "small0", "topic": "small", "score": 1.0}])
    assert baselines.coverage(partial, models={})["equal_sampling"] is False


def test_purge_does_not_spend_retries_on_quota_wipeouts(tmp_path):
    """枠切れで一斉に潰れた分は再試行回数に数えない。

    利用枠が切れると SDK は即エラーを返し、残り全問が「1 問 2 秒・未回答」で
    記録される（実測で残り 2,557 問が一斉に潰れた）。これを再試行に数えると
    枠切れが数回起きただけで全問が「確定失敗」になり、二度と解かれなくなる。
    """
    path = tmp_path / "records.jsonl"
    _write_records(path, [
        {"question_name": "ok", "provider": "claude", "answer": "A",
         "elapsed_sec": 30.0},
        # 枠切れ（即エラー）: 回数に数えず対象へ戻す
        {"question_name": "wiped", "provider": "claude", "answer": None,
         "error": "session limit", "elapsed_sec": 2.1},
        # 本当に時間をかけて失敗したもの: 回数に数える
        {"question_name": "real", "provider": "claude", "answer": None,
         "error": "timeout", "elapsed_sec": 600.0},
    ])
    stats = runner.purge_failed(path, max_retries=3)
    assert stats["purged"] == 2 and stats["not_attempted"] == 1
    counts = json.loads((tmp_path / runner.PURGE_COUNTS_FILE).read_text(encoding="utf-8"))
    assert counts["retries"] == {"real": 1}      # wiped は通常カウンタに入らない
    assert counts["fast"] == {"wiped": 1}        # 速い失敗は別カウンタで数える

    # 枠切れを何度繰り返しても（上限内なら）確定失敗にならない
    for _ in range(10):
        _write_records(path, [{"question_name": "wiped", "provider": "claude",
                               "answer": None, "elapsed_sec": 2.1}])
        stats = runner.purge_failed(path, max_retries=3, not_attempted_cap=50)
        assert stats["given_up"] == 0
        assert runner.load_done(path) == set()      # 毎回キューに戻る


def test_report_row_label_matches_the_skew_cause(tmp_path, monkeypatch):
    """単純平均の行の説明も、偏りの原因に合わせる（「均等抽出」と決め打ちしない）。"""
    full = {"big": 6, "small": 2}
    monkeypatch.setattr(baselines, "full_benchmark_topics", lambda models=None: full)
    monkeypatch.setattr(baselines, "discover_models", lambda: {})

    def render(records):
        report.write_report(score.score_records(records), tmp_path, "t")
        return (tmp_path / "report.md").read_text(encoding="utf-8")

    def rec(name, topic):
        return {"question_name": name, "topic": topic, "metric_kind": "mcq",
                "score_map": {"A": 1, "B": 0}, "answer": "A", "provider": "claude",
                "model": "m"}

    equal = render([rec("big0", "big"), rec("small0", "small")])
    assert "均等抽出なので**公表値とは比較不可**" in equal

    partial = render([rec(f"big{i}", "big") for i in range(6)] + [rec("small0", "small")])
    assert "構成比が偏っているので**公表値とは比較不可**" in partial
    assert "均等抽出なので" not in partial


def test_purge_gives_up_on_a_fast_failure_that_never_succeeds(tmp_path):
    """速い拒否は速い枠切れと区別が付かないので、別カウンタで上限を設ける。

    上限が無いと、モデルが安全上の理由で即座に断る問題（実例: TATP の合成に使う
    薬品を問う設問）が永久に再実行され、ループが終わらない。
    """
    path = tmp_path / "records.jsonl"
    failing = {"question_name": "refused", "provider": "claude", "answer": None,
               "error": "Claude Code returned an error result", "elapsed_sec": 3.5}
    for _ in range(3):
        _write_records(path, [failing])
        stats = runner.purge_failed(path, max_retries=5, not_attempted_cap=3)
        assert stats["not_attempted"] == 1 and stats["given_up"] == 0
        assert runner.load_done(path) == set()          # 毎回キューへ戻る

    # 上限に達したら確定失敗として残し、ループを前へ進める
    _write_records(path, [failing])
    stats = runner.purge_failed(path, max_retries=5, not_attempted_cap=3)
    assert stats["given_up"] == 1 and stats["purged"] == 0
    kept = runner.read_records(path)
    assert kept[0]["purge_given_up_reason"] == "not_attempted_cap"
    assert runner.load_done(path) == {("claude", "refused")}   # もう再実行しない

    # 通常の再試行カウンタは消費していない（枠切れ由来と混ぜない）
    stored = json.loads((tmp_path / runner.PURGE_COUNTS_FILE).read_text(encoding="utf-8"))
    assert stored["retries"] == {} and stored["fast"] == {"refused": 3}


def test_purge_reads_the_old_flat_counts_file(tmp_path):
    """旧形式（{name: 回数}）の purge_counts.json も読めること（実行中の run を壊さない）。"""
    path = tmp_path / "records.jsonl"
    (tmp_path / runner.PURGE_COUNTS_FILE).write_text('{"hard": 2}', encoding="utf-8")
    _write_records(path, [{"question_name": "hard", "provider": "claude",
                           "answer": None, "elapsed_sec": 600.0}])
    stats = runner.purge_failed(path, max_retries=3)
    assert stats["purged"] == 1                      # 2 < 3 なのでまだ再試行
    stored = json.loads((tmp_path / runner.PURGE_COUNTS_FILE).read_text(encoding="utf-8"))
    assert stored["retries"] == {"hard": 3}


def test_gap_note_scales_with_n(tmp_path, monkeypatch):
    """差の注記が n に応じて変わる（n=2,788 で「n が小さいので」と言わない）。"""
    monkeypatch.setattr(baselines, "discover_models", lambda: {})

    def render(n_questions, ours_right):
        recs, bares = [], []
        for i in range(n_questions):
            base = {"question_name": f"q{i}", "topic": "general_chemistry",
                    "metric_kind": "mcq", "score_map": {"A": 1, "B": 0},
                    "provider": "claude", "model": "m"}
            recs.append({**base, "answer": "A" if i < ours_right else "B"})
            bares.append({**base, "mode": "bare", "answer": "A"})
        report.write_report(score.score_records(recs), tmp_path, "t",
                            compare_records=score.score_records(bares),
                            compare_label="b")
        return (tmp_path / "report.md").read_text(encoding="utf-8")

    small = render(70, 69)          # 1 問だけ負け
    assert "傾向として読める差ではない" in small

    large = render(2000, 1986)      # 14 問負け = 0.7% 差
    assert "14 問ぶん" in large
    assert "harness による改善も悪化も" in large
    assert "n が小さいので" not in large
