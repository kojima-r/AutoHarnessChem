"""ChemBench で ahc を評価する CLI。

  # トピック一覧と問題数（データ取得は不要。公式クローンだけで動く）
  python -m benchmarks_chembench.evaluate topics

  # 採点が公式と一致するかの確認（公開 report で照合）
  python -m benchmarks_chembench.evaluate validate

  # 小さく試す（各トピック 2 問、claude で実行 → 採点 → レポート）
  python -m benchmarks_chembench.evaluate run --label smoke --limit-per-topic 2

  # harness なし（素の LLM・ツールなし 1 往復）の同じ問題での成績
  python -m benchmarks_chembench.evaluate run --label smoke-bare --bare \
      --limit-per-topic 2

  # 採点だけやり直す / 素の LLM を比較列として並べる
  python -m benchmarks_chembench.evaluate score --label smoke --compare-label smoke-bare

出力は benchmarks_chembench/results/<label>/ に
  records.jsonl（実行の生ログ・再開に使う） / scored.jsonl（採点付き）
  metrics.json（集計） / report.md（レポート）
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks_chembench import bare, dataset, report, runner, score, validate  # noqa: E402
from benchmarks_chembench.catalog import METRIC_KINDS, TOPICS, load_catalog  # noqa: E402

RESULTS_DIR = Path(__file__).parent / "results"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks_chembench.evaluate",
        description="ChemBench で AutoHarnessChem (ahc) を評価する")
    parser.add_argument("--config", help="harness の設定 yaml（config/default.yaml へマージ）")
    sub = parser.add_subparsers(dest="command", required=True)

    p_topics = sub.add_parser("topics", help="トピックと問題数を表示する")
    p_topics.add_argument("--reference-model", dest="reference_model",
                          default=dataset.DEFAULT_REFERENCE_MODEL,
                          help="出題を読み出す公開 report（既定 gpt-4o）")

    p_validate = sub.add_parser(
        "validate", help="本実装の採点が公式 ChemBench と一致するか確認する")
    p_validate.add_argument("--model", action="append", dest="models",
                            help="照合に使うモデル（複数指定可。既定は代表 3 つ）")
    p_validate.add_argument("--limit", type=int, help="1 モデルあたりの問題数上限")

    p_verify = sub.add_parser(
        "verify-hf", help="出題が HuggingFace 版データセットと一致するか照合する")
    _add_selection_args(p_verify)
    p_verify.add_argument("--force", action="store_true", help="parquet を再取得する")

    p_run = sub.add_parser("run", help="ahc に解かせて採点する")
    _add_selection_args(p_run)
    p_run.add_argument("--label", default="chembench", help="結果の保存先名（results/<label>/）")
    p_run.add_argument("--provider", action="append", dest="providers",
                       choices=["deepagents", "claude", "openai"],
                       help="複数指定で SDK 横断比較（既定は config の provider）")
    p_run.add_argument("--concurrency", type=int, default=2, help="同時に走らせる問題数")
    p_run.add_argument("--item-timeout", type=int, default=600, dest="item_timeout",
                       help="1 問の実時間上限（秒。到達したら打ち切って次へ）")
    p_run.add_argument("--max-replans", type=int, default=1, dest="max_replans",
                       help="答えファイルが無い場合の再試行回数")
    p_run.add_argument("--cleanup-workspaces", action="store_true",
                       help="採点に使った後 workspace を削除する（大量実行時）")
    p_run.add_argument("--overwrite", action="store_true",
                       help="records.jsonl を作り直す（既定は未実行分だけ追加）")
    p_run.add_argument("--no-score", action="store_true", dest="no_score",
                       help="実行のみ（採点は後で score サブコマンドで行う）")
    p_run.add_argument("--dry-run", action="store_true", dest="dry_run",
                       help="対象問題数だけ表示して終了する")
    p_run.add_argument("--bare", action="store_true",
                       help="harness を通さず素の LLM に 1 ターンで解かせる"
                            "（ツールなし。文献と同条件のベースライン用）")
    p_run.add_argument("--model", help="--bare で使うモデル名（既定は SDK の既定モデル）")
    p_run.add_argument("--compare-label", dest="compare_label",
                       help="採点時に比較列として並べる別 label")
    p_run.add_argument("--retry-failed", action="store_true", dest="retry_failed",
                       help="再開前に「答えが取れていない record」を落として再実行対象に"
                            "戻す（枠切れをまたぐ長期実行では必須）")
    p_run.add_argument("--max-retries", type=int, default=3, dest="max_retries",
                       help="--retry-failed で同じ問題を再実行する上限（既定 3）")

    p_purge = sub.add_parser(
        "purge", help="答えが取れていない record を落として再実行対象に戻す")
    p_purge.add_argument("--label", default="chembench")
    p_purge.add_argument("--max-retries", type=int, default=3, dest="max_retries")
    p_purge.add_argument("--max-not-attempted", type=int, default=12,
                         dest="max_not_attempted",
                         help="「速い未回答」（枠切れ / 即時拒否）を何回まで見逃すか。"
                              "上限に達したら確定失敗として残す（無限ループ防止）")

    p_score = sub.add_parser("score", help="既存の records.jsonl を採点し直す")
    p_score.add_argument("--label", default="chembench")
    p_score.add_argument("--compare-label", dest="compare_label",
                         help="別の label の結果を比較列として並べる"
                              "（例: 素の LLM ベースラインの label）")
    return parser


def _add_selection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--topic", action="append", dest="topics", choices=list(TOPICS),
                        help="トピックで絞る（複数指定可）")
    parser.add_argument("--metric-kind", action="append", dest="metric_kinds",
                        choices=list(METRIC_KINDS), help="mcq か numeric で絞る")
    parser.add_argument("--requires", action="append", dest="requires",
                        help="要求能力で絞る（例 Calculation）")
    parser.add_argument("--difficulty", action="append", dest="difficulties",
                        help="難易度で絞る（difficulty-basic / difficulty-advanced）")
    parser.add_argument("--human-subset", action="store_true", dest="human_subset",
                        help="人間が解いた部分集合だけを対象にする")
    parser.add_argument("--limit-per-topic", type=int, default=5, dest="limit_per_topic",
                        help="トピックごとの出題数（0 で全件）")
    parser.add_argument("--seed", type=int, default=0, help="サンプリングの seed")
    parser.add_argument("--reference-model", dest="reference_model",
                        default=dataset.DEFAULT_REFERENCE_MODEL,
                        help="出題を読み出す公開 report（既定 gpt-4o）")


def _load_selected_items(args) -> list:
    return dataset.load_items(
        reference_model=args.reference_model,
        topics=args.topics,
        metric_kinds=args.metric_kinds,
        requires=args.requires,
        difficulties=args.difficulties,
        human_subset_only=args.human_subset,
        limit_per_topic=args.limit_per_topic or None,
        seed=args.seed,
    )


def cmd_topics(args) -> int:
    catalog = load_catalog()
    items = dataset.load_items(reference_model=args.reference_model)
    print(f"出題元: {args.reference_model} "
          f"（使えるもの: {', '.join(dataset.available_reference_models())}）\n")
    print(f"{'topic':22} {'n':>5} {'mcq':>5} {'num':>5} {'human':>6} {'task_type':10} name")
    for topic in catalog:
        rows = [i for i in items if i.topic == topic.id]
        mcq = sum(1 for i in rows if i.metric_kind == "mcq")
        human = sum(1 for i in rows if i.in_human_subset)
        print(f"{topic.id:22} {len(rows):5} {mcq:5} {len(rows) - mcq:5} {human:6} "
              f"{topic.ahc_task_type:10} {topic.name}")
    mcq = sum(1 for i in items if i.metric_kind == "mcq")
    human = sum(1 for i in items if i.in_human_subset)
    print(f"\n{'合計':22} {len(items):5} {mcq:5} {len(items) - mcq:5} {human:6}")
    return 0


def cmd_validate(args) -> int:
    models = args.models or ["gpt-4o", "claude3.5", "random_baseline"]
    result = validate.validate(models, limit=args.limit)
    print(f"{'model':24} {'compared':>9} {'agree':>7} {'MCQ':>18} {'numeric':>18}")
    for entry in result["models"]:
        print(f"{entry['model']:24} {entry['compared']:9} "
              f"{_pct(entry['agreement']):>7} "
              f"{entry['mcq_agree']}/{entry['mcq']} ({_pct(entry['mcq_agreement'])})".ljust(0)
              + f"  {entry['numeric_agree']}/{entry['numeric']} "
                f"({_pct(entry['numeric_agreement'])})")
        if entry["refusal"] or entry["unresolved"]:
            print(f"{'':24} 除外: refusal {entry['refusal']} / "
                  f"選択肢と正解の対応が復元できない {entry['unresolved']}")
        for bad in entry["disagreements"][:3]:
            print(f"{'':24} 不一致 {bad['question_name']} "
                  f"(本実装 {bad['ours']} / 公式 {bad['official']})")
    print(f"\n全体: {result['agree']}/{result['compared']} "
          f"({_pct(result['agreement'])}) が一致")
    return 0


def _pct(value) -> str:
    return "-" if value is None else f"{value * 100:.2f}%"


def cmd_verify_hf(args) -> int:
    items = _load_selected_items(args)
    result = dataset.verify_huggingface(items, force=args.force)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    ok = result["matched"] == result["n_items"]
    print("\n出題は HuggingFace 版と一致しています" if ok
          else f"\n{result['n_unmatched']} 問が一致しませんでした")
    return 0 if ok else 1


def cmd_run(args, config) -> int:
    items = _load_selected_items(args)
    if not items:
        print("[chembench] 対象の問題がありません（--topic などの指定を確認してください）")
        return 1
    # bare モードは claude-agent-sdk 直叩きなので、config の provider（deepagents 等）を
    # そのまま名乗らせるとレポートで「どの SDK で測ったか」を誤らせる
    default_provider = "claude" if getattr(args, "bare", False) else config.runtime.provider
    providers = tuple(args.providers or [default_provider])
    topics = sorted({item.topic for item in items})
    print(f"[chembench] {len(items)} 問 / {len(topics)} トピック / providers={providers}")
    if args.dry_run:
        for topic in topics:
            rows = [i for i in items if i.topic == topic]
            mcq = sum(1 for i in rows if i.metric_kind == "mcq")
            print(f"  {topic:22} {len(rows):4} 問（選択肢 {mcq} / 数値 {len(rows) - mcq}）")
        return 0

    out_dir = RESULTS_DIR / args.label
    records_path = out_dir / "records.jsonl"
    if getattr(args, "retry_failed", False):
        stats = runner.purge_failed(records_path, max_retries=args.max_retries)
        print(f"[chembench] 再実行対象へ戻した: {stats['purged']} 件 / "
              f"確定失敗として残した: {stats['given_up']} 件 / "
              f"記録済み: {stats['kept']} 件")
    runner_config = runner.RunnerConfig(
        label=args.label, providers=providers, concurrency=args.concurrency,
        item_timeout_sec=args.item_timeout, max_replans=args.max_replans,
        cleanup_workspaces=args.cleanup_workspaces, overwrite=args.overwrite)
    if getattr(args, "bare", False):
        # 素の LLM ベースライン（ツールも Verifier も再計画も使わない）
        print("[chembench] bare モード: ツールなし・1 往復で解かせます")
        asyncio.run(bare.run_items_bare(items, runner_config, records_path,
                                        model=getattr(args, "model", None)))
    else:
        asyncio.run(runner.run_items(config, items, runner_config, records_path))

    if args.no_score:
        print(f"[chembench] records: {records_path}（採点は score サブコマンドで）")
        return 0
    return _score_and_report(args, out_dir, records_path)


def cmd_purge(args) -> int:
    records_path = RESULTS_DIR / args.label / "records.jsonl"
    if not records_path.exists():
        print(f"[chembench] {records_path} がありません")
        return 1
    stats = runner.purge_failed(records_path, max_retries=args.max_retries,
                                not_attempted_cap=args.max_not_attempted)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"\n{stats['purged']} 件を再実行対象に戻しました"
          f"（同じ label で run し直すと解き直します）")
    return 0


def cmd_score(args) -> int:
    out_dir = RESULTS_DIR / args.label
    records_path = out_dir / "records.jsonl"
    if not records_path.exists():
        print(f"[chembench] {records_path} がありません（先に run してください）")
        return 1
    return _score_and_report(args, out_dir, records_path)


def _score_and_report(args, out_dir: Path, records_path: Path) -> int:
    records = runner.read_records(records_path)
    if not records:
        print("[chembench] 採点対象のレコードがありません")
        return 1
    scored = score.score_records(records)
    compare_records = None
    compare_label = getattr(args, "compare_label", None)
    if compare_label:
        compare_path = RESULTS_DIR / compare_label / "records.jsonl"
        if not compare_path.exists():
            print(f"[chembench] {compare_path} がありません（--compare-label を確認）")
            return 1
        compare_records = score.score_records(runner.read_records(compare_path))
        print(f"[chembench] 比較列: {compare_label}（{len(compare_records)} 件）")
    summary = report.write_report(scored, out_dir, args.label,
                                  compare_records=compare_records,
                                  compare_label=compare_label)
    _print_summary(summary)
    print(f"\n[chembench] report: {out_dir / 'report.md'}")
    print(f"[chembench] metrics: {out_dir / 'metrics.json'}")
    return 0


def _print_summary(summary: dict) -> None:
    for provider, block in summary["providers"].items():
        overall = block["overall"]
        compared = block.get("baselines") or {}
        print(f"\n=== {provider}: 正答率 = {overall['score']} "
              f"（{overall['scored']}/{overall['n']} 問）===")
        other = compared.get("other") or {}
        if other.get("accuracy") is not None:
            print(f"  素のLLM（{other['label']}）: {other['accuracy']} "
                  f"（同一 {other['n']} 問）")
        top = list((compared.get("systems") or {}).items())[:3]
        for model_id, system in top:
            print(f"  文献 {system['name']:34} {system['accuracy']} "
                  f"（n={system['n']}）")
        for topic, group in (block.get("topics") or {}).items():
            print(f"    {topic:22} n={group['n']:4} score={group['score']} "
                  f"answered={group['answered_rate']}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "topics":
        return cmd_topics(args)
    if args.command == "validate":
        return cmd_validate(args)
    if args.command == "verify-hf":
        return cmd_verify_hf(args)
    if args.command == "purge":
        return cmd_purge(args)
    from harness.config import load_config
    config = load_config(args.config)
    if args.command == "run":
        return cmd_run(args, config)
    if args.command == "score":
        return cmd_score(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
