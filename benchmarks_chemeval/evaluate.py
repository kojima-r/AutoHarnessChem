"""ChemEval で ahc を評価する CLI。

  # データ取得（HuggingFace の parquet → JSONL、初回のみ）
  python -m benchmarks_chemeval.evaluate prepare

  # タスク一覧（id / metric / task_type を確認する）
  python -m benchmarks_chemeval.evaluate tasks

  # 小さく試す（各タスク 2 問、0-shot、claude で実行 → 採点 → レポート）
  python -m benchmarks_chemeval.evaluate run --label smoke --limit-per-task 2

  # レベルを絞って本番評価（judge も使う）
  python -m benchmarks_chemeval.evaluate run --label l4 \
      --level scientific_knowledge_deduction --limit-per-task 20 --judge auto

  # run を回さず採点だけやり直す（rdkit / judge の設定を変えたとき）
  python -m benchmarks_chemeval.evaluate score --label l4 --judge auto

出力は benchmarks_chemeval/results/<label>/ に
  records.jsonl（実行の生ログ・再開に使う） / scored.jsonl（採点付き）
  metrics.json（集計） / report.md（レポート） / judge_cache.json
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks_chemeval import dataset, report, runner, score  # noqa: E402
from benchmarks_chemeval.catalog import LEVELS, load_catalog  # noqa: E402
from benchmarks_chemeval.judge import Judge  # noqa: E402
from benchmarks_chemeval.metrics import METRICS  # noqa: E402

RESULTS_DIR = Path(__file__).parent / "results"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks_chemeval.evaluate",
        description="ChemEval で AutoHarnessChem (ahc) を評価する")
    parser.add_argument("--config", help="harness の設定 yaml（config/default.yaml へマージ）")
    sub = parser.add_subparsers(dest="command", required=True)

    p_prepare = sub.add_parser("prepare", help="ChemEval データを取得して JSONL へ変換する")
    p_prepare.add_argument("--split", action="append", choices=[*dataset.SPLITS, "all"],
                           help="既定は text（multimodal は画像も展開する）")
    p_prepare.add_argument("--force", action="store_true", help="再ダウンロード・再変換する")

    p_tasks = sub.add_parser("tasks", help="タスクカタログを表示する")
    p_tasks.add_argument("--split", action="append", choices=list(dataset.SPLITS))
    p_tasks.add_argument("--level", action="append", choices=list(LEVELS))
    p_tasks.add_argument("--metric", action="append", choices=sorted(METRICS))

    p_run = sub.add_parser("run", help="ahc に解かせて採点する")
    _add_selection_args(p_run)
    p_run.add_argument("--label", default="chemeval", help="結果の保存先名（results/<label>/）")
    p_run.add_argument("--provider", action="append", dest="providers",
                       choices=["deepagents", "claude", "openai"],
                       help="複数指定で SDK 横断比較（既定は config の provider）")
    p_run.add_argument("--concurrency", type=int, default=2, help="同時に走らせる問題数")
    p_run.add_argument("--item-timeout", type=int, default=900, dest="item_timeout",
                       help="1 問の実時間上限（秒。到達したら打ち切って次へ）")
    p_run.add_argument("--max-replans", type=int, default=1, dest="max_replans",
                       help="答えファイルが無い場合の再試行回数")
    p_run.add_argument("--cleanup-workspaces", action="store_true",
                       help="採点に使った後 workspace を削除する（大量実行時）")
    p_run.add_argument("--overwrite", action="store_true",
                       help="records.jsonl を作り直す（既定は未実行分だけ追加）")
    p_run.add_argument("--judge", default="none",
                       help="自由記述タスクの LLM 採点: none | auto | claude | anthropic")
    p_run.add_argument("--judge-model", dest="judge_model", help="judge のモデル名")
    p_run.add_argument("--no-chem", action="store_true", dest="no_chem",
                       help="rdkit による分子系の採点をしない（文字列一致に退避）")
    p_run.add_argument("--no-score", action="store_true", dest="no_score",
                       help="実行のみ（採点は後で score サブコマンドで行う）")
    p_run.add_argument("--dry-run", action="store_true", dest="dry_run",
                       help="対象問題数だけ表示して終了する")

    p_score = sub.add_parser("score", help="既存の records.jsonl を採点し直す")
    p_score.add_argument("--label", default="chemeval")
    p_score.add_argument("--judge", default="none")
    p_score.add_argument("--judge-model", dest="judge_model")
    p_score.add_argument("--no-chem", action="store_true", dest="no_chem")
    return parser


def _add_selection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--split", default="text", choices=list(dataset.SPLITS))
    parser.add_argument("--task", action="append", dest="task_ids", help="タスク id で絞る")
    parser.add_argument("--level", action="append", dest="levels", choices=list(LEVELS))
    parser.add_argument("--dimension", action="append", dest="dimensions")
    parser.add_argument("--metric", action="append", dest="metrics", choices=sorted(METRICS))
    parser.add_argument("--shot", type=int, default=0, choices=[0, 3],
                        help="0=0-shot（既定） / 3=3-shot")
    parser.add_argument("--all-shots", action="store_true", dest="all_shots",
                        help="0-shot と 3-shot の両方を対象にする")
    parser.add_argument("--limit-per-task", type=int, default=5, dest="limit_per_task",
                        help="タスクごとの出題数（0 で全件）")
    parser.add_argument("--seed", type=int, default=0, help="サンプリングの seed")


def cmd_prepare(args) -> int:
    splits = args.split or ["text"]
    if "all" in splits:
        splits = list(dataset.SPLITS)
    paths = dataset.prepare(splits, force=args.force)
    for split, path in paths.items():
        rows = sum(1 for _ in path.open(encoding="utf-8"))
        print(f"[chemeval] {split}: {path} ({rows} 行)")
    return 0


def cmd_tasks(args) -> int:
    catalog = load_catalog()
    selected = catalog.select(levels=args.level, metrics=args.metric, splits=args.split)
    print(f"{'id':38} {'split':11} {'metric':16} {'ahc_task_type':24} level / dimension")
    for task in selected:
        print(f"{task.id:38} {task.split:11} {task.metric:16} {task.ahc_task_type:24} "
              f"{task.level} / {task.dimension}")
    print(f"\n{len(selected)} / {len(catalog)} タスク")
    return 0


def _load_selected_items(args) -> list:
    return dataset.load_items(
        args.split,
        task_ids=args.task_ids,
        levels=args.levels,
        dimensions=args.dimensions,
        metrics=args.metrics,
        shot=None if args.all_shots else args.shot,
        limit_per_task=args.limit_per_task or None,
        seed=args.seed,
    )


def _judge(args, out_dir: Path) -> Judge | None:
    if args.judge in ("none", "off"):
        return None
    judge = Judge(args.judge, model=args.judge_model,
                  cache_path=out_dir / "judge_cache.json")
    if not judge.available:
        print("[chemeval] judge に使える SDK が見つかりません "
              "（自由記述タスクは score=None のまま集計外になります）")
    return judge


def _print_summary(summary: dict) -> None:
    for provider, block in summary["providers"].items():
        overall = block["overall"]
        print(f"\n=== {provider}: macro task score = {overall['macro_task_score']} "
              f"({overall['scored_tasks']} タスク) ===")
        for level, level_block in block["levels"].items():
            print(f"  {level:34} n={level_block['n']:4} score={level_block['score']} "
                  f"answered={level_block['answered_rate']} "
                  f"verifier_passed={level_block['harness_passed_rate']}")


def cmd_run(args, config) -> int:
    items = _load_selected_items(args)
    if not items:
        print("[chemeval] 対象の問題がありません（--task / --level の指定を確認してください）")
        return 1
    tasks = sorted({item.task_id for item in items})
    providers = tuple(args.providers or [config.runtime.provider])
    print(f"[chemeval] {len(items)} 問 / {len(tasks)} タスク / providers={providers}")
    if args.dry_run:
        for task_id in tasks:
            count = sum(1 for item in items if item.task_id == task_id)
            print(f"  {task_id:38} {count} 問")
        return 0

    out_dir = RESULTS_DIR / args.label
    records_path = out_dir / "records.jsonl"
    runner_config = runner.RunnerConfig(
        label=args.label, providers=providers, concurrency=args.concurrency,
        item_timeout_sec=args.item_timeout, max_replans=args.max_replans,
        cleanup_workspaces=args.cleanup_workspaces, overwrite=args.overwrite)
    asyncio.run(runner.run_items(config, items, runner_config, records_path))

    if args.no_score:
        print(f"[chemeval] records: {records_path}（採点は score サブコマンドで）")
        return 0
    return _score_and_report(args, config, out_dir, records_path)


def cmd_score(args, config) -> int:
    out_dir = RESULTS_DIR / args.label
    records_path = out_dir / "records.jsonl"
    if not records_path.exists():
        print(f"[chemeval] {records_path} がありません（先に run してください）")
        return 1
    return _score_and_report(args, config, out_dir, records_path)


def _score_and_report(args, config, out_dir: Path, records_path: Path) -> int:
    records = runner.read_records(records_path)
    if not records:
        print("[chemeval] 採点対象のレコードがありません")
        return 1
    judge = _judge(args, out_dir)
    scored = asyncio.run(score.score_records(
        records, config=config, judge=judge,
        workspace=config.paths.workspaces / f"chemeval-scoring-{args.label}",
        use_chem=not args.no_chem))
    summary = report.write_report(scored, out_dir, args.label)
    _print_summary(summary)
    print(f"\n[chemeval] report: {out_dir / 'report.md'}")
    print(f"[chemeval] metrics: {out_dir / 'metrics.json'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    from harness.config import load_config

    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        return cmd_prepare(args)
    if args.command == "tasks":
        return cmd_tasks(args)
    config = load_config(args.config)
    if args.command == "run":
        return cmd_run(args, config)
    if args.command == "score":
        return cmd_score(args, config)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
