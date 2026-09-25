"""CLI エントリポイント。

  ahc run "リクエスト" [--provider claude] [--input path/to.csv]
          [--on-timeout ask|extend|stop] [--extend 7200]
  ahc skills list | compile [--provider all] | lock
  ahc benchmark [--provider deepagents --provider claude] [--tag smoke]
  ahc chemeval run [--label smoke] [--limit-per-task 2] (= python -m benchmarks_chemeval.evaluate)
  ahc chembench run [--label smoke] [--limit-per-topic 2] [--bare] (= python -m benchmarks_chembench.evaluate)
  ahc verify --workspace workspaces/run-xxxx --task-type orbital_calculation
  ahc evolve analyze | propose | evaluate --proposal <id> | promote --proposal <id>
  ahc demo [--env all|pyscf|reactiont5|aizynth] [--out DIR]
  ahc api [--port 8000]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

# リポジトリ直下から `python -m app.cli` / インストール後は `ahc` で動くようにする
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.config import load_config  # noqa: E402

# examples/ のデモ: conda env 名 → スクリプト（`ahc demo` / examples/run_examples.sh 共通）
DEMOS = {
    "pyscf": "pyscf_opt_tddft_demo.py",
    "aizynth": "aizynth_demo.py",
    "reactiont5": "reactiont5_demo.py",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ahc", description="AutoHarnessChem — switchable agentic harness")
    parser.add_argument("--config", help="override config yaml (merged onto config/default.yaml)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="タスクを実行する")
    p_run.add_argument("request", help="ユーザ要求（自然言語）")
    p_run.add_argument("--provider", choices=["deepagents", "claude", "openai"])
    p_run.add_argument("--task-type", dest="task_type")
    p_run.add_argument("--input", action="append", default=[], dest="inputs",
                       help="workspace へコピーする入力ファイル（複数可）")
    p_run.add_argument("--expect", action="append", default=[], dest="expected",
                       help="期待する出力ファイル（複数可、glob可）")
    p_run.add_argument("--quiet", action="store_true",
                       help="実行過程（イベントの逐次表示）を抑制する")
    p_run.add_argument("--on-timeout", dest="on_timeout",
                       choices=["ask", "extend", "stop"],
                       help="実時間上限に達したときの動作（既定は config の設定）。"
                            "ask=延長するか対話で確認 / extend=自動延長 / stop=打ち切り")
    p_run.add_argument("--extend", type=int, dest="extend_sec",
                       help="1回の延長で足す秒数（既定は config の timeout_extension_sec）")

    p_skills = sub.add_parser("skills", help="Skill の一覧・配置・ロック")
    p_skills.add_argument("action", choices=["list", "compile", "lock", "check"])
    p_skills.add_argument("--provider", default="all",
                          choices=["all", "deepagents", "claude", "openai"])

    p_bench = sub.add_parser("benchmark", help="ベンチマークをSDK横断で実行する")
    p_bench.add_argument("--provider", action="append", dest="providers",
                         choices=["deepagents", "claude", "openai"])
    p_bench.add_argument("--tag", action="append", dest="tags")
    p_bench.add_argument("--id", action="append", dest="ids")
    p_bench.add_argument("--label", default="baseline")

    # ChemEval（benchmarks_chemeval/）での評価。引数はそのまま
    # `python -m benchmarks_chemeval.evaluate` へ渡す
    p_chemeval = sub.add_parser(
        "chemeval", help="ChemEval で評価する（prepare | tasks | run | score）")
    p_chemeval.add_argument("chemeval_args", nargs=argparse.REMAINDER,
                            help="benchmarks_chemeval.evaluate へ渡す引数")

    # ChemBench（benchmarks_chembench/）での評価。引数はそのまま
    # `python -m benchmarks_chembench.evaluate` へ渡す
    p_chembench = sub.add_parser(
        "chembench",
        help="ChemBench で評価する（topics | validate | verify-hf | run | score）")
    p_chembench.add_argument("chembench_args", nargs=argparse.REMAINDER,
                             help="benchmarks_chembench.evaluate へ渡す引数")

    p_verify = sub.add_parser("verify", help="既存 workspace を Verifier で再判定する")
    p_verify.add_argument("--workspace", required=True)
    p_verify.add_argument("--task-type", dest="task_type", default="generic")
    p_verify.add_argument("--expect", action="append", default=[], dest="expected")

    p_evolve = sub.add_parser("evolve", help="自己改善ループ（開発モードのみ）")
    p_evolve.add_argument("action", choices=["analyze", "propose", "evaluate", "promote"])
    p_evolve.add_argument("--proposal", help="evaluate/promote 対象の proposal id")
    p_evolve.add_argument("--provider", action="append", dest="providers")
    p_evolve.add_argument("--tag", action="append", dest="tags")

    p_demo = sub.add_parser(
        "demo", help="各 conda 環境のライブラリを直接使うサンプル（examples/）を実行する")
    p_demo.add_argument("--env", default="all",
                        choices=["all", *DEMOS], help="実行するデモ（既定: all）")
    p_demo.add_argument("--out", help="出力ディレクトリ（既定: examples/output/<env>）")
    p_demo.add_argument("--arg", action="append", default=[], dest="extra",
                        help="デモスクリプトへ渡す追加引数。`=` で繋ぐこと "
                             "（例 --arg=--nstates --arg=10）")

    p_api = sub.add_parser("api", help="Web UI + FastAPI サーバを起動する")
    p_api.add_argument("--port", type=int, default=8000)
    p_api.add_argument("--host", default="127.0.0.1")

    args = parser.parse_args(argv)
    config = load_config(args.config)
    if getattr(args, "on_timeout", None):
        config.runtime.on_attempt_timeout = args.on_timeout
    if getattr(args, "extend_sec", None):
        config.runtime.timeout_extension_sec = args.extend_sec

    if args.command == "run":
        return _cmd_run(config, args)
    if args.command == "skills":
        return _cmd_skills(config, args)
    if args.command == "benchmark":
        return _cmd_benchmark(config, args)
    if args.command == "chemeval":
        return _cmd_chemeval(args)
    if args.command == "chembench":
        return _cmd_chembench(args)
    if args.command == "verify":
        return _cmd_verify(config, args)
    if args.command == "evolve":
        return _cmd_evolve(config, args)
    if args.command == "demo":
        return _cmd_demo(config, args)
    if args.command == "api":
        import uvicorn
        from app.api import create_app
        print(f"Web UI: http://{args.host}:{args.port}/  (API docs: /docs)")
        uvicorn.run(create_app(config), host=args.host, port=args.port)
        return 0
    return 1


def _cmd_run(config, args) -> int:
    from adapters import AdapterUnavailable
    from harness.controller import HarnessController

    controller = HarnessController(config)
    try:
        report = _run_task(controller, args)
    except AdapterUnavailable as e:
        print(f"error: {e}", file=sys.stderr)
        print("SDK を1つ以上インストールしてください: "
              'pip install -e ".[claude]" / ".[openai]" / ".[deepagents]"', file=sys.stderr)
        return 3
    print(f"\n=== run {report.run_id} ({report.provider}) ===")
    print(f"passed={report.passed} attempts={report.attempts}")
    print(f"report: {config.paths.workspaces / report.run_id / 'report.md'}")
    print("\n" + report.final_message)
    return 0 if report.passed else 2


def _run_task(controller, args):
    return asyncio.run(controller.run(
        args.request,
        provider=args.provider,
        task_type=args.task_type,
        expected_outputs=args.expected or None,
        copy_inputs=args.inputs,
        on_event=None if args.quiet else make_event_printer(),
        on_timeout=(make_timeout_prompt(controller.config)
                    if controller.config.runtime.on_attempt_timeout == "ask" else None),
    ))


def make_timeout_prompt(config):
    """実時間上限に達したときに「さらに待つか」を端末で確認するリスナーを返す。

    数日かかる計算もあるため、既定は「待つ」。応答が取れない環境（非対話）では
    None を返して config の方針（extend / stop）に委ねる。
    """
    import os

    if not (sys.stdin and sys.stdin.isatty()) or os.environ.get("AHC_NONINTERACTIVE"):
        return None

    default_sec = config.runtime.timeout_extension_sec

    def ask(info: dict):
        hours = info["waited_sec"] / 3600
        print(f"\n[!] 試行 {info['attempt']} が {info['waited_sec']}s "
              f"({hours:.1f} 時間) 経過しました（延長 {info['extensions']} 回）。"
              f"\n    workspace: {info['workspace']}"
              f"\n    さらに待ちますか？ [Enter=+{default_sec}s / 秒数 / "
              f"h+時間 (例 12h) / d+日 (例 2d) / s=打ち切り]: ", end="", file=sys.stderr,
              flush=True)
        try:
            answer = input().strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("  → 打ち切ります", file=sys.stderr)
            return False
        if answer in ("s", "stop", "n", "no"):
            print("  → 打ち切ります", file=sys.stderr)
            return False
        seconds = _parse_duration(answer, default_sec)
        print(f"  → あと {seconds}s 待ちます", file=sys.stderr)
        return seconds

    return ask


def _parse_duration(text: str, default_sec: int) -> int:
    """'', '3600', '12h', '2d' を秒に変換する（解釈できなければ既定値）。"""
    text = (text or "").strip().lower()
    if not text:
        return default_sec
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    if text[-1] in units:
        try:
            return max(1, int(float(text[:-1]) * units[text[-1]]))
        except ValueError:
            return default_sec
    try:
        return max(1, int(float(text)))
    except ValueError:
        return default_sec


# ---------------------------------------------------------------------------
# 実行過程の逐次表示（TraceWriter リスナー）
# ---------------------------------------------------------------------------

_EVENT_COLORS = {
    "tool_call": "36",          # cyan
    "tool_result": "32",        # green
    "artifact": "33",           # yellow
    "delegation": "34",         # blue
    "error": "31",              # red
    "final": "35",              # magenta
    "reasoning_summary": "90",  # gray
}
_RESULT_MARKS = {"success": "✔", "partial": "◑", "failed": "✘", "blocked": "⛔"}


def make_event_printer(stream=None):
    """AgentEvent を1行ずつ整形して出力するリスナーを返す。"""
    import os

    out = stream or sys.stderr
    use_color = hasattr(out, "isatty") and out.isatty() and not os.environ.get("NO_COLOR")

    def paint(text: str, code: str) -> str:
        return f"\x1b[{code}m{text}\x1b[0m" if use_color else text

    def show(event) -> None:
        line = format_event(event)
        if use_color:
            line = paint(line, _EVENT_COLORS.get(event.event_type, "0"))
        print(line, file=out, flush=True)

    return show


def format_event(event) -> str:
    p = event.payload or {}
    kind = event.event_type
    if kind == "tool_call":
        arguments = json.dumps(p.get("arguments", {}), ensure_ascii=False)
        text = f"→ {p.get('tool')} {arguments[:160]}"
    elif kind == "tool_result":
        mark = _RESULT_MARKS.get(p.get("status", ""), "?")
        text = f"{mark} {p.get('tool')} [{p.get('status')}] {p.get('summary', '')}"
        if p.get("error_type"):
            text += f" (error_type={p['error_type']})"
        if p.get("stderr_tail"):
            # 全文は workspace の tool_errors.jsonl に残っている
            last = str(p["stderr_tail"]).strip().splitlines()[-1][:160]
            text += f" | stderr: {last} → {p.get('error_log', 'tool_errors.jsonl')}"
    elif kind == "artifact":
        text = f"⎘ {p.get('path', '')} ({p.get('bytes', '?')}B)"
    elif kind == "final":
        text = "最終メッセージ: " + str(p.get("text", ""))[:200].replace("\n", " ")
    elif kind == "error":
        text = str(p.get("error") or p)[:300]
    elif p.get("phase") == "start":
        text = (f"開始 provider={p.get('provider')} task_type={p.get('task_type')} "
                f"skills={','.join(p.get('skills', []))}")
    elif p.get("phase") == "attempt":
        text = f"試行 {p.get('attempt')}/{p.get('max_attempts')}"
        if p.get("repairs"):
            text += f" （修復指示 {len(p['repairs'])} 件）"
    elif p.get("phase") == "attempt_timeout_pending":
        text = (f"⏳ 試行 {p.get('attempt')} が {p.get('waited_sec')}s 経過 — "
                "さらに待つか確認します")
    elif p.get("phase") == "attempt_timeout_extended":
        text = (f"延長 +{p.get('extend_sec')}s（累計 {p.get('new_deadline_sec')}s、"
                f"{p.get('extensions', 0) + 1} 回目）")
    elif p.get("phase") == "attempt_timeout_limit":
        text = (f"延長回数の上限 {p.get('max_timeout_extensions')} に達したため打ち切ります")
    elif p.get("phase") == "verifying":
        text = f"Verifier 検査中 (attempt {p.get('attempt')})"
    elif p.get("phase") == "finished":
        text = f"終了 status={p.get('status')} attempts={p.get('attempts')}"
    elif event.actor == "verifier":
        ok = "合格" if p.get("passed") else "不合格"
        text = (f"検証{ok}: missing={len(p.get('requirements_missing', []))} "
                f"warnings={len(p.get('scientific_warnings', []))}")
    else:
        text = json.dumps(p, ensure_ascii=False)[:200]
    stamp = event.timestamp.astimezone().strftime("%H:%M:%S")
    return f"[{stamp}] {kind:<17} {event.actor:<11} {text}"


def _cmd_skills(config, args) -> int:
    from harness.skill_registry import SkillCompiler, SkillRegistry

    registry = SkillRegistry(config.paths.skills)
    if args.action == "list":
        for name in registry.names():
            skill = registry.get(name)
            print(f"{name:26s} v{skill.version:8s} [{skill.risk_level}] {skill.description}")
        return 0
    if args.action == "compile":
        compiler = SkillCompiler(registry, config.paths.root)
        targets = ["deepagents", "claude", "openai"] if args.provider == "all" else [args.provider]
        for provider in targets:
            for path in compiler.compile(provider):
                print(f"[{provider}] {path}")
        return 0
    if args.action == "lock":
        lock = registry.write_lockfile(config.paths.root / config.skill_lockfile)
        print(f"wrote {config.skill_lockfile} ({len(lock)} skills)")
        return 0
    if args.action == "check":
        drift = registry.check_lockfile(config.paths.root / config.skill_lockfile)
        if drift:
            print(f"drift detected: {drift}")
            return 2
        print("skills.lock is consistent")
        return 0
    return 1


def _cmd_benchmark(config, args) -> int:
    from benchmarks.runner import run_benchmark

    providers = args.providers or [config.runtime.provider]
    result = asyncio.run(run_benchmark(config, providers, tags=args.tags,
                                       ids=args.ids, label=args.label))
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))
    return 0


def _cmd_chemeval(args) -> int:
    """ChemEval 評価 CLI へ委譲する（`ahc --config` はそのまま引き継ぐ）。"""
    from benchmarks_chemeval.evaluate import main as chemeval_main

    forwarded = list(args.chemeval_args or [])
    if args.config:
        forwarded = ["--config", args.config, *forwarded]
    if not ({"prepare", "tasks", "run", "score"} & set(forwarded)):
        forwarded.append("tasks")   # サブコマンド未指定ならタスク一覧を出す
    return chemeval_main(forwarded)


def _cmd_chembench(args) -> int:
    """ChemBench 評価 CLI へ委譲する（`ahc --config` はそのまま引き継ぐ）。"""
    from benchmarks_chembench.evaluate import build_parser, main as chembench_main

    forwarded = list(args.chembench_args or [])
    if args.config:
        forwarded = ["--config", args.config, *forwarded]
    # サブコマンド名は委譲先のパーサから取る（ここに列挙すると追加時に取りこぼす）
    known = set()
    for action in build_parser()._subparsers._group_actions:
        known |= set(action.choices or {})
    if not (known & set(forwarded)):
        forwarded.append("topics")  # サブコマンド未指定ならトピック一覧を出す
    return chembench_main(forwarded)


def _cmd_verify(config, args) -> int:
    from harness.verifier import ScientificVerifier
    from schemas import TaskSpec

    task = TaskSpec(description="re-verification", task_type=args.task_type,
                    expected_outputs=args.expected)
    verification = ScientificVerifier().verify(task, Path(args.workspace))
    print(verification.model_dump_json(indent=2))
    return 0 if verification.passed else 2


def _cmd_demo(config, args) -> int:
    """examples/ のデモを該当 conda 環境で実行する（各環境のライブラリ直接利用の例）。"""
    import shutil
    import subprocess

    examples_dir = config.paths.root / "examples"
    if not shutil.which("conda"):
        print("conda が見つかりません。デモは各環境の python で直接実行してください: "
              f"python {examples_dir}/<script>.py", file=sys.stderr)
        return 3

    envs = list(DEMOS) if args.env == "all" else [args.env]
    failed = []
    for env in envs:
        script = examples_dir / DEMOS[env]
        out_dir = Path(args.out) if args.out else examples_dir / "output" / env
        print(f"\n=== demo: {env} ({script.name}) → {out_dir} ===", flush=True)
        completed = subprocess.run(
            ["conda", "run", "--no-capture-output", "-n", env, "python", str(script),
             "--out", str(out_dir), *args.extra],
            check=False,
        )
        if completed.returncode != 0:
            failed.append(f"{env} (exit {completed.returncode})")

    if failed:
        print(f"\n失敗したデモ: {', '.join(failed)}", file=sys.stderr)
        return 2
    print(f"\n全デモが成功しました（{len(envs)} 環境）")
    return 0


def _cmd_evolve(config, args) -> int:
    if not config.self_improvement:
        print("self_improvement is disabled (production mode). Aborting.")
        return 2

    from evolver.analyzer import analyze_traces
    from evolver.proposer import propose, save_proposals

    analysis = analyze_traces(config.paths.traces)

    if args.action == "analyze":
        print(f"runs analyzed: {analysis.n_runs}")
        print("failure counts:")
        print(json.dumps(analysis.counts, indent=2, ensure_ascii=False))
        print(f"recoveries detected: {len(analysis.recoveries)}")
        for rec in analysis.recoveries:
            change = ",".join(rec.changed_keys) or "retry-only"
            print(f"  - {rec.tool} [{rec.error_type}] recovered via: {change}")
        return 0

    if args.action == "propose":
        proposals = propose(analysis, config.paths.root, config.paths.skills)
        if not proposals:
            print("no proposals (not enough recurring failures).")
            return 0
        for dest in save_proposals(proposals, config.paths.proposals):
            print(f"proposal written: {dest}")
        print("\n改善候補は自動適用されません。patch.diff をレビューして git apply してください。")
        return 0

    if args.action in ("evaluate", "promote"):
        if not args.proposal:
            print("--proposal <id> を指定してください")
            return 1
        proposal_dir = config.paths.proposals / args.proposal
        from evolver.evaluator import evaluate
        from evolver.promoter import decide
        from evolver.proposer import Proposal

        proposal = Proposal.model_validate_json(
            (proposal_dir / "proposal.json").read_text(encoding="utf-8"))
        baseline_path = config.paths.benchmarks / "results" / "baseline.json"
        if not baseline_path.exists():
            print("baseline がありません。先に `ahc benchmark --label baseline` を実行してください。")
            return 2
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))["summary"]
        providers = args.providers or [config.runtime.provider]
        evaluation = asyncio.run(evaluate(proposal, config, providers, baseline, tags=args.tags))
        (proposal_dir / "evaluation.json").write_text(
            evaluation.model_dump_json(indent=2), encoding="utf-8")
        print(evaluation.model_dump_json(indent=2))
        if args.action == "promote":
            decision = decide(evaluation, config.promotion_gate)
            (proposal_dir / "decision.json").write_text(
                decision.model_dump_json(indent=2), encoding="utf-8")
            print(decision.model_dump_json(indent=2))
            return 0 if decision.promoted else 2
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
