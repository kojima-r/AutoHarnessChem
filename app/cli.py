"""CLI エントリポイント。

  ahc run "リクエスト" [--provider claude] [--input path/to.csv]
  ahc skills list | compile [--provider all] | lock
  ahc benchmark [--provider deepagents --provider claude] [--tag smoke]
  ahc verify --workspace workspaces/run-xxxx --task-type orbital_calculation
  ahc evolve analyze | propose | evaluate --proposal <id> | promote --proposal <id>
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

    p_verify = sub.add_parser("verify", help="既存 workspace を Verifier で再判定する")
    p_verify.add_argument("--workspace", required=True)
    p_verify.add_argument("--task-type", dest="task_type", default="generic")
    p_verify.add_argument("--expect", action="append", default=[], dest="expected")

    p_evolve = sub.add_parser("evolve", help="自己改善ループ（開発モードのみ）")
    p_evolve.add_argument("action", choices=["analyze", "propose", "evaluate", "promote"])
    p_evolve.add_argument("--proposal", help="evaluate/promote 対象の proposal id")
    p_evolve.add_argument("--provider", action="append", dest="providers")
    p_evolve.add_argument("--tag", action="append", dest="tags")

    p_api = sub.add_parser("api", help="Web UI + FastAPI サーバを起動する")
    p_api.add_argument("--port", type=int, default=8000)
    p_api.add_argument("--host", default="127.0.0.1")

    args = parser.parse_args(argv)
    config = load_config(args.config)

    if args.command == "run":
        return _cmd_run(config, args)
    if args.command == "skills":
        return _cmd_skills(config, args)
    if args.command == "benchmark":
        return _cmd_benchmark(config, args)
    if args.command == "verify":
        return _cmd_verify(config, args)
    if args.command == "evolve":
        return _cmd_evolve(config, args)
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
    ))


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


def _cmd_verify(config, args) -> int:
    from harness.verifier import ScientificVerifier
    from schemas import TaskSpec

    task = TaskSpec(description="re-verification", task_type=args.task_type,
                    expected_outputs=args.expected)
    verification = ScientificVerifier().verify(task, Path(args.workspace))
    print(verification.model_dump_json(indent=2))
    return 0 if verification.passed else 2


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
