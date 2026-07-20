"""改善候補生成。

trace 解析から Skill (SKILL.md) への追記案を生成し、unified diff として
evolver/proposals/<id>/ に出力する。自動適用はしない（人手レビュー前提）。

2段階で提案する:
  1. リカバリベース（優先）: 失敗→成功が観測されたツールについて、実際にどの引数を
     どう変えたら通ったかを、そのツールを required_tools に持つ Skill の
     Recovery procedure へ具体的に恒久化する。汎用文ではなく再現可能な手順を書く。
  2. 汎用テンプレート（フォールバック）: リカバリが観測されなかった頻出失敗にのみ、
     予防的な一般ガイダンスを追記する。
"""
from __future__ import annotations

import difflib
from collections import Counter, defaultdict
from pathlib import Path

from pydantic import BaseModel, Field

from evolver.analyzer import AnalysisReport, RecoveryRecord, _SKILL_BY_CATEGORY
from evolver.guard import assert_allowed
from harness.skill_registry import SkillRegistry
from schemas import new_id

# category → SKILL.md の Recovery procedure へ追記する汎用ガイダンス（フォールバック）
_GUIDANCE_TEMPLATES = {
    "timeout": "- [evolver候補] timeout が頻発しています（{n}件）。計算前に分子サイズ（重原子数）を確認し、"
               "重原子数 > 20 なら最初から基底を STO-3G に落としてください。",
    "scf_failed": "- [evolver候補] SCF 収束失敗が頻発しています（{n}件）。最初の実行から "
                  "`mf.level_shift=0.2` を設定することを検討してください。",
    "verification_missing_output": "- [evolver候補] 期待出力の欠落が頻発しています（{n}件）。"
                                   "最終報告の前に必ず期待出力ファイルの存在を `inspect_artifact` で確認してください。",
    "missing_dependency": "- [evolver候補] 依存パッケージ不足が頻発しています（{n}件）。"
                          "コード生成前に import 可否を1行スクリプトで確認してください。",
    "scientific_warning": "- [evolver候補] 科学的警告が頻発しています（{n}件）。単位変換と値域チェックを"
                          "計算直後に自分で行ってから報告してください。",
    "tool_error": "- [evolver候補] ツール実行エラーが頻発しています（{n}件）。引数のスキーマ"
                  "（必須キー・型）をツール説明で再確認してから呼び出してください。",
}

MIN_OCCURRENCES = 2       # 汎用テンプレートを出す最小件数
RECOVERY_MIN_OCCURRENCES = 1  # 具体的リカバリは1件でも恒久化する価値がある
_MAX_VALUE_LEN = 80


class Proposal(BaseModel):
    proposal_id: str = Field(default_factory=lambda: new_id("prop"))
    target_file: str
    category: str
    n_failures: int
    rationale: str
    diff: str


def propose(analysis: AnalysisReport, repo_root: Path, skills_dir: Path) -> list[Proposal]:
    registry = SkillRegistry(skills_dir)
    tool_owner = _tool_owner_map(registry)
    proposals: list[Proposal] = []
    # 具体的提案でカバーした (skill, category) は汎用提案から除外する
    covered: set[tuple[str, str]] = set()

    # --- 1) リカバリベースの具体的提案 ---------------------------------
    grouped: dict[tuple[str, str, str], list[RecoveryRecord]] = defaultdict(list)
    for rec in analysis.recoveries:
        signature = "+".join(rec.changed_keys) if rec.changed_keys else "retry"
        skill_name = (tool_owner.get(rec.tool)
                      or _SKILL_BY_CATEGORY.get(rec.error_type)
                      or "execution-recovery")
        grouped[(skill_name, rec.tool, signature)].append(rec)

    for (skill_name, tool, signature), recs in sorted(grouped.items()):
        if len(recs) < RECOVERY_MIN_OCCURRENCES:
            continue
        marker = f"[evolver候補:recovery:{tool}:{signature}]"
        proposal = _build_proposal(
            repo_root, skills_dir, skill_name, marker,
            guidance=_recovery_guidance(marker, tool, recs),
            category=f"recovery:{recs[0].error_type}",
            n=len(recs),
            rationale=(f"直近 {analysis.n_runs} run で `{tool}` の失敗→成功リカバリを "
                       f"{len(recs)} 回検出。その具体的な回復方法を Skill `{skill_name}` に"
                       "恒久化し、次回以降の再試行を不要にする。"),
        )
        if proposal is not None:
            proposals.append(proposal)
            covered.add((skill_name, recs[0].error_type))

    # --- 2) 汎用テンプレート（リカバリ未観測の頻出失敗のみ） -----------
    by_key = Counter((f.category, f.skill_hint) for f in analysis.failures if f.skill_hint)
    for (category, skill_name), count in by_key.most_common():
        if count < MIN_OCCURRENCES or category not in _GUIDANCE_TEMPLATES:
            continue
        if (skill_name, category) in covered:
            continue  # 具体的リカバリで既に対処済み
        marker = f"[evolver候補:{category}]"
        proposal = _build_proposal(
            repo_root, skills_dir, skill_name, marker,
            guidance=_GUIDANCE_TEMPLATES[category].format(n=count).replace("[evolver候補]", marker),
            category=category, n=count,
            rationale=(f"直近 {analysis.n_runs} run の trace で `{category}` が {count} 件発生"
                       "（具体的なリカバリは観測されず）。Skill "
                       f"`{skill_name}` に予防ガイダンスを追記する。"),
        )
        if proposal is not None:
            proposals.append(proposal)
    return proposals


def _tool_owner_map(registry: SkillRegistry) -> dict[str, str]:
    """ツール名 → それを required_tools に持つ Skill 名。ドメイン Skill を優先する。"""
    owner: dict[str, str] = {}
    for name in sorted(registry.names()):
        skill = registry.get(name)
        for tool in skill.required_tools:
            current = owner.get(tool)
            if current is None:
                owner[tool] = name
            elif not registry.get(current).task_types and skill.task_types:
                # task_types を持つドメイン Skill を、常時ロード系より優先
                owner[tool] = name
    return owner


def _short(value) -> str:
    text = "(未指定)" if value is None else str(value)
    return text if len(text) <= _MAX_VALUE_LEN else text[:_MAX_VALUE_LEN] + "…"


def _recovery_guidance(marker: str, tool: str, recs: list[RecoveryRecord]) -> str:
    rec = recs[0]
    n = len(recs)
    if rec.changed_keys:
        changes = "; ".join(
            f"`{key}`: `{_short(rec.failed_arguments.get(key))}` → "
            f"`{_short(rec.recovered_arguments.get(key))}`"
            for key in rec.changed_keys
        )
        return (
            f"- {marker} 過去に `{tool}` が「{rec.failure_summary}」"
            f"(error_type={rec.error_type}) で失敗したが、{changes} と変更して成功した"
            f"（{n}回観測）。同種の失敗時は最初からこの変更を適用すること。"
        )
    return (
        f"- {marker} `{tool}` の {rec.error_type}（「{rec.failure_summary}」）は "
        f"同一引数の再試行で回復した（{n}回観測）。恒久的な設定変更は不要だが、"
        "同じ失敗時はまず同一引数で1回再試行してよい。"
    )


def _build_proposal(repo_root: Path, skills_dir: Path, skill_name: str, marker: str,
                    guidance: str, category: str, n: int, rationale: str) -> Proposal | None:
    skill_md = Path(skills_dir) / skill_name / "SKILL.md"
    if not skill_md.exists():
        return None
    rel_path = skill_md.relative_to(repo_root)
    assert_allowed([rel_path], repo_root)
    original = skill_md.read_text(encoding="utf-8")
    if marker in original:
        return None  # 同一提案が適用済みなら重複させない
    modified = _append_to_recovery(original, guidance)
    diff = "".join(difflib.unified_diff(
        original.splitlines(keepends=True),
        modified.splitlines(keepends=True),
        fromfile=f"a/{rel_path}", tofile=f"b/{rel_path}",
    ))
    return Proposal(target_file=str(rel_path), category=category,
                    n_failures=n, rationale=rationale, diff=diff)


def _append_to_recovery(skill_md_text: str, guidance: str) -> str:
    marker = "# Recovery procedure"
    if marker in skill_md_text:
        head, _, tail = skill_md_text.partition(marker)
        return head + marker + tail.rstrip() + "\n" + guidance + "\n"
    return skill_md_text.rstrip() + f"\n\n{marker}\n\n{guidance}\n"


def save_proposals(proposals: list[Proposal], proposals_dir: Path) -> list[Path]:
    saved = []
    for proposal in proposals:
        dest = Path(proposals_dir) / proposal.proposal_id
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "patch.diff").write_text(proposal.diff, encoding="utf-8")
        (dest / "proposal.json").write_text(proposal.model_dump_json(indent=2), encoding="utf-8")
        (dest / "rationale.md").write_text(
            f"# {proposal.proposal_id}\n\n- target: `{proposal.target_file}`\n"
            f"- category: {proposal.category} ({proposal.n_failures} occurrences)\n\n"
            f"{proposal.rationale}\n\n適用方法（人手レビュー後）:\n\n"
            f"```bash\ngit apply {dest / 'patch.diff'}\n```\n",
            encoding="utf-8",
        )
        saved.append(dest)
    return saved
