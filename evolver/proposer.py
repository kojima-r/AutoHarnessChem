"""改善候補生成。

失敗分類から Skill (SKILL.md) への追記案を生成し、unified diff として
evolver/proposals/<id>/ に出力する。自動適用はしない（specification.md §10-8）。
LLM が使える場合はより具体的な提案文を生成できるが、既定はテンプレートベース。
"""
from __future__ import annotations

import difflib
import json
from collections import Counter
from pathlib import Path

from pydantic import BaseModel, Field

from evolver.analyzer import AnalysisReport
from evolver.guard import assert_allowed
from schemas import new_id

# category → SKILL.md の Recovery procedure へ追記するガイダンス
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

MIN_OCCURRENCES = 2  # これ未満の失敗カテゴリには提案を出さない


class Proposal(BaseModel):
    proposal_id: str = Field(default_factory=lambda: new_id("prop"))
    target_file: str
    category: str
    n_failures: int
    rationale: str
    diff: str


def propose(analysis: AnalysisReport, repo_root: Path, skills_dir: Path) -> list[Proposal]:
    proposals: list[Proposal] = []
    by_key = Counter((f.category, f.skill_hint) for f in analysis.failures if f.skill_hint)

    for (category, skill_name), count in by_key.most_common():
        if count < MIN_OCCURRENCES or category not in _GUIDANCE_TEMPLATES:
            continue
        skill_md = Path(skills_dir) / skill_name / "SKILL.md"
        if not skill_md.exists():
            continue
        rel_path = skill_md.relative_to(repo_root)
        assert_allowed([rel_path], repo_root)

        original = skill_md.read_text(encoding="utf-8")
        marker = f"[evolver候補:{category}]"
        if marker in original:
            continue  # 同一カテゴリの提案が適用済みなら重複させない
        guidance = _GUIDANCE_TEMPLATES[category].format(n=count).replace(
            "[evolver候補]", marker
        )
        modified = _append_to_recovery(original, guidance)
        diff = "".join(difflib.unified_diff(
            original.splitlines(keepends=True),
            modified.splitlines(keepends=True),
            fromfile=f"a/{rel_path}", tofile=f"b/{rel_path}",
        ))
        proposals.append(Proposal(
            target_file=str(rel_path),
            category=category,
            n_failures=count,
            rationale=(f"直近 {analysis.n_runs} run の trace で `{category}` が {count} 件発生。"
                       f"Skill `{skill_name}` の Recovery procedure に予防ガイダンスを追記する。"),
            diff=diff,
        ))
    return proposals


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
            f"- category: {proposal.category} ({proposal.n_failures} failures)\n\n"
            f"{proposal.rationale}\n\n適用方法（人手レビュー後）:\n\n"
            f"```bash\ngit apply {dest / 'patch.diff'}\n```\n",
            encoding="utf-8",
        )
        saved.append(dest)
    return saved
