"""ChemBench トピックカタログの読み込み（`benchmarks_chembench/topics.yaml`）。

ChemEval の `tasks.yaml` に相当するもの。ただし ChemBench は「トピック × 1 問」の
構造なので、カタログの単位は **9 トピック**であって「タスク」ではない。

各問題がどのトピックに属するかは公式リポジトリ同梱の
`chembench/scripts/classified_questions_leaderboard.csv`（question_name → topic /
requires / difficulty）で決まる。この CSV は 2,854 行あり、公開 report の 2,788 問を
**すべて**覆っている（欠けは無い）ので、突き合わせに推測は要らない。
"""
from __future__ import annotations

import csv
import functools
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).parent
CATALOG_PATH = ROOT / "topics.yaml"
CHEMBENCH_DIR = ROOT / "chembench"
CLASSIFIED_CSV = CHEMBENCH_DIR / "scripts" / "classified_questions_leaderboard.csv"
HUMAN_SUBSET_CSV = CHEMBENCH_DIR / "scripts" / "human_subset.csv"

# 表示・集計の順（問題数の多い順ではなく論文の並びに合わせる）
TOPICS = (
    "analytical_chemistry",
    "chemical_preference",
    "general_chemistry",
    "inorganic_chemistry",
    "materials_science",
    "organic_chemistry",
    "physical_chemistry",
    "technical_chemistry",
    "toxicity_and_safety",
)

# 採点方式は 2 種類しかない（chembench の metrics 列に対応）
METRIC_KINDS = ("mcq", "numeric")


def topic_id(label: str) -> str:
    """CSV の topic 表記（"Chemical Preference"）→ カタログ id（snake_case）。"""
    return str(label).strip().lower().replace(" ", "_").replace("-", "_")


@dataclass(frozen=True)
class TopicDef:
    """カタログ 1 行。"""
    id: str
    name: str
    description: str
    ahc_task_type: str


@dataclass(frozen=True)
class QuestionMeta:
    """classified_questions_leaderboard.csv の 1 行（問題ごとのメタ情報）。"""
    question_name: str
    topic: str                 # カタログ id
    requires: str              # Knowledge / Reasoning / Calculation / Intuition の組み合わせ
    difficulty: str            # difficulty-basic / difficulty-advanced / 空
    source_path: str           # 公式リポジトリ内の元 JSON のパス（出典の追跡用）


class Catalog:
    """トピック定義と、問題 → トピックの対応表。"""

    def __init__(self, topics: list[TopicDef], questions: dict[str, QuestionMeta],
                 human_subset: frozenset[str], raw: dict):
        self.topics = topics
        self.questions = questions
        self.human_subset = human_subset
        self.raw = raw
        self._by_id = {t.id: t for t in topics}

    def __len__(self) -> int:
        return len(self.topics)

    def __iter__(self):
        return iter(self.topics)

    def by_id(self, topic: str) -> TopicDef | None:
        return self._by_id.get(topic)

    def meta(self, question_name: str) -> QuestionMeta | None:
        return self.questions.get(question_name)

    def task_type(self, topic: str) -> str:
        found = self._by_id.get(topic)
        return found.ahc_task_type if found else self.raw["default_ahc_task_type"]

    def select(self, topics: list[str] | None = None) -> list[TopicDef]:
        if not topics:
            return list(self.topics)
        wanted = {topic_id(t) for t in topics}
        return [t for t in self.topics if t.id in wanted]


def _read_csv_column(path: Path, column: str) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        return [row for row in csv.DictReader(fh) if (row.get(column) or "").strip()]


def load_question_meta(path: Path | None = None) -> dict[str, QuestionMeta]:
    """question_name → メタ情報。CSV が無ければ空（トピック不明として扱う）。"""
    target = Path(path) if path else CLASSIFIED_CSV
    out: dict[str, QuestionMeta] = {}
    for row in _read_csv_column(target, "index"):
        name = row["index"].strip()
        out[name] = QuestionMeta(
            question_name=name,
            topic=topic_id(row.get("topic") or ""),
            requires=(row.get("requires") or "").strip(),
            difficulty=(row.get("difficulty") or "").strip(),
            source_path=(row.get("question") or "").strip(),
        )
    return out


def load_human_subset(path: Path | None = None) -> frozenset[str]:
    """人間との比較に使われた問題（`scripts/human_subset.csv`）の question_name。"""
    target = Path(path) if path else HUMAN_SUBSET_CSV
    return frozenset(row["index"].strip() for row in _read_csv_column(target, "index"))


@functools.lru_cache(maxsize=4)
def load_catalog(path: str | Path | None = None) -> Catalog:
    raw = yaml.safe_load(Path(path or CATALOG_PATH).read_text(encoding="utf-8"))
    default_type = raw["default_ahc_task_type"]
    overrides = raw.get("task_type_overrides") or {}
    topics = [
        TopicDef(id=entry["id"], name=entry["name"],
                 description=entry.get("description", ""),
                 ahc_task_type=overrides.get(entry["id"], default_type))
        for entry in raw["topics"]
    ]
    ids = [t.id for t in topics]
    if len(set(ids)) != len(ids):
        raise ValueError("topics.yaml: id が重複しています")
    unknown = set(ids) - set(TOPICS)
    if unknown:
        raise ValueError(f"topics.yaml: 未知のトピック {sorted(unknown)}")
    return Catalog(topics, load_question_meta(), load_human_subset(), raw)


__all__ = ["Catalog", "METRIC_KINDS", "QuestionMeta", "TOPICS", "TopicDef",
           "load_catalog", "load_human_subset", "load_question_meta", "topic_id"]
