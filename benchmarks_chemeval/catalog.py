"""ChemEval タスクカタログの読み込み（`benchmarks_chemeval/tasks.yaml`）。

ChemEval のデータ 1 件は `filename` でタスクを表す。`3shot_` 接頭・`_3shot` 接尾・
拡張子・Windows 由来の `\\` を落としたものを **key** とし、それでカタログを引く。
key は正規化後の完全一致で引き、見つからない場合だけ末尾一致（葉の名前）で救済する
（`极性_test` のように葉の名前が重複するタスクがあるため、完全一致を優先する）。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

CATALOG_PATH = Path(__file__).parent / "tasks.yaml"

# 論文の 4 レベル（集計・表示順）
LEVELS = (
    "advanced_knowledge_qa",
    "literature_understanding",
    "molecular_understanding",
    "scientific_knowledge_deduction",
)


@dataclass(frozen=True)
class TaskDef:
    """カタログ 1 行。"""
    id: str
    key: str
    name: str
    level: str
    dimension: str
    metric: str
    ahc_task_type: str
    answer_type: str
    split: str = "text"          # text | multimodal（key は split をまたいで重複しない）


def normalize_key(filename: str) -> tuple[str, int]:
    """ChemEval の `filename` → (key, shot 数)。

    >>> normalize_key("3shot_BBBP_test_3shot.json")
    ('BBBP_test', 3)
    >>> normalize_key("2.文献理解\\\\1.信息抽取\\\\10.催化类型抽取_自建\\\\催化类型抽取_test.json")[1]
    0
    """
    name = str(filename).strip().replace("\\", "/")
    shot = 0
    if name.startswith("3shot_"):
        name, shot = name[len("3shot_"):], 3
    match = re.search(r"_3shot(\.jsonl?)?$", name)
    if match:
        name, shot = name[:match.start()], 3
    name = re.sub(r"\.jsonl?$", "", name)
    return name, shot


class Catalog:
    """key / id からタスク定義を引く。"""

    def __init__(self, tasks: list[TaskDef]):
        self.tasks = tasks
        self._by_key = {t.key: t for t in tasks}
        self._by_id = {t.id: t for t in tasks}
        # 葉の名前 → タスク。重複する葉（极性_test）は救済対象から外す
        leaves: dict[str, list[TaskDef]] = {}
        for task in tasks:
            leaves.setdefault(task.key.rsplit("/", 1)[-1], []).append(task)
        self._by_leaf = {leaf: found[0] for leaf, found in leaves.items() if len(found) == 1}

    def __len__(self) -> int:
        return len(self.tasks)

    def __iter__(self):
        return iter(self.tasks)

    def by_id(self, task_id: str) -> TaskDef | None:
        return self._by_id.get(task_id)

    def lookup(self, filename: str) -> tuple[TaskDef | None, int]:
        """データの `filename` からタスク定義と shot 数を返す。"""
        key, shot = normalize_key(filename)
        task = self._by_key.get(key) or self._by_leaf.get(key.rsplit("/", 1)[-1])
        return task, shot

    def select(self, task_ids: list[str] | None = None,
               levels: list[str] | None = None,
               dimensions: list[str] | None = None,
               metrics: list[str] | None = None,
               splits: list[str] | None = None) -> list[TaskDef]:
        found = list(self.tasks)
        if splits:
            found = [t for t in found if t.split in set(splits)]
        if task_ids:
            found = [t for t in found if t.id in set(task_ids)]
        if levels:
            found = [t for t in found if t.level in set(levels)]
        if dimensions:
            found = [t for t in found if t.dimension in set(dimensions)]
        if metrics:
            found = [t for t in found if t.metric in set(metrics)]
        return found


def load_catalog(path: Path | None = None) -> Catalog:
    raw = yaml.safe_load(Path(path or CATALOG_PATH).read_text(encoding="utf-8"))
    tasks = [TaskDef(**entry) for entry in raw["tasks"]]
    ids = [t.id for t in tasks]
    keys = [t.key for t in tasks]
    if len(set(ids)) != len(ids):
        raise ValueError("tasks.yaml: id が重複しています")
    if len(set(keys)) != len(keys):
        raise ValueError("tasks.yaml: key が重複しています")
    return Catalog(tasks)
