"""ChemEval データセットの取得と読み込み。

ChemEval の評価スクリプト（`benchmarks_chemeval/ChemEval/`）にはデータが含まれず、
本体は HuggingFace（`Ooo1/ChemEval`）の parquet 2 本（text / multimodal）にある。
ここでは

  1. parquet をダウンロード（`data/raw/`）
  2. JSONL へ変換（`data/<split>.jsonl`。multimodal は画像も `data/images/` へ展開）
  3. カタログ（tasks.yaml）と突き合わせて `ChemEvalItem` の列を返す

を行う。parquet の読み出しには pyarrow が必要だが、**harness 本体の環境には重い依存を
入れない**方針なので、pyarrow を持つ python（conda 環境など）を探して subprocess で
変換する（tools/envrun.py と同じ「別環境へスクリプトを投げる」考え方）。
"""
from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

from benchmarks_chemeval.catalog import Catalog, TaskDef, load_catalog

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
IMAGE_DIR = DATA_DIR / "images"

HF_DATASET = "Ooo1/ChemEval"
SPLIT_FILES = {
    "text": "data/text-00000-of-00001.parquet",
    "multimodal": "data/multimodal-00000-of-00001.parquet",
}
SPLITS = tuple(SPLIT_FILES)


@dataclass
class ChemEvalItem:
    """ChemEval の 1 問（+ カタログ由来のメタ情報）。"""
    item_id: str
    task_id: str
    task_name: str
    level: str
    dimension: str
    metric: str
    ahc_task_type: str
    answer_type: str
    split: str
    shot: int
    filename: str
    query: str
    target: str
    image: str | None = None          # data/images 以下の相対パス（multimodal のみ）
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# --- 取得 ---------------------------------------------------------------

def parquet_path(split: str) -> Path:
    return RAW_DIR / f"{split}.parquet"


def jsonl_path(split: str) -> Path:
    return DATA_DIR / f"{split}.jsonl"


def download_parquet(split: str, *, force: bool = False) -> Path:
    """HuggingFace から parquet を取得する（`resolve/main` の直接ダウンロード）。"""
    if split not in SPLIT_FILES:
        raise ValueError(f"unknown split: {split} (expected one of {SPLITS})")
    target = parquet_path(split)
    if target.exists() and not force:
        return target
    url = f"https://huggingface.co/datasets/{HF_DATASET}/resolve/main/{SPLIT_FILES[split]}"
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".part")
    print(f"[chemeval] downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as response, tmp.open("wb") as out:
        shutil.copyfileobj(response, out)
    tmp.replace(target)
    print(f"[chemeval] saved {target} ({target.stat().st_size / 1e6:.1f} MB)")
    return target


# parquet → JSONL 変換スクリプト（pyarrow を持つ python で実行する自己完結スクリプト）
_CONVERT_SCRIPT = r'''
import json, sys
import pyarrow.parquet as pq

src, dst, image_dir = sys.argv[1], sys.argv[2], sys.argv[3]
table = pq.read_table(src)
columns = table.column_names
rows = table.to_pylist()
written = 0
with open(dst, "w", encoding="utf-8") as out:
    for index, row in enumerate(rows):
        record = {"index": index}
        for key in ("query", "target", "filename", "img_name", "img_path", "file_path"):
            if key in columns and row.get(key) is not None:
                record[key] = row[key]
        image = row.get("image") if "image" in columns else None
        if isinstance(image, dict) and image.get("bytes"):
            import os
            os.makedirs(image_dir, exist_ok=True)
            name = str(row.get("img_name") or f"{index}.png").strip()
            name = name.replace("/", "_").replace("\\", "_") or f"{index}.png"
            if "." not in name:
                name += ".png"
            name = f"{index:05d}_{name}"
            with open(os.path.join(image_dir, name), "wb") as fh:
                fh.write(image["bytes"])
            record["image"] = name
        out.write(json.dumps(record, ensure_ascii=False) + "\n")
        written += 1
print(json.dumps({"rows": written, "columns": columns}))
'''


def _interpreter_candidates() -> list[str]:
    """pyarrow を持つ python の候補（現在の環境 → conda 環境 → PATH）。"""
    candidates = [sys.executable]
    env_override = os.environ.get("CHEMEVAL_PYTHON")
    if env_override:
        candidates.insert(0, env_override)
    conda_root = Path(os.environ.get("CONDA_ROOT", Path.home() / "miniconda3"))
    if conda_root.exists():
        candidates.append(str(conda_root / "bin" / "python"))
        envs = conda_root / "envs"
        if envs.exists():
            candidates += [str(p / "bin" / "python") for p in sorted(envs.iterdir())
                           if (p / "bin" / "python").exists()]
    for name in ("python3", "python"):
        found = shutil.which(name)
        if found:
            candidates.append(found)
    seen: set[str] = set()
    return [c for c in candidates if c and not (c in seen or seen.add(c))]


def find_parquet_interpreter() -> str | None:
    """parquet を読める python を 1 つ選ぶ（見つからなければ None）。"""
    for candidate in _interpreter_candidates():
        try:
            done = subprocess.run([candidate, "-c", "import pyarrow"],
                                  capture_output=True, timeout=120)
        except (OSError, subprocess.SubprocessError):
            continue
        if done.returncode == 0:
            return candidate
    return None


def convert_parquet(split: str, *, force: bool = False) -> Path:
    """parquet → JSONL（+ 画像展開）。"""
    out_path = jsonl_path(split)
    if out_path.exists() and not force:
        return out_path
    source = download_parquet(split, force=force)
    interpreter = find_parquet_interpreter()
    if interpreter is None:
        raise RuntimeError(
            "parquet を読める python が見つかりません。`pip install pyarrow` するか、"
            "pyarrow を持つ環境の python を CHEMEVAL_PYTHON で指定してください "
            "（例: CHEMEVAL_PYTHON=~/miniconda3/envs/aizynth/bin/python）。"
            "JSONL を自分で用意する場合は "
            f"{out_path} に query/target/filename を含む JSON Lines を置いても構いません。")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    script = DATA_DIR / "_convert_parquet.py"
    script.write_text(_CONVERT_SCRIPT, encoding="utf-8")
    tmp = out_path.with_suffix(".part")
    done = subprocess.run([interpreter, str(script), str(source), str(tmp), str(IMAGE_DIR)],
                          capture_output=True, text=True, timeout=1800)
    if done.returncode != 0:
        raise RuntimeError(f"parquet の変換に失敗しました ({interpreter}):\n{done.stderr[-2000:]}")
    tmp.replace(out_path)
    print(f"[chemeval] {out_path} <- {source.name} {done.stdout.strip()}")
    return out_path


def prepare(splits: list[str] | None = None, *, force: bool = False) -> dict[str, Path]:
    """指定 split のデータを使える状態にする（既にあれば何もしない）。"""
    return {split: convert_parquet(split, force=force) for split in (splits or ["text"])}


# --- 読み込み -----------------------------------------------------------

def _task_field(row: dict, key: str) -> str:
    """欠損（None や文字列 "nan"）を空文字に潰して返す。

    multimodal split は `filename` が空で、タスク名が `file_path` に入っている。
    parquet 由来の欠損は文字列 "nan" として現れることがある。
    """
    value = str(row.get(key) or "").strip()
    return "" if value.lower() in ("", "nan", "none", "null") else value


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_items(
    split: str = "text",
    *,
    catalog: Catalog | None = None,
    task_ids: list[str] | None = None,
    levels: list[str] | None = None,
    dimensions: list[str] | None = None,
    metrics: list[str] | None = None,
    shot: int | None = 0,
    limit_per_task: int | None = None,
    seed: int = 0,
    path: Path | None = None,
    auto_prepare: bool = True,
) -> list[ChemEvalItem]:
    """JSONL を読み、カタログで絞り込んだ `ChemEvalItem` の列を返す。

    shot=0（既定）は 0-shot のみ、shot=3 は 3-shot のみ、None は両方。
    limit_per_task はタスクごとの件数上限（seed 固定のサンプリング）。
    """
    catalog = catalog or load_catalog()
    source = Path(path) if path else jsonl_path(split)
    if not source.exists():
        if not auto_prepare:
            raise FileNotFoundError(
                f"{source} がありません。`python -m benchmarks_chemeval.evaluate prepare` "
                "を実行してください。")
        source = convert_parquet(split)

    wanted = {t.id for t in catalog.select(task_ids, levels, dimensions, metrics,
                                          splits=[split])}
    grouped: dict[tuple[str, int], list[ChemEvalItem]] = {}
    unknown: dict[str, int] = {}
    for row in read_jsonl(source):
        filename = _task_field(row, "filename") or _task_field(row, "file_path")
        task, item_shot = catalog.lookup(filename)
        if task is None:
            unknown[filename] = unknown.get(filename, 0) + 1
            continue
        if shot is not None and item_shot != shot:
            continue
        if task.id not in wanted:
            continue
        index = int(row.get("index", len(grouped.get((task.id, item_shot), []))))
        item = ChemEvalItem(
            item_id=f"{task.id}-{item_shot}shot-{index:05d}",
            task_id=task.id,
            task_name=task.name,
            level=task.level,
            dimension=task.dimension,
            metric=task.metric,
            ahc_task_type=task.ahc_task_type,
            answer_type=task.answer_type,
            split=split,
            shot=item_shot,
            filename=filename,
            query=row.get("query") or "",
            target="" if row.get("target") is None else str(row["target"]),
            image=_task_field(row, "image") or None,
            extra={k: _task_field(row, k) for k in ("img_path", "file_path")
                   if _task_field(row, k)},
        )
        grouped.setdefault((task.id, item_shot), []).append(item)

    if unknown:
        print(f"[chemeval] カタログ未登録の filename を {len(unknown)} 種類 "
              f"({sum(unknown.values())} 件) 読み飛ばしました: "
              f"{list(unknown)[:3]}")

    items: list[ChemEvalItem] = []
    order = {t.id: i for i, t in enumerate(catalog)}
    for (task_id, item_shot), group in sorted(grouped.items(),
                                              key=lambda kv: (order[kv[0][0]], kv[0][1])):
        group.sort(key=lambda it: it.item_id)
        if limit_per_task is not None and len(group) > limit_per_task:
            group = random.Random(f"{seed}:{task_id}:{item_shot}").sample(group, limit_per_task)
            group.sort(key=lambda it: it.item_id)
        items += group
    return items


def image_path(item: ChemEvalItem) -> Path | None:
    return IMAGE_DIR / item.image if item.image else None


__all__ = [
    "ChemEvalItem", "SPLITS", "TaskDef", "download_parquet", "image_path",
    "jsonl_path", "load_items", "prepare",
]
