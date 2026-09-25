"""ChemBench の問題の読み込み。

**出題の正本は公式リポジトリ同梱の公開 report**（`chembench/reports/<model>/reports/*/*.json`）
にしている。HuggingFace（`jablonkagroup/ChemBench`）ではなく report を使う理由:

1. **文献値と 1 対 1 で突き合わせられる。** report の `name`（= question_name）が
   `chembench/reports/<model>/<model>.json` の per-question `all_correct` と同じ鍵なので、
   ahc が解いた問題だけに文献値を絞れる（推測による対応付けが一切いらない）。
   HuggingFace 側の `name` は重複が多く、問題文での突き合わせは 85% しか一致しない。
2. **文献の各モデルが実際に見たプロンプトそのもの**が入っている（選択肢の順番まで）。
   ahc にも同じ文字列を渡せるので、出題条件の差を消せる。
3. ネットワークもデータセットの依存も要らない（クローンだけで完結する）。

report に入っている `targets_`（正解）は生の LaTeX 表記（`\\ce{NaCl}` など）で、
プロンプト側の選択肢は chembench の後処理で剥がされた表記になっている。
`post_process()` は公式 `utils.py` の 5 つの正規表現をそのまま移したもので、これを
掛けると **12,720 件中すべて**（全モデルの report を横断して検証済み）で選択肢テキストが
一致する。そのため「選択肢の文字 → 得点」の対応を推測なしに復元できる。

HuggingFace 版は `verify_huggingface()` で照合にだけ使う（出題には使わない）。
"""
from __future__ import annotations

import json
import os
import random
import re
import shutil
import subprocess
import sys
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

from benchmarks_chembench.catalog import (CHEMBENCH_DIR, Catalog, QuestionMeta,
                                          load_catalog)

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
REPORTS_DIR = CHEMBENCH_DIR / "reports"

# 出題（プロンプト・正解）の取り出し元。成績には一切使わない ―― どのモデルの
# report を使っても問題文と正解は同じで、違うのは選択肢の並び順だけ。
DEFAULT_REFERENCE_MODEL = "gpt-4o"

HF_DATASET = "jablonkagroup/ChemBench"
HF_CONFIGS = ("analytical_chemistry", "chemical_preference", "general_chemistry",
              "inorganic_chemistry", "materials_science", "organic_chemistry",
              "physical_chemistry", "technical_chemistry", "toxicity_and_safety")

# --- chembench の prompt 後処理（公式 utils.py の LATEX_ENV_REGEX と同一） -------
_LATEX_PATTERNS = (
    re.compile(r"\\ce\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"),
    re.compile(r"\$([^$]+)\$"),
    re.compile(r"\[START_SMILES\](.*?)\[END_SMILES\]", re.DOTALL),
    re.compile(r"\[START_RXNSMILES\](.*?)\[END_RXNSMILES\]", re.DOTALL),
    re.compile(r"\\pu\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"),
)


def post_process(text: str) -> str:
    """`\\ce{}` / `$..$` / SMILES タグ / `\\pu{}` を中身だけにする（公式と同じ順）。"""
    out = str(text)
    for pattern in _LATEX_PATTERNS:
        out = pattern.sub(lambda m: m.group(1), out)
    return out


@dataclass
class ChemBenchItem:
    """ChemBench の 1 問（+ カタログ由来のメタ情報）。"""
    question_name: str            # 文献値と突き合わせる鍵
    topic: str
    topic_name: str
    requires: str
    difficulty: str
    metric_kind: str              # mcq | numeric
    ahc_task_type: str
    prompt: str                   # 文献の各モデルが見たプロンプト（原文のまま）
    question: str                 # プロンプトから切り出した問題文
    options: dict[str, str] = field(default_factory=dict)   # 文字 → 選択肢テキスト
    score_map: dict[str, float] = field(default_factory=dict)  # 文字 → 得点（1 が正解）
    target: float | None = None   # numeric の正解
    tolerance: float | None = None  # numeric の許容差（chembench と同じ 0.01 * target）
    keywords: list[str] = field(default_factory=list)
    description: str = ""
    in_human_subset: bool = False
    source_path: str = ""

    @property
    def correct_letters(self) -> list[str]:
        return sorted(letter for letter, score in self.score_map.items() if score == 1)

    def to_dict(self) -> dict:
        return asdict(self)


# --- report の読み出し ---------------------------------------------------

def available_reference_models() -> list[str]:
    """問題ごとの report を持つモデル（= 出題元に使えるもの）。"""
    if not REPORTS_DIR.exists():
        return []
    found = []
    for path in sorted(REPORTS_DIR.iterdir()):
        if path.is_dir() and any(path.glob("reports/*/*.json")):
            found.append(path.name)
    return found


def _question_report_files(model: str) -> list[Path]:
    return sorted((REPORTS_DIR / model).glob("reports/*/*.json"))


def parse_question(prompt: str) -> str:
    """プロンプトから問題文だけを切り出す。"""
    match = re.search(r"Question:\s*(.*?)(?:\n\nOptions:|\n\nYou MUST|\n\nConstraints:|\Z)",
                      prompt, re.DOTALL)
    return match.group(1).strip() if match else prompt.strip()


def parse_options(prompt: str) -> dict[str, str]:
    """プロンプトの Options ブロック → 文字 → 選択肢テキスト。

    選択肢は改行を含むことがあるので、行頭の `A. ` の出現位置で区切る。
    """
    match = re.search(r"\nOptions:\n(.*?)(?:\n\nYou MUST|\n\nConstraints:|\Z)",
                      prompt, re.DOTALL)
    if not match:
        return {}
    body = match.group(1)
    starts = [(m.start(), m.group(1)) for m in re.finditer(r"(?m)^([A-Z])\.\s", body)]
    options: dict[str, str] = {}
    for index, (position, letter) in enumerate(starts):
        end = starts[index + 1][0] if index + 1 < len(starts) else len(body)
        text = re.sub(r"^[A-Z]\.\s*", "", body[position:end]).strip()
        options[letter] = text
    return options


def resolve_score_map(options: dict[str, str],
                      targets: dict[str, float]) -> dict[str, float] | None:
    """選択肢の文字 → 得点。復元できなければ None（その問題は落とす）。

    `targets` の鍵は生の LaTeX 表記、`options` の値は後処理済み表記なので、
    `post_process` を掛けた上で**完全一致**で対応させる。1 対 1 にならない場合は
    採点を誤るので None を返す（黙って 0 点にしない）。
    """
    if not options or not targets:
        return None
    processed: dict[str, list[float]] = {}
    for text, score in targets.items():
        processed.setdefault(post_process(text).strip(), []).append(float(score))
    score_map: dict[str, float] = {}
    for letter, text in options.items():
        scores = processed.get(text.strip())
        if not scores:
            return None
        # 同一テキストの選択肢が複数ある場合は前から 1 つずつ割り当てる
        score_map[letter] = scores.pop(0)
        if not scores:
            processed.pop(text.strip(), None)
    if any(processed.values()):
        return None                      # 使われなかった正解が残った = 対応が崩れている
    if not [s for s in score_map.values() if s == 1]:
        return None                      # 正解が 1 つも無い（hamming が計算できない）
    return score_map


def _build_item(report: dict, meta: QuestionMeta | None, catalog: Catalog,
                human_subset: frozenset[str]) -> ChemBenchItem | None:
    name = report.get("name")
    prompt = report.get("prompt") or ""
    targets = report.get("targets_")
    if not name or not prompt or targets is None:
        return None
    topic = meta.topic if meta else ""
    topic_def = catalog.by_id(topic)
    common = dict(
        question_name=name,
        topic=topic or "unknown",
        topic_name=topic_def.name if topic_def else "Unknown",
        requires=meta.requires if meta else "",
        difficulty=meta.difficulty if meta else "",
        ahc_task_type=catalog.task_type(topic),
        prompt=prompt,
        question=parse_question(prompt),
        keywords=list(report.get("keywords") or []),
        description=report.get("description") or "",
        in_human_subset=name in human_subset,
        source_path=meta.source_path if meta else "",
    )
    if isinstance(targets, dict):
        options = parse_options(prompt)
        score_map = resolve_score_map(options, targets)
        if score_map is None:
            return None
        return ChemBenchItem(metric_kind="mcq", options=options,
                             score_map=score_map, **common)
    try:
        target = float(targets)
    except (TypeError, ValueError):
        return None
    # chembench の許容差は relative_tolerance が無ければ 0.01 * target（絶対量として
    # mae と比較される）。target <= 0 では許容差が 0 以下になり原理的に不正解になるが、
    # 文献値も同じ規則で採点されているので**そのまま踏襲する**（README に注記）。
    return ChemBenchItem(metric_kind="numeric", target=target,
                         tolerance=0.01 * target, **common)


def load_items(
    *,
    catalog: Catalog | None = None,
    reference_model: str = DEFAULT_REFERENCE_MODEL,
    topics: list[str] | None = None,
    metric_kinds: list[str] | None = None,
    requires: list[str] | None = None,
    difficulties: list[str] | None = None,
    human_subset_only: bool = False,
    limit_per_topic: int | None = None,
    seed: int = 0,
) -> list[ChemBenchItem]:
    """条件で絞った `ChemBenchItem` の列を返す（トピック順・question_name 順）。

    `limit_per_topic` はトピックごとの件数上限（seed 固定のサンプリング）。
    """
    catalog = catalog or load_catalog()
    files = _question_report_files(reference_model)
    if not files:
        raise FileNotFoundError(
            f"{REPORTS_DIR / reference_model} に問題ごとの report がありません。"
            f"出題元に使えるモデル: {', '.join(available_reference_models()) or 'なし'}"
            f"（`git submodule` ではなく通常のクローンとして "
            f"{CHEMBENCH_DIR} が展開されているか確認してください）")

    wanted_topics = {t.id for t in catalog.select(topics)}
    grouped: dict[str, list[ChemBenchItem]] = {}
    dropped = {"no_meta": 0, "unresolved": 0}
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        report = payload[0] if isinstance(payload, list) and payload else payload
        if not isinstance(report, dict):
            continue
        meta = catalog.meta(report.get("name", ""))
        if meta is None:
            dropped["no_meta"] += 1
            continue
        item = _build_item(report, meta, catalog, catalog.human_subset)
        if item is None:
            dropped["unresolved"] += 1
            continue
        if item.topic not in wanted_topics:
            continue
        if metric_kinds and item.metric_kind not in set(metric_kinds):
            continue
        if requires and item.requires not in set(requires):
            continue
        if difficulties and item.difficulty not in set(difficulties):
            continue
        if human_subset_only and not item.in_human_subset:
            continue
        grouped.setdefault(item.topic, []).append(item)

    if dropped["no_meta"] or dropped["unresolved"]:
        print(f"[chembench] 読み飛ばし: トピック不明 {dropped['no_meta']} 件 / "
              f"選択肢と正解の対応が復元できない {dropped['unresolved']} 件")

    order = {topic.id: i for i, topic in enumerate(catalog)}
    selected: list[tuple[str, list[ChemBenchItem]]] = []
    for topic, group in sorted(grouped.items(), key=lambda kv: order.get(kv[0], 999)):
        group.sort(key=lambda it: it.question_name)
        if limit_per_topic is not None and len(group) > limit_per_topic:
            group = random.Random(f"{seed}:{topic}").sample(group, limit_per_topic)
            group.sort(key=lambda it: it.question_name)
        selected.append((topic, group))

    if limit_per_topic is None:
        # 全件（= 文献値と同じ設定）は 2,788 問あり、利用枠をまたぐので必ず途中経過で
        # レポートを出すことになる。トピックごとに固めて並べると「先頭のトピックだけ
        # 終わった偏った部分集合」になってしまうため、**各トピックの構成比を保った順**に
        # 並べ替える（どこで切っても母集団の縮小版になる層化順序）。
        # 並べ替えても再開は question_name で判定するので結果には影響しない。
        ordered: list[tuple[float, int, ChemBenchItem]] = []
        for topic, group in selected:
            size = len(group)
            for index, item in enumerate(group):
                ordered.append(((index + 0.5) / size, order.get(topic, 999), item))
        ordered.sort(key=lambda row: (row[0], row[1]))
        return [item for _, _, item in ordered]

    return [item for _, group in selected for item in group]


# --- HuggingFace 版との照合（出題には使わない） --------------------------

_CONVERT_SCRIPT = r'''
import json, sys
import pyarrow.parquet as pq

out = {}
for path in sys.argv[2:]:
    table = pq.read_table(path)
    rows = []
    for row in table.to_pylist():
        examples = row.get("examples") or []
        first = examples[0] if examples else {}
        rows.append({"name": row.get("name"), "uuid": row.get("uuid"),
                     "subfield": row.get("subfield"),
                     "input": first.get("input"), "target": first.get("target"),
                     "target_scores": first.get("target_scores"),
                     "metrics": row.get("metrics")})
    out[path.split("/")[-1].replace(".parquet", "")] = rows
with open(sys.argv[1], "w", encoding="utf-8") as fh:
    json.dump(out, fh, ensure_ascii=False)
print(json.dumps({"configs": len(out), "rows": sum(len(v) for v in out.values())}))
'''


def _interpreter_candidates() -> list[str]:
    candidates = [sys.executable]
    override = os.environ.get("CHEMBENCH_PYTHON")
    if override:
        candidates.insert(0, override)
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
    """parquet を読める python を 1 つ選ぶ（harness 本体に pyarrow を入れないため）。"""
    for candidate in _interpreter_candidates():
        try:
            done = subprocess.run([candidate, "-c", "import pyarrow"],
                                  capture_output=True, timeout=120)
        except (OSError, subprocess.SubprocessError):
            continue
        if done.returncode == 0:
            return candidate
    return None


def download_huggingface(*, force: bool = False) -> list[Path]:
    """HuggingFace の parquet（9 config、合計約 0.5MB）を取得する。"""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for config in HF_CONFIGS:
        target = RAW_DIR / f"{config}.parquet"
        if target.exists() and not force:
            paths.append(target)
            continue
        url = (f"https://huggingface.co/datasets/{HF_DATASET}/resolve/main/"
               f"{config}/train-00000-of-00001.parquet")
        print(f"[chembench] downloading {url}")
        tmp = target.with_suffix(".part")
        with urllib.request.urlopen(url, timeout=120) as response, tmp.open("wb") as out:
            shutil.copyfileobj(response, out)
        tmp.replace(target)
        paths.append(target)
    return paths


def load_huggingface(*, force: bool = False) -> dict[str, list[dict]]:
    """parquet → config 名 → 行の dict（pyarrow を持つ python へ委譲して変換）。"""
    cache = DATA_DIR / "huggingface.json"
    if cache.exists() and not force:
        return json.loads(cache.read_text(encoding="utf-8"))
    paths = download_huggingface(force=force)
    interpreter = find_parquet_interpreter()
    if interpreter is None:
        raise RuntimeError(
            "parquet を読める python が見つかりません。`pip install pyarrow` するか、"
            "pyarrow を持つ環境を CHEMBENCH_PYTHON で指定してください。")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    script = DATA_DIR / "_convert_parquet.py"
    script.write_text(_CONVERT_SCRIPT, encoding="utf-8")
    done = subprocess.run([interpreter, str(script), str(cache), *map(str, paths)],
                          capture_output=True, text=True, timeout=1800)
    if done.returncode != 0:
        raise RuntimeError(f"parquet の変換に失敗しました ({interpreter}):\n"
                           f"{done.stderr[-2000:]}")
    print(f"[chembench] HuggingFace: {done.stdout.strip()}")
    return json.loads(cache.read_text(encoding="utf-8"))


def _fingerprint_text(text: str) -> str:
    """照合用の指紋。後処理した上で `\\` と `{}` を落とす。

    古い時期の report は `\\ce{C4H12As2}` を `\\{C4H12As2}` と描画していて（当時の
    後処理の取りこぼし）、HuggingFace 側の生表記とは記号だけが食い違う。中身も
    正解も同じなので、照合の指紋では記号を無視する。**採点には使わない**
    （採点側は report 内で完結しているので、この食い違いの影響を受けない）。
    """
    return re.sub(r"[\\{}]", "", post_process(text)).strip()


def verify_huggingface(items: list[ChemBenchItem], *, force: bool = False) -> dict:
    """出題（report 由来）が HuggingFace 版と一致するかを照合する。

    正解の集合（MCQ は target_scores の鍵と値、numeric は target）を指紋にして
    突き合わせる。**出題には使わない**、あくまで整合性の確認用。
    """
    data = load_huggingface(force=force)
    fingerprints: dict[str, list[str]] = {}
    for config, rows in data.items():
        for row in rows:
            scores = row.get("target_scores")
            if scores:
                parsed = json.loads(scores) if isinstance(scores, str) else scores
                key = json.dumps({_fingerprint_text(k): float(v)
                                  for k, v in parsed.items()}, sort_keys=True)
            elif row.get("target") is not None:
                key = f"num:{float(row['target'])}"
            else:
                continue
            fingerprints.setdefault(key, []).append(config)

    matched, missing, topic_ok = 0, [], 0
    for item in items:
        if item.metric_kind == "mcq":
            key = json.dumps({_fingerprint_text(text): float(item.score_map[letter])
                              for letter, text in item.options.items()},
                             sort_keys=True)
        else:
            key = f"num:{float(item.target)}"
        configs = fingerprints.get(key)
        if configs:
            matched += 1
            if item.topic in configs:
                topic_ok += 1
        else:
            missing.append(item.question_name)
    return {"n_items": len(items), "matched": matched,
            "topic_agrees": topic_ok, "unmatched": missing[:20],
            "n_unmatched": len(missing),
            "hf_rows": sum(len(rows) for rows in data.values())}


__all__ = ["ChemBenchItem", "DEFAULT_REFERENCE_MODEL", "available_reference_models",
           "load_items", "parse_options", "parse_question", "post_process",
           "resolve_score_map", "verify_huggingface"]
