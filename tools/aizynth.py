"""AiZynthFinder による逆合成経路解析ツール。

ReactionT5 の `predict_reaction_t5(task="retrosynthesis")` が「1 段階の前駆体予測」
であるのに対し、こちらは expansion policy + stock を使った**多段の経路探索**
（MCTS / Retro* ）で、購入可能な出発物質まで遡った合成経路を返す。

aizynthfinder（+ ONNX / TensorFlow 系依存）は harness 本体や pyscf 環境には入れず、
`SandboxConfig.named_envs["aizynth"]`（既定: conda env `aizynth`）で自己完結
スクリプトとして実行する。

学習済みモデル（USPTO expansion policy + stock）は別途ダウンロードが必要:

    conda run -n aizynth download_public_data /path/to/aizynth_data
    export AIZYNTH_CONFIG=/path/to/aizynth_data/config.yml
"""
from __future__ import annotations

import os
from pathlib import Path

from schemas import ToolResult
from tools.envrun import EnvScript, artifact, run_env_script

# config.yml（expansion policy / stock の場所を書いたファイル）の既定探索順
_CONFIG_ENV_VARS = ("AIZYNTH_CONFIG",)
_CONFIG_DIR_ENV_VARS = ("AIZYNTH_DATA",)
_CONFIG_FALLBACK_DIRS = (
    Path(__file__).resolve().parent.parent / "data" / "aizynth",  # DL 先への symlink 可
    Path.home() / "aizynth_data",
    Path.home() / ".aizynthfinder",
)

_DOWNLOAD_HINT = (
    "学習済みモデルが見つかりません。次のコマンドで公開データ（USPTO expansion "
    "policy + ZINC stock）を取得し、config.yml の場所を AIZYNTH_CONFIG に設定して"
    "ください（agent 側では修復不能）:\n"
    "  conda run -n aizynth download_public_data /path/to/aizynth_data\n"
    "  export AIZYNTH_CONFIG=/path/to/aizynth_data/config.yml"
)

_AIZYNTH_FAILURES = (
    (r"aizynth config not found", "model_unavailable", False, _DOWNLOAD_HINT),
    (r"No such file or directory: '(?P<path>[^']*\.(onnx|hdf5|csv\.gz|yml))'",
     "model_unavailable", False,
     "モデル/ストックファイル `{path}` が存在しません。" + _DOWNLOAD_HINT),
    (r"ValueError: The key '(?P<key>[^']+)' is not in the collection",
     "invalid_input", False,
     "指定した `{key}` が config.yml に定義されていません。"
     "stock / expansion の名前を確認してください。"),
)

_SCRIPT_BODY = r'''
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import yaml

spec = json.loads(Path("__INPUT_JSON__").read_text(encoding="utf-8"))


def resolve_config_path(spec):
    """config.yml の場所。harness 側で解決できていなければ実行環境側で探す
    （docker sandbox では image に焼き込んだ AIZYNTH_CONFIG が使われる）。"""
    candidates = [spec.get("config_yaml"), os.environ.get("AIZYNTH_CONFIG")]
    data_dir = os.environ.get("AIZYNTH_DATA")
    if data_dir:
        candidates.append(str(Path(data_dir) / "config.yml"))
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    raise SystemExit("aizynth config not found: set AIZYNTH_CONFIG or pass config_yaml")


config_path = resolve_config_path(spec)
print("[aizynth] config=%s" % config_path)
config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
config["search"] = {**(config.get("search") or {}), **spec["search"]}
config["post_processing"] = {**(config.get("post_processing") or {}),
                            **spec["post_processing"]}

from aizynthfinder.aizynthfinder import AiZynthFinder

finder = AiZynthFinder(configdict=config)

# stock / expansion policy / filter policy の選択（未指定なら config の全件・先頭）
if spec["stock"]:
    finder.stock.select(spec["stock"])
elif finder.stock.items:
    finder.stock.select(finder.stock.items)
if spec["expansion"]:
    finder.expansion_policy.select(spec["expansion"])
elif finder.expansion_policy.items:
    finder.expansion_policy.select(finder.expansion_policy.items[0])
if spec["filter_policy"]:
    finder.filter_policy.select(spec["filter_policy"])

selection = {
    "stock": list(finder.stock.selection or []),
    "expansion_policy": list(finder.expansion_policy.selection or []),
    "filter_policy": list(finder.filter_policy.selection or []),
}
print("[aizynth] selection=%s" % selection)


def leaves(node):
    """経路木の葉（それ以上分解されない分子ノード）を列挙する。"""
    if node.get("type") == "mol" and not node.get("children"):
        return [node]
    found = []
    for child in node.get("children") or []:
        found.extend(leaves(child))
    return found


def count_reactions(node):
    total = 1 if node.get("type") == "reaction" else 0
    for child in node.get("children") or []:
        total += count_reactions(child)
    return total


def route_score(route):
    scores = route.get("scores") or {}
    if "state score" in scores:
        return float(scores["state score"])
    numeric = [v for v in scores.values() if isinstance(v, (int, float))]
    return float(numeric[0]) if numeric else None


import csv


def write_outputs(entries):
    """1 分子終わるたびに JSON/CSV と途中結果を書く（打ち切られても残る）。"""
    payload = {
        "targets": entries,
        "selection": selection,
        "search": config["search"],
        "n_solved": sum(1 for t in entries if t["solved"]),
        "n_requested": len(spec["targets"]),
    }
    Path(spec["output_json"]).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    with open(spec["output_csv"], "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=[
            "target_smiles", "route_rank", "solved", "n_steps", "n_precursors",
            "score", "precursors", "search_time_s"])
        writer.writeheader()
        for entry in entries:
            if not entry["routes"]:
                writer.writerow({"target_smiles": entry["target"], "route_rank": 0,
                                 "solved": False, "n_steps": 0, "n_precursors": 0,
                                 "score": None, "precursors": "",
                                 "search_time_s": round(entry["search_time_s"], 2)})
                continue
            for route in entry["routes"]:
                writer.writerow({
                    "target_smiles": entry["target"],
                    "route_rank": route["rank"],
                    "solved": route["solved"],
                    "n_steps": route["n_steps"],
                    "n_precursors": route["n_precursors"],
                    "score": route["score"],
                    "precursors": ".".join(route["precursors"]),
                    "search_time_s": round(entry["search_time_s"], 2),
                })
    return payload


targets_out = []
for index, target in enumerate(spec["targets"]):
    finder.target_smiles = target
    started = time.time()
    finder.tree_search()
    finder.build_routes()
    elapsed = time.time() - started

    try:
        statistics = finder.extract_statistics()
    except Exception as e:                       # 解析が壊れても経路自体は返す
        statistics = {"error": "%s: %s" % (type(e).__name__, e)}

    try:
        route_dicts = list(finder.routes.dict_with_scores())
    except Exception:
        route_dicts = list(finder.routes.dicts or [])

    routes = []
    for rank, route in enumerate(route_dicts[: spec["n_routes"]], start=1):
        route_leaves = leaves(route)
        routes.append({
            "rank": rank,
            "score": route_score(route),
            "n_steps": count_reactions(route),
            "n_precursors": len(route_leaves),
            "solved": bool(route_leaves) and all(leaf.get("in_stock")
                                                for leaf in route_leaves),
            "precursors": [leaf.get("smiles", "") for leaf in route_leaves],
            "precursors_in_stock": [bool(leaf.get("in_stock")) for leaf in route_leaves],
            "tree": route,
        })

    solved = any(route["solved"] for route in routes)
    targets_out.append({
        "target": target,
        "solved": solved,
        "search_time_s": elapsed,
        "n_routes": len(routes),
        "routes": routes,
        "statistics": statistics,
    })
    print("[aizynth] target %d: solved=%s routes=%d time=%.1fs"
          % (index, solved, len(routes), elapsed))

    # 経路 JSON / CSV を都度更新し、途中結果も回収できるようにする
    payload = write_outputs(targets_out)
    partial = dict(payload)
    partial["partial"] = True
    tmp = Path("__PARTIAL_JSON__.tmp")
    tmp.write_text(json.dumps(partial, ensure_ascii=False, default=str),
                   encoding="utf-8")
    tmp.replace(Path("__PARTIAL_JSON__"))

payload = write_outputs(targets_out)
Path("__OUTPUT_JSON__").write_text(json.dumps(payload, ensure_ascii=False, default=str),
                                  encoding="utf-8")
print("[aizynth] solved %d/%d targets" % (payload["n_solved"], len(targets_out)))
'''

SCRIPT = EnvScript(body=_SCRIPT_BODY, script_name="_aizynth_script.py",
                   input_json="_aizynth_input.json",
                   output_json="_aizynth_output.json",
                   partial_json="_aizynth_partial.json")


def resolve_config(workspace: Path, config_yaml: str | None = None) -> Path | None:
    """AiZynthFinder の config.yml を解決する（見つからなければ None）。

    優先順: 明示指定 → $AIZYNTH_CONFIG → $AIZYNTH_DATA/config.yml → 既定ディレクトリ。
    """
    if config_yaml:
        path = Path(config_yaml)
        if not path.is_absolute():
            for base in (Path(workspace), Path.cwd()):
                if (base / path).exists():
                    return (base / path).resolve()
        return path.resolve() if path.exists() else None

    for var in _CONFIG_ENV_VARS:
        value = os.environ.get(var, "").strip()
        if value and Path(value).exists():
            return Path(value).resolve()
    for var in _CONFIG_DIR_ENV_VARS:
        value = os.environ.get(var, "").strip()
        if value and (Path(value) / "config.yml").exists():
            return (Path(value) / "config.yml").resolve()
    for directory in _CONFIG_FALLBACK_DIRS:
        if (directory / "config.yml").exists():
            return (directory / "config.yml").resolve()
    return None


def plan_retrosynthesis(
    workspace: Path,
    targets: list[str],
    algorithm: str = "mcts",
    iteration_limit: int = 100,
    time_limit_sec: int = 120,
    max_transforms: int = 6,
    n_routes: int = 5,
    stock: list[str] | None = None,
    expansion: list[str] | None = None,
    filter_policy: list[str] | None = None,
    config_yaml: str | None = None,
    output_json: str = "retrosynthesis_routes.json",
    output_csv: str = "retrosynthesis_routes.csv",
    timeout_sec: int | None = None,
    memory_limit_mb: int = 16384,
    *,
    sandbox,
) -> ToolResult:
    """目標分子から購入可能な出発物質までの逆合成経路を探索する。"""
    if isinstance(targets, str):
        targets = [targets]
    if not targets:
        return ToolResult(status="failed", summary="targets is empty",
                          retryable=False, error_type="invalid_input")
    if algorithm not in ("mcts", "retrostar"):
        return ToolResult(status="failed",
                          summary=f"unknown algorithm `{algorithm}` "
                                  "(expected 'mcts' or 'retrostar')",
                          retryable=False, error_type="invalid_input")

    workspace = Path(workspace)
    # docker sandbox では image 内の AIZYNTH_CONFIG を使う（ホスト側のパスは見えない）
    on_docker = sandbox.config.type == "docker"
    config_path = resolve_config(workspace, config_yaml)
    if config_path is None and not on_docker:
        return ToolResult(status="blocked", summary=_DOWNLOAD_HINT,
                          retryable=False, error_type="model_unavailable")

    spec = {
        "config_yaml": str(config_path) if config_path else None,
        "targets": list(targets),
        "n_routes": int(n_routes),
        "stock": list(stock) if stock else None,
        "expansion": list(expansion) if expansion else None,
        "filter_policy": list(filter_policy) if filter_policy else None,
        "output_json": output_json,
        "output_csv": output_csv,
        "search": {
            "algorithm": algorithm,
            "iteration_limit": int(iteration_limit),
            "time_limit": int(time_limit_sec),
            "max_transforms": int(max_transforms),
        },
        "post_processing": {"min_routes": min(5, int(n_routes)),
                            "max_routes": max(5, int(n_routes)),
                            "all_routes": False},
    }
    # 各 target が time_limit まで探索しうるので、全体の上限はその合計 + 余裕
    default_timeout = len(targets) * (int(time_limit_sec) + 120) + 180
    # stock DB（zinc_stock.hdf5 は ~650MB）と onnxruntime を載せるため、既定の
    # メモリ上限（4GB）では足りない
    run = run_env_script(
        sandbox, workspace, SCRIPT, spec,
        timeout_sec=timeout_sec or default_timeout,
        memory_limit_mb=memory_limit_mb,
        extra_failures=_AIZYNTH_FAILURES,
        timeout_hint="target を分割するか iteration_limit / time_limit_sec を下げてください。",
    )
    if run.error is not None:
        return run.error

    payload = run.payload
    entries = payload.get("targets", [])
    n_solved = payload.get("n_solved", 0)
    json_path, csv_path = workspace / output_json, workspace / output_csv
    # tree を含む生データは JSON 側にあるので、ToolResult には要約だけ載せる
    digest = [
        {"target": entry["target"], "solved": entry["solved"],
         "n_routes": entry["n_routes"],
         "search_time_s": round(entry["search_time_s"], 1),
         "top_route": ({k: v for k, v in entry["routes"][0].items() if k != "tree"}
                       if entry["routes"] else None)}
        for entry in entries
    ]
    if not entries:
        return ToolResult(status="failed", summary="経路探索の結果が空です",
                          data={"stdout": run.stdout[-2000:]},
                          retryable=True, error_type="runtime_error")

    searched = {entry["target"] for entry in entries}
    pending = [t for t in targets if t not in searched]
    n_with_routes = sum(1 for entry in entries if entry["n_routes"])
    if n_solved == len(targets):
        status = "success"
    elif n_with_routes:
        # 経路は得られたが stock まで到達していない / 未探索の target がある
        status = "partial"
    else:
        status = "failed"

    summary = (f"AiZynthFinder ({algorithm}): {n_solved}/{len(targets)} 分子で "
               f"stock まで到達する経路を発見 → {output_json} / {output_csv}")
    if run.partial:
        listed = ", ".join(pending[:3]) + ("…" if len(pending) > 3 else "")
        summary += (f" ※途中で打ち切られました（{run.interrupted_reason}）"
                    f"。未探索 {len(pending)} 件（{listed}）は分けて呼び直してください")
    return ToolResult(
        status=status,
        summary=summary,
        data={"n_solved": n_solved, "n_targets": len(entries),
              "n_requested": len(targets), "targets": digest, "pending": pending,
              "selection": payload.get("selection", {}),
              "search": payload.get("search", {}), "interrupted": run.partial,
              "output_json": str(json_path), "output_csv": str(csv_path),
              "config_yaml": str(config_path) if config_path else "(実行環境の AIZYNTH_CONFIG)"},
        artifacts=[artifact(p) for p in (json_path, csv_path) if p.exists()],
        retryable=status != "success",
        error_type=("timeout" if run.partial
                    else (None if n_solved == len(targets) else "no_route_found")),
    )
