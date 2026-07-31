#!/usr/bin/env python
"""AiZynthFinder による逆合成経路探索のデモ。

conda 環境 `aizynth` で実行する（harness を経由しない、素の aizynthfinder の使用例）。

  conda run -n aizynth python examples/aizynth_demo.py
  conda run -n aizynth python examples/aizynth_demo.py \
      --target "CC(=O)Nc1ccc(O)cc1" --iteration-limit 200 --time-limit 180

学習済みモデル（USPTO expansion policy + ZINC stock）が必要:

  conda run -n aizynth download_public_data /path/to/aizynth_data
  export AIZYNTH_CONFIG=/path/to/aizynth_data/config.yml

config.yml は --config / $AIZYNTH_CONFIG / $AIZYNTH_DATA/config.yml /
既定ディレクトリ（<repo>/data/aizynth/, ~/aizynth_data/）の順で探索する。
DL 先が別の場所なら `ln -s <dl先> data/aizynth` でも認識される。

出力（--out、既定 examples/output/aizynth）:
  aizynth_demo_routes.json  … 経路木を含む生データ
  aizynth_demo_routes.csv   … 1 経路 1 行の要約（段数・前駆体・score）
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

DEFAULT_OUT = Path(__file__).resolve().parent / "output" / "aizynth"
CONFIG_CANDIDATES = (
    Path(__file__).resolve().parent.parent / "data" / "aizynth" / "config.yml",
    Path.home() / "aizynth_data" / "config.yml",
)
DEFAULT_TARGETS = [
    "CC(=O)Nc1ccc(O)cc1",                      # パラセタモール
    "CCN(CC)CCNC(=S)NC1CCCc2cc(C)cnc21",       # ReactionT5 デモと同じ分子
]


def resolve_config(explicit: str | None) -> Path | None:
    if explicit:
        return Path(explicit) if Path(explicit).exists() else None
    for var in ("AIZYNTH_CONFIG",):
        value = os.environ.get(var, "").strip()
        if value and Path(value).exists():
            return Path(value)
    data_dir = os.environ.get("AIZYNTH_DATA", "").strip()
    if data_dir and (Path(data_dir) / "config.yml").exists():
        return Path(data_dir) / "config.yml"
    for candidate in CONFIG_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", action="append", default=[], dest="targets",
                        help="目標分子の SMILES（複数可）")
    parser.add_argument("--algorithm", choices=["mcts", "retrostar"], default="mcts")
    parser.add_argument("--iteration-limit", type=int, default=100)
    parser.add_argument("--time-limit", type=int, default=120, help="1 分子あたりの秒数")
    parser.add_argument("--max-transforms", type=int, default=6, help="経路の最大段数")
    parser.add_argument("--n-routes", type=int, default=5, help="保存する上位経路数")
    parser.add_argument("--config", help="AiZynthFinder の config.yml")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def leaves(node: dict) -> list[dict]:
    """経路木の葉（それ以上分解されない分子ノード）。"""
    if node.get("type") == "mol" and not node.get("children"):
        return [node]
    found: list[dict] = []
    for child in node.get("children") or []:
        found.extend(leaves(child))
    return found


def count_reactions(node: dict) -> int:
    total = 1 if node.get("type") == "reaction" else 0
    for child in node.get("children") or []:
        total += count_reactions(child)
    return total


def route_score(route: dict) -> float | None:
    scores = route.get("scores") or {}
    if "state score" in scores:
        return float(scores["state score"])
    numeric = [v for v in scores.values() if isinstance(v, (int, float))]
    return float(numeric[0]) if numeric else None


def main() -> int:
    args = parse_args()
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    config_path = resolve_config(args.config)
    if config_path is None:
        print("学習済みモデルが見つかりません。次を実行してください:\n"
              "  conda run -n aizynth download_public_data /path/to/aizynth_data\n"
              "  export AIZYNTH_CONFIG=/path/to/aizynth_data/config.yml",
              file=sys.stderr)
        return 2

    import yaml
    from aizynthfinder.aizynthfinder import AiZynthFinder

    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    config["search"] = {**(config.get("search") or {}),
                        "algorithm": args.algorithm,
                        "iteration_limit": args.iteration_limit,
                        "time_limit": args.time_limit,
                        "max_transforms": args.max_transforms}
    config["post_processing"] = {**(config.get("post_processing") or {}),
                                 "min_routes": min(5, args.n_routes),
                                 "max_routes": max(5, args.n_routes),
                                 "all_routes": False}

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[config] {config_path}")
    finder = AiZynthFinder(configdict=config)
    finder.stock.select(finder.stock.items)                    # config の全 stock
    finder.expansion_policy.select(finder.expansion_policy.items[0])
    print(f"[setup] stock={finder.stock.selection} "
          f"expansion={finder.expansion_policy.selection} "
          f"algorithm={args.algorithm} iterations={args.iteration_limit}")

    targets = args.targets or DEFAULT_TARGETS
    entries = []
    for index, target in enumerate(targets):
        finder.target_smiles = target
        started = time.time()
        finder.tree_search()
        finder.build_routes()
        elapsed = time.time() - started

        routes = []
        for rank, route in enumerate(finder.routes.dict_with_scores()[: args.n_routes], 1):
            route_leaves = leaves(route)
            routes.append({
                "rank": rank,
                "score": route_score(route),
                "n_steps": count_reactions(route),
                "n_precursors": len(route_leaves),
                "solved": bool(route_leaves) and all(leaf.get("in_stock")
                                                     for leaf in route_leaves),
                "precursors": [leaf.get("smiles", "") for leaf in route_leaves],
                "tree": route,
            })
        solved = any(route["solved"] for route in routes)
        entries.append({"target": target, "solved": solved, "n_routes": len(routes),
                        "search_time_s": round(elapsed, 1), "routes": routes,
                        "statistics": finder.extract_statistics()})
        best = routes[0] if routes else None
        print(f"[target {index}] solved={solved} routes={len(routes)} "
              f"time={elapsed:.1f}s"
              + (f" best: {best['n_steps']} 段 / 前駆体 {best['n_precursors']} 個"
                 f" (score={best['score']})" if best else ""))
        if best:
            print(f"            出発物質: {'.'.join(best['precursors'])}")

    json_path = out_dir / "aizynth_demo_routes.json"
    json_path.write_text(json.dumps({"targets": entries}, indent=2, ensure_ascii=False,
                                    default=str), encoding="utf-8")

    csv_path = out_dir / "aizynth_demo_routes.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["target_smiles", "route_rank", "solved",
                                                "n_steps", "n_precursors", "score",
                                                "precursors", "search_time_s"])
        writer.writeheader()
        for entry in entries:
            for route in entry["routes"] or [None]:
                writer.writerow({
                    "target_smiles": entry["target"],
                    "route_rank": route["rank"] if route else 0,
                    "solved": route["solved"] if route else False,
                    "n_steps": route["n_steps"] if route else 0,
                    "n_precursors": route["n_precursors"] if route else 0,
                    "score": route["score"] if route else None,
                    "precursors": ".".join(route["precursors"]) if route else "",
                    "search_time_s": entry["search_time_s"],
                })

    n_solved = sum(1 for entry in entries if entry["solved"])
    print(f"\n[done] {n_solved}/{len(entries)} 分子で stock まで到達 -> {out_dir}")
    for name in sorted(p.name for p in out_dir.iterdir()):
        print(f"  - {name}")
    print("\n注意: score は経路探索の内部指標であり、収率の予測値ではありません。")
    return 0 if n_solved else 1


if __name__ == "__main__":
    sys.exit(main())
