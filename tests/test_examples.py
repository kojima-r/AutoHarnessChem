"""examples/ のデモが「実行可能な成果物」として壊れていないことを確認する。

各デモは専用 conda 環境（pyscf / reactiont5 / aizynth）で動くため、ここでは
実行はせず、構文・CLI 定義・harness 側の登録との整合だけを検査する。
"""
import ast
import os
import stat
from pathlib import Path

import pytest

from app.cli import DEMOS

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


@pytest.mark.parametrize("script", sorted(DEMOS.values()))
def test_demo_scripts_are_valid_and_executable(script):
    path = EXAMPLES / script
    assert path.exists(), f"{script} が examples/ にありません"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    assert ast.get_docstring(tree), "デモは使い方を説明する docstring を持つこと"
    # シェバン付き + 実行権限（`./examples/<script>` で直接起動できる）
    assert path.read_text(encoding="utf-8").startswith("#!")
    assert os.stat(path).st_mode & stat.S_IXUSR


@pytest.mark.parametrize("script", sorted(DEMOS.values()))
def test_demos_accept_out_option(script):
    """`ahc demo` / run_examples.sh は --out で出力先を渡す。"""
    source = (EXAMPLES / script).read_text(encoding="utf-8")
    assert '"--out"' in source


@pytest.mark.parametrize("script", sorted(DEMOS.values()))
def test_demos_do_not_import_the_harness(script):
    """デモは各環境のライブラリだけで動く（harness の依存を持ち込まない）。"""
    source = (EXAMPLES / script).read_text(encoding="utf-8")
    for module in ("from schemas", "import schemas", "from harness", "from tools",
                   "import pydantic"):
        assert module not in source, f"{script} が harness 側の {module} に依存している"


def test_runner_script_is_executable_and_covers_all_demos():
    runner = EXAMPLES / "run_examples.sh"
    assert os.stat(runner).st_mode & stat.S_IXUSR
    source = runner.read_text(encoding="utf-8")
    for env, script in DEMOS.items():
        assert f"[{env}]=" in source and script in source


def test_readme_documents_every_demo():
    readme = (EXAMPLES / "README.md").read_text(encoding="utf-8")
    for env, script in DEMOS.items():
        assert script in readme and env in readme


def test_cli_registers_demo_command():
    from app.cli import main

    with pytest.raises(SystemExit):
        main(["demo", "--env", "unknown-env"])     # argparse が choices で弾く
