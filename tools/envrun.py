"""ツール専用環境（conda env / docker image）でのスクリプト実行を共通化する薄い層。

重い依存（pyscf + opt_tddft、aizynthfinder、torch など）は harness 本体の
プロセスに入れず、`SandboxConfig.named_envs` で解決される専用環境へ
「自己完結スクリプト + JSON 入出力」の形で投げる。tools/reactiont5.py で使った
方式をそのまま共通化したもので、tools/opttddft.py と tools/aizynth.py が使う。

呼び出し側の責務は 3 つだけ:
  1. EnvScript（スクリプト本体 + 入出力ファイル名）を用意する
  2. spec（スクリプトへ渡す JSON 化可能な dict）を作る
  3. 返ってきた payload を ToolResult に整形する
"""
from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from pathlib import Path

from schemas import ToolResult

# stderr のパターン → (error_type, retryable, ヒント文（{env} / 名前付きgroupを展開）)
FailureRule = tuple[str, str, bool, str]

_DOCKER_HINT = (
    "docker image が使えません（未ビルド / daemon 停止 / 権限不足）。"
    "`docker build -t <image> -f docker/Dockerfile.<tool> .` でビルドするか、"
    "config の sandbox.type を local にして conda 環境で実行してください"
    "（agent 側では修復不能）。"
)

_COMMON_FAILURES: tuple[FailureRule, ...] = (
    (r"Unable to find image|pull access denied|manifest for .* not found",
     "missing_environment", False, "指定された " + _DOCKER_HINT),
    (r"Cannot connect to the Docker daemon|permission denied while trying to connect"
     r"|Error response from daemon|docker: 'run' is not a docker command",
     "missing_environment", False, _DOCKER_HINT),
    (r"EnvironmentLocationNotFound|Could not find conda environment",
     "missing_environment", False,
     "conda 環境 `{env}` が見つかりません。"
     "config の sandbox.named_envs を確認してください（agent 側では修復不能）。"),
    (r"ModuleNotFoundError: No module named '(?P<module>[\w.]+)'",
     "missing_dependency", False,
     "専用環境 `{env}` に `{module}` がありません（agent 側では修復不能）。"),
    (r"MemoryError|Cannot allocate memory|std::bad_alloc|Killed",
     "out_of_memory", True,
     "メモリ不足で終了しました。分子数・基底関数・状態数を減らしてください。"),
)


@dataclass
class EnvScript:
    """専用環境で実行する自己完結スクリプトと、その入出力ファイル名。

    body 中の `__INPUT_JSON__` / `__OUTPUT_JSON__` / `__PARTIAL_JSON__` が
    実ファイル名へ置換される（.format() は本文の {} と衝突するため使わない）。

    partial_json を持つスクリプトは、1 件処理するたびに途中結果をそこへ書き出す。
    タイムアウトや強制終了でも「そこまでの結果」を回収できる。
    """
    body: str
    script_name: str
    input_json: str
    output_json: str
    partial_json: str | None = None

    def render(self) -> str:
        return (self.body
                .replace("__INPUT_JSON__", self.input_json)
                .replace("__OUTPUT_JSON__", self.output_json)
                .replace("__PARTIAL_JSON__", self.partial_json or "_partial.json"))


@dataclass
class EnvRun:
    """実行結果。error が None でなければそのまま return してよい ToolResult。

    partial=True は「途中で打ち切られたが payload に完了分が入っている」状態。
    呼び出し側は status="partial" の ToolResult を組み立てる。
    """
    payload: dict | None
    error: ToolResult | None
    stdout: str = ""
    stderr: str = ""
    new_files: tuple[str, ...] = ()
    partial: bool = False
    interrupted_reason: str = ""


def with_limits(sandbox, timeout_sec: int | None = None, threads: int = 1,
                memory_limit_mb: int | None = None):
    """実行上限だけを差し替えた同型 sandbox を返す（元の sandbox は変更しない）。

    量子化学計算は既定の timeout（600s）では終わらないことが多いため、
    重いツールは自分の timeout_sec を指定できる。LocalSandbox の RLIMIT_CPU は
    全スレッドの CPU 時間合計なので、スレッド数を掛けた値を上限にする。
    """
    if timeout_sec is None and memory_limit_mb is None:
        return sandbox
    update: dict = {}
    if timeout_sec is not None:
        update["timeout_sec"] = int(timeout_sec)
        # 全スレッドが回ると CPU 時間 = 実時間 × スレッド数 になるため、実時間の
        # timeout が先に効くように 1 スレッド分の余裕を持たせる（CPU 上限による
        # SIGKILL は「タイムアウト」より原因が分かりにくいので後回しにする）
        update["cpu_limit_sec"] = int(timeout_sec) * (max(1, int(threads)) + 1)
    if memory_limit_mb is not None:
        update["memory_limit_mb"] = int(memory_limit_mb)
    relaxed = copy.copy(sandbox)
    relaxed.config = sandbox.config.model_copy(update=update)
    return relaxed


def classify_failure(stderr: str, env: str,
                     extra: tuple[FailureRule, ...] = ()) -> tuple[str, bool, str]:
    """stderr から (error_type, retryable, ヒント) を決める。extra が優先。"""
    for pattern, error_type, retryable, hint in (*extra, *_COMMON_FAILURES):
        match = re.search(pattern, stderr)
        if match:
            fields = {k: v for k, v in (match.groupdict() or {}).items() if v}
            return error_type, retryable, hint.format(env=env, **fields)
    return "runtime_error", True, "専用環境でのスクリプト実行に失敗しました。"


def run_env_script(sandbox, workspace: Path, script: EnvScript, spec: dict, *,
                   timeout_sec: int | None = None, threads: int = 1,
                   memory_limit_mb: int | None = None,
                   extra_failures: tuple[FailureRule, ...] = (),
                   timeout_hint: str = "入力を分割して実行してください。") -> EnvRun:
    """spec を JSON で渡してスクリプトを専用環境で実行し、出力 JSON を読み取る。"""
    workspace = Path(workspace)
    sandbox = with_limits(sandbox, timeout_sec, threads, memory_limit_mb)
    (workspace / script.input_json).write_text(
        json.dumps(spec, ensure_ascii=False), encoding="utf-8")
    output_path = workspace / script.output_json
    partial_path = workspace / script.partial_json if script.partial_json else None
    for stale in (output_path, partial_path):
        if stale is not None and stale.exists():
            stale.unlink()  # 前回の結果を成功と誤認しないように消す

    result = sandbox.run(script.render(), script_name=script.script_name)
    common = {"stdout": result.stdout[-4000:], "stderr": result.stderr[-4000:]}
    new_files = tuple(result.new_files)

    def rescue_partial(reason: str) -> EnvRun | None:
        """打ち切られても、スクリプトが書いた途中結果があればそれを返す。"""
        if partial_path is None or not partial_path.exists():
            return None
        try:
            payload = json.loads(partial_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        return EnvRun(payload, None, result.stdout, result.stderr, new_files,
                      partial=True, interrupted_reason=reason)

    if result.timed_out:
        timeout_message = (f"{sandbox.config.timeout_sec}s でタイムアウトしました。"
                           f"{timeout_hint}")
        rescued = rescue_partial(timeout_message)
        if rescued is not None:
            return rescued
        return EnvRun(None, ToolResult(
            status="failed", summary=timeout_message,
            data=common, retryable=True, error_type="timeout",
        ), result.stdout, result.stderr, new_files)

    if result.returncode != 0 or not output_path.exists():
        env = sandbox.config.conda_env or sandbox.config.image
        if result.returncode in (-9, 137):
            # SIGKILL。多くは RLIMIT_AS / RLIMIT_CPU 超過（メッセージはロケール依存なので
            # 文字列ではなく returncode で判定する）
            error_type, retryable = "out_of_memory", True
            hint = (f"プロセスが強制終了されました（メモリ上限 "
                    f"{sandbox.config.memory_limit_mb}MB / CPU 上限 "
                    f"{sandbox.config.cpu_limit_sec}s の超過が主因）。"
                    "入力を小さくするか memory_limit_mb を上げてください。")
        elif result.returncode in (-11, 139):
            # SIGSEGV。BLAS/pyscf のような C 拡張は、アドレス空間上限
            # (RLIMIT_AS = memory_limit_mb) に達すると malloc 失敗を検査せず
            # segfault することがある。まずメモリ上限を疑う
            error_type, retryable = "out_of_memory", True
            hint = (f"計算ライブラリが SIGSEGV で落ちました。メモリ上限 "
                    f"{sandbox.config.memory_limit_mb}MB が不足している可能性が高い"
                    "です（C 拡張は確保失敗を検査せず落ちることがあります）。"
                    "memory_limit_mb を上げる、基底関数・状態数・分子数を下げる、"
                    "スレッド数を減らす、の順に試してください。")
        else:
            error_type, retryable, hint = classify_failure(result.stderr, env, extra_failures)
        message = f"{hint} (returncode={result.returncode})"
        # 途中まで計算できていれば、環境の問題でない限り部分結果として返す
        if error_type not in ("missing_environment", "missing_dependency"):
            rescued = rescue_partial(message)
            if rescued is not None:
                return rescued
        return EnvRun(None, ToolResult(
            status="failed", summary=message,
            data=common, retryable=retryable, error_type=error_type,
        ), result.stdout, result.stderr, new_files)

    try:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return EnvRun(None, ToolResult(
            status="failed", summary=f"出力 JSON を解釈できません: {e}",
            data=common, retryable=True, error_type="runtime_error",
        ), result.stdout, result.stderr, new_files)

    return EnvRun(payload, None, result.stdout, result.stderr, new_files)


def artifact(path: Path, kind: str | None = None):
    """workspace 内のファイルを Artifact として記述する。"""
    import mimetypes

    from schemas import Artifact

    path = Path(path)
    if kind is None:
        kind = "figure" if path.suffix.lower() in (".png", ".svg", ".pdf") else "data"
    return Artifact(
        path=str(path),
        mime=mimetypes.guess_type(path.name)[0] or "application/octet-stream",
        bytes=path.stat().st_size if path.exists() else 0,
        kind=kind,  # type: ignore[arg-type]
    )
