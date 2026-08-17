"""ReactionT5v2 学習済みモデルによる反応予測ツール（test_reactiont5.py 由来）。

- yield          : 反応収率の回帰予測 (sagawa/ReactionT5v2-yield, 0–100%)
- forward        : 生成物予測 (sagawa/ReactionT5v2-forward)
- retrosynthesis : 逆合成（前駆体予測）(sagawa/ReactionT5v2-retrosynthesis)

torch / transformers は harness 環境（pyscf 環境）に入れない。
SandboxConfig.named_envs["reactiont5"] に従い、専用の conda 環境（既定: reactiont5）
または docker image で自己完結スクリプトとして実行することで環境を分離している。
実行は他の重いツール（opttddft / aizynth）と同じ tools/envrun.py 経由なので、
実行上限（timeout_sec / memory_limit_mb）の差し替えと途中結果の回収ができる。
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

from schemas import ToolResult
from tools.envrun import EnvScript, artifact, run_env_script

MODELS = {
    "yield": "sagawa/ReactionT5v2-yield",
    "forward": "sagawa/ReactionT5v2-forward",
    "retrosynthesis": "sagawa/ReactionT5v2-retrosynthesis",
}

INPUT_JSON = "_reactiont5_input.json"
OUTPUT_JSON = "_reactiont5_output.json"
PARTIAL_JSON = "_reactiont5_partial.json"
SCRIPT_NAME = "_reactiont5_script.py"

# torch を import するだけで、OpenBLAS がコア数分のバッファを確保しようとするため、
# sandbox 既定のメモリ上限ではアドレス空間 (RLIMIT_AS) が足りずに
# `OpenBLAS error: Memory allocation still failed after 10 retries` で落ちる。
# 88 コア機での実測: 4096 / 8192 / 12288MB は失敗、16384MB で成功（RSS は 1.6GB
# 程度で、必要なのは実メモリではなくアドレス空間）。
DEFAULT_MEMORY_LIMIT_MB = 16384

# torch は既定で全コアを使うので、CPU 時間の上限（LocalSandbox の RLIMIT_CPU は
# 全スレッドの合計）はコア数で見積もる。これを小さくすると、実時間の timeout より
# 先に SIGKILL が飛んで原因が分かりにくくなる（1 反応で user time 約 16s）。
_CPU_THREADS = os.cpu_count() or 8

_T5_FAILURES = (
    # OpenBLAS / tokenizers(rayon) / mmap はいずれもアドレス空間不足で落ちるが、
    # envrun の共通ルール（Cannot allocate memory 等）には引っかからない文言を使う
    (r"OpenBLAS error: Memory allocation|ThreadPoolBuildError|unable to mmap"
     r"|Cannot allocate memory|Errno 12",
     "out_of_memory", True,
     "メモリ上限が足りません（torch / transformers は import だけでコア数分の"
     "アドレス空間を要求します）。memory_limit_mb を上げてください"
     f"（既定 {DEFAULT_MEMORY_LIMIT_MB}MB、不足する場合は 32768 以上）。"),
    (r"HTTPError|ConnectionError|offline|Can't load|We couldn't connect|OSError",
     "model_unavailable", False,
     "モデルのダウンロード/ロードに失敗しました。ネットワークまたは "
     "HuggingFace キャッシュ（~/.cache/huggingface）を確認してください。"),
)

# 専用環境で実行される自己完結スクリプト。
# .format() は本文中の {} と衝突するため __PLACEHOLDER__ 置換を使う（envrun が行う）。
_SCRIPT_BODY = '''
import json
import logging
import warnings
from pathlib import Path

logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from transformers import (AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer,
                          PreTrainedModel, T5Config, T5ForConditionalGeneration)

spec = json.loads(Path("__INPUT_JSON__").read_text(encoding="utf-8"))
task = spec["task"]
reactions = spec["reactions"]
model_name = spec["model_name"]
num_beams = int(spec.get("num_beams", 1))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
results = []


def dump_partial():
    """1 反応ごとに途中結果を書く（打ち切られても完了分を回収できるようにする）。"""
    tmp = Path("__PARTIAL_JSON__.tmp")
    tmp.write_text(json.dumps({"task": task, "model": model_name,
                               "results": results, "partial": True},
                              ensure_ascii=False), encoding="utf-8")
    tmp.replace(Path("__PARTIAL_JSON__"))


if task == "yield":
    # https://huggingface.co/sagawa/ReactionT5v2-yield （test_reactiont5.py と同一構成）
    class ReactionT5Yield2(PreTrainedModel):
        config_class = T5Config
        base_model_prefix = "model"

        def __init__(self, config):
            super().__init__(config)
            self.config = config
            # from_pretrained をここで呼ばない（meta 衝突を回避）
            self.model = T5ForConditionalGeneration(config)
            self.model.resize_token_embeddings(self.config.vocab_size)
            hidden = self.config.hidden_size
            self.fc1 = nn.Linear(hidden, hidden // 2)
            self.fc2 = nn.Linear(hidden, hidden // 2)
            self.fc3 = nn.Linear(hidden // 2 * 2, hidden)
            self.fc4 = nn.Linear(hidden, hidden)
            self.fc5 = nn.Linear(hidden, 1)
            self.post_init()
            # Transformers v5 の mark_tied_weights_as_initialized 対策
            self.all_tied_weights_keys = {}

        def forward(self, inputs):
            encoder_outputs = self.model.encoder(**inputs)
            encoder_hidden = encoder_outputs[0]
            outputs = self.model.decoder(
                input_ids=torch.full(
                    (inputs["input_ids"].size(0), 1),
                    self.config.decoder_start_token_id,
                    dtype=torch.long,
                    device=inputs["input_ids"].device,
                ),
                encoder_hidden_states=encoder_hidden,
            )
            hidden = self.config.hidden_size
            out1 = self.fc1(outputs[0].view(-1, hidden))
            out2 = self.fc2(encoder_hidden[:, 0, :].view(-1, hidden))
            out = self.fc3(torch.hstack((out1, out2)))
            out = self.fc5(self.fc4(out))
            return out * 100

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = ReactionT5Yield2.from_pretrained(model_name).to(device).eval()
    with torch.no_grad():
        for reaction in reactions:
            inputs = {k: v.to(device)
                      for k, v in tokenizer([reaction], return_tensors="pt").items()}
            value = float(model(inputs).item())
            results.append({"input": reaction, "predicted_yield": value})
            dump_partial()
else:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device).eval()
    with torch.no_grad():
        for reaction in reactions:
            inputs = tokenizer(reaction, return_tensors="pt").to(device)
            output = model.generate(
                **inputs, num_beams=num_beams, num_return_sequences=num_beams,
                return_dict_in_generate=True, output_scores=True,
            )
            candidates = [
                tokenizer.decode(seq, skip_special_tokens=True).replace(" ", "").rstrip(".")
                for seq in output["sequences"]
            ]
            results.append({"input": reaction, "prediction": candidates[0],
                            "candidates": candidates})
            dump_partial()

Path("__OUTPUT_JSON__").write_text(
    json.dumps({"task": task, "model": model_name, "results": results},
               ensure_ascii=False),
    encoding="utf-8",
)
print(f"predicted {len(results)} entries with {model_name} on {device}")
'''

SCRIPT = EnvScript(body=_SCRIPT_BODY, script_name=SCRIPT_NAME,
                   input_json=INPUT_JSON, output_json=OUTPUT_JSON,
                   partial_json=PARTIAL_JSON)


def _write_csv(path: Path, task: str, rows: list[dict]) -> None:
    fieldnames = ["input"] + (["predicted_yield"] if task == "yield"
                              else ["prediction", "candidates"])
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            record = {k: v for k, v in row.items() if k in fieldnames}
            if "candidates" in record:
                record["candidates"] = "|".join(record["candidates"])
            writer.writerow(record)


def predict_reaction_t5(
    workspace: Path,
    reactions: list[str],
    task: str = "forward",
    num_beams: int = 1,
    output_csv: str = "reactiont5_predictions.csv",
    timeout_sec: int | None = None,
    memory_limit_mb: int = DEFAULT_MEMORY_LIMIT_MB,
    *,
    sandbox,
) -> ToolResult:
    if task not in MODELS:
        return ToolResult(status="failed",
                          summary=f"unknown task `{task}` (expected {list(MODELS)})",
                          retryable=False, error_type="invalid_input")
    if isinstance(reactions, str):
        reactions = [reactions]
    if not reactions:
        return ToolResult(status="failed", summary="reactions is empty",
                          retryable=False, error_type="invalid_input")

    workspace = Path(workspace)
    spec = {"task": task, "model_name": MODELS[task],
            "reactions": list(reactions), "num_beams": int(num_beams)}
    run = run_env_script(
        sandbox, workspace, SCRIPT, spec,
        timeout_sec=timeout_sec or sandbox.config.timeout_sec,
        threads=_CPU_THREADS, memory_limit_mb=memory_limit_mb,
        extra_failures=_T5_FAILURES,
        timeout_hint="反応を分割して呼び直してください。",
    )
    if run.error is not None:
        return run.error

    rows = run.payload.get("results", [])
    done = {row.get("input") for row in rows}
    pending = [r for r in reactions if r not in done]
    if not rows:
        return ToolResult(
            status="failed",
            summary=("予測結果が空です"
                     + (f"（{run.interrupted_reason}）" if run.partial else "")),
            data={"pending": pending, "stdout": run.stdout[-2000:]},
            retryable=True,
            error_type="timeout" if run.partial else "runtime_error",
        )

    csv_path = workspace / output_csv
    _write_csv(csv_path, task, rows)
    summary = (f"ReactionT5v2-{task}: {len(rows)}/{len(reactions)} 件を予測 "
               f"→ {csv_path.name}")
    if task == "yield":
        values = [r["predicted_yield"] for r in rows]
        summary += f" (yield {min(values):.1f}–{max(values):.1f}%)"
    if run.partial:
        listed = ", ".join(pending[:2]) + ("…" if len(pending) > 2 else "")
        summary += (f" ※途中で打ち切られました（{run.interrupted_reason}）"
                    f"。未処理 {len(pending)} 件（{listed}）は分けて呼び直してください")
    return ToolResult(
        status="partial" if run.partial else "success",
        summary=summary,
        data={"task": task, "model": run.payload.get("model", MODELS[task]),
              "results": rows, "pending": pending, "interrupted": run.partial,
              "output_csv": str(csv_path)},
        artifacts=[artifact(csv_path, kind="data")] if csv_path.exists() else [],
        retryable=bool(run.partial),
        error_type="timeout" if run.partial else None,
    )
