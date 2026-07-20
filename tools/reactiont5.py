"""ReactionT5v2 学習済みモデルによる反応予測ツール（test_reactiont5.py 由来）。

- yield          : 反応収率の回帰予測 (sagawa/ReactionT5v2-yield, 0–100%)
- forward        : 生成物予測 (sagawa/ReactionT5v2-forward)
- retrosynthesis : 逆合成（前駆体予測）(sagawa/ReactionT5v2-retrosynthesis)

torch / transformers は harness 環境（pyscf 環境）に入れない。
SandboxConfig.named_envs["reactiont5"] に従い、専用の conda 環境（既定: reactiont5）
または docker image で自己完結スクリプトとして実行することで環境を分離している。
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from schemas import Artifact, ToolResult

MODELS = {
    "yield": "sagawa/ReactionT5v2-yield",
    "forward": "sagawa/ReactionT5v2-forward",
    "retrosynthesis": "sagawa/ReactionT5v2-retrosynthesis",
}

INPUT_JSON = "_reactiont5_input.json"
OUTPUT_JSON = "_reactiont5_output.json"
SCRIPT_NAME = "_reactiont5_script.py"

# 専用環境で実行される自己完結スクリプト。
# .format() は本文中の {} と衝突するため __PLACEHOLDER__ 置換を使う。
_SCRIPT_TEMPLATE = '''
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

Path("__OUTPUT_JSON__").write_text(
    json.dumps({"task": task, "model": model_name, "results": results},
               ensure_ascii=False),
    encoding="utf-8",
)
print(f"predicted {len(results)} entries with {model_name} on {device}")
'''


def _classify_failure(stderr: str, conda_env: str) -> tuple[str, str]:
    if re.search(r"ModuleNotFoundError", stderr):
        return ("missing_dependency",
                f"専用環境 `{conda_env}` に torch/transformers がありません。"
                "環境へのインストールが必要です（agent側では修正できません）。")
    if re.search(r"EnvironmentLocationNotFound|Could not find conda environment", stderr,
                 re.IGNORECASE):
        return ("missing_environment",
                f"conda 環境 `{conda_env}` が見つかりません。"
                "config の sandbox.named_envs.reactiont5.conda_env を確認してください。")
    if re.search(r"(OSError|HTTPError|ConnectionError|offline|Can't load)", stderr):
        return ("model_unavailable",
                "モデルのダウンロード/ロードに失敗しました。ネットワークまたは "
                "HuggingFace キャッシュ（~/.cache/huggingface）を確認してください。")
    return ("runtime_error", "ReactionT5 スクリプトの実行に失敗しました。")


def predict_reaction_t5(
    workspace: Path,
    reactions: list[str],
    task: str = "forward",
    num_beams: int = 1,
    output_csv: str = "reactiont5_predictions.csv",
    *,
    sandbox,
) -> ToolResult:
    if task not in MODELS:
        return ToolResult(status="failed",
                          summary=f"unknown task `{task}` (expected {list(MODELS)})",
                          retryable=False, error_type="invalid_input")
    if not reactions:
        return ToolResult(status="failed", summary="reactions is empty",
                          retryable=False, error_type="invalid_input")

    workspace = Path(workspace)
    (workspace / INPUT_JSON).write_text(
        json.dumps({"task": task, "model_name": MODELS[task],
                    "reactions": reactions, "num_beams": num_beams},
                   ensure_ascii=False),
        encoding="utf-8",
    )
    script = (_SCRIPT_TEMPLATE
              .replace("__INPUT_JSON__", INPUT_JSON)
              .replace("__OUTPUT_JSON__", OUTPUT_JSON))
    result = sandbox.run(script, script_name=SCRIPT_NAME)

    if result.timed_out:
        return ToolResult(
            status="failed",
            summary=f"ReactionT5 の実行が {sandbox.config.timeout_sec}s でタイムアウトしました。"
                    "分子数を分割してください。",
            data={"stderr": result.stderr[-2000:]},
            retryable=True, error_type="timeout",
        )
    output_path = workspace / OUTPUT_JSON
    if result.returncode != 0 or not output_path.exists():
        error_type, hint = _classify_failure(result.stderr, sandbox.config.conda_env or "?")
        return ToolResult(
            status="failed",
            summary=f"{hint} (returncode={result.returncode})",
            data={"stdout": result.stdout[-2000:], "stderr": result.stderr[-3000:]},
            retryable=error_type == "runtime_error",
            error_type=error_type,
        )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    rows = payload["results"]
    fieldnames = ["input"] + (["predicted_yield"] if task == "yield"
                              else ["prediction", "candidates"])
    csv_path = workspace / output_csv
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            record = dict(row)
            if "candidates" in record:
                record["candidates"] = "|".join(record["candidates"])
            writer.writerow(record)

    summary = f"ReactionT5v2-{task}: {len(rows)} 件を予測 → {csv_path.name}"
    if task == "yield":
        values = [r["predicted_yield"] for r in rows]
        summary += f" (yield {min(values):.1f}–{max(values):.1f}%)"
    return ToolResult(
        status="success",
        summary=summary,
        data={"task": task, "model": payload["model"], "results": rows,
              "output_csv": str(csv_path)},
        artifacts=[Artifact(path=str(csv_path), mime="text/csv",
                            bytes=csv_path.stat().st_size, kind="data")],
    )
