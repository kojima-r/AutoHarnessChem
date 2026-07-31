#!/usr/bin/env python
"""ReactionT5v2（transformers）を直接使うデモ。

conda 環境 `reactiont5` で実行する（harness を経由しない、素の transformers の使用例）。

  conda run -n reactiont5 python examples/reactiont5_demo.py                 # forward
  conda run -n reactiont5 python examples/reactiont5_demo.py --task all --beams 3

3 つの task:
  forward        … REACTANT/REAGENT から生成物 SMILES を予測（sagawa/ReactionT5v2-forward）
  retrosynthesis … 生成物から前駆体 SMILES を予測（sagawa/ReactionT5v2-retrosynthesis）
  yield          … 反応の収率 (0-100%) を回帰予測（sagawa/ReactionT5v2-yield）

初回実行時に HuggingFace から学習済みモデル（1 モデルあたり ~250MB）を
~/.cache/huggingface へダウンロードするため、ネットワークが必要。

出力（--out、既定 examples/output/reactiont5）:
  reactiont5_demo_<task>.csv  … 入力と予測（候補は | 区切り）
  reactiont5_demo_summary.json
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
import warnings
from pathlib import Path

DEFAULT_OUT = Path(__file__).resolve().parent / "output" / "reactiont5"

MODELS = {
    "forward": "sagawa/ReactionT5v2-forward",
    "retrosynthesis": "sagawa/ReactionT5v2-retrosynthesis",
    "yield": "sagawa/ReactionT5v2-yield",
}

# 形式: forward は REACTANT:...REAGENT:...、yield は PRODUCT: まで、retro は生成物のみ
DEFAULT_INPUTS = {
    "forward": [
        "REACTANT:COC(=O)C1=CCCN(C)C1.O.[Al+3].[H-].[Li+].[Na+].[OH-]REAGENT:C1CCOC1",
        "REACTANT:CC(=O)OC(C)=O.Nc1ccc(O)cc1REAGENT:",
    ],
    "retrosynthesis": [
        "CCN(CC)CCNC(=S)NC1CCCc2cc(C)cnc21",
        "CC(=O)Nc1ccc(O)cc1",
    ],
    "yield": [
        "REACTANT:CC(C)(C)OC(=O)N1CCC(C(=O)O)CC1.NCc1ccccc1"
        "REAGENT:CN(C)C=O.ClCCl PRODUCT:CC(C)(C)OC(=O)N1CCC(C(=O)NCc2ccccc2)CC1",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=[*MODELS, "all"], default="forward")
    parser.add_argument("--input", action="append", default=[], dest="inputs",
                        help="モデル入力文字列（複数可、既定は task ごとのサンプル）")
    parser.add_argument("--beams", type=int, default=3,
                        help="forward/retrosynthesis の候補数（ビーム幅）")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def load_yield_model(model_name: str, device):
    """収率モデルは回帰ヘッド付きの独自クラス（HuggingFace のモデルカードと同一構成）。"""
    import torch
    import torch.nn as nn
    from transformers import AutoTokenizer, PreTrainedModel, T5Config, T5ForConditionalGeneration

    class ReactionT5Yield(PreTrainedModel):
        config_class = T5Config
        base_model_prefix = "model"

        def __init__(self, config):
            super().__init__(config)
            self.config = config
            self.model = T5ForConditionalGeneration(config)
            self.model.resize_token_embeddings(self.config.vocab_size)
            hidden = self.config.hidden_size
            self.fc1 = nn.Linear(hidden, hidden // 2)
            self.fc2 = nn.Linear(hidden, hidden // 2)
            self.fc3 = nn.Linear(hidden // 2 * 2, hidden)
            self.fc4 = nn.Linear(hidden, hidden)
            self.fc5 = nn.Linear(hidden, 1)
            self.post_init()
            self.all_tied_weights_keys = {}       # transformers v5 対策

        def forward(self, inputs):
            encoder_outputs = self.model.encoder(**inputs)
            encoder_hidden = encoder_outputs[0]
            outputs = self.model.decoder(
                input_ids=torch.full((inputs["input_ids"].size(0), 1),
                                     self.config.decoder_start_token_id,
                                     dtype=torch.long, device=inputs["input_ids"].device),
                encoder_hidden_states=encoder_hidden,
            )
            hidden = self.config.hidden_size
            out1 = self.fc1(outputs[0].view(-1, hidden))
            out2 = self.fc2(encoder_hidden[:, 0, :].view(-1, hidden))
            out = self.fc3(torch.hstack((out1, out2)))
            return self.fc5(self.fc4(out)) * 100

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = ReactionT5Yield.from_pretrained(model_name).to(device).eval()
    return tokenizer, model


def predict(task: str, inputs: list[str], beams: int) -> list[dict]:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = MODELS[task]
    print(f"[{task}] loading {model_name} on {device} ...")
    rows: list[dict] = []

    if task == "yield":
        tokenizer, model = load_yield_model(model_name, device)
        with torch.no_grad():
            for text in inputs:
                encoded = {k: v.to(device)
                           for k, v in tokenizer([text], return_tensors="pt").items()}
                value = float(model(encoded).item())
                rows.append({"input": text, "predicted_yield": round(value, 2)})
                print(f"[{task}] {value:6.2f}%  <- {text[:70]}...")
        return rows

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device).eval()
    with torch.no_grad():
        for text in inputs:
            encoded = tokenizer(text, return_tensors="pt").to(device)
            output = model.generate(**encoded, num_beams=beams, num_return_sequences=beams,
                                    return_dict_in_generate=True, output_scores=True)
            candidates = [tokenizer.decode(seq, skip_special_tokens=True)
                          .replace(" ", "").rstrip(".") for seq in output["sequences"]]
            rows.append({"input": text, "prediction": candidates[0],
                         "candidates": "|".join(candidates)})
            print(f"[{task}] {candidates[0]}  <- {text[:70]}...")
    return rows


def main() -> int:
    args = parse_args()
    logging.getLogger("transformers").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = list(MODELS) if args.task == "all" else [args.task]

    summary: dict = {"tasks": {}, "beams": args.beams}
    for task in tasks:
        inputs = args.inputs or DEFAULT_INPUTS[task]
        started = time.time()
        try:
            rows = predict(task, inputs, args.beams)
        except Exception as e:
            print(f"[{task}] FAILED: {type(e).__name__}: {e}")
            summary["tasks"][task] = {"error": f"{type(e).__name__}: {e}"}
            continue
        elapsed = time.time() - started

        csv_path = out_dir / f"reactiont5_demo_{task}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        summary["tasks"][task] = {"model": MODELS[task], "n_inputs": len(inputs),
                                  "elapsed_sec": round(elapsed, 1),
                                  "output_csv": csv_path.name, "results": rows}
        print(f"[{task}] {len(rows)} 件を予測 ({elapsed:.1f}s) -> {csv_path.name}")

    (out_dir / "reactiont5_demo_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[done] -> {out_dir}")
    for name in sorted(p.name for p in out_dir.iterdir()):
        print(f"  - {name}")
    print("\n注意: 予測値は学習済みモデル (ReactionT5v2) による推定で、実験値ではありません。")
    return 0 if any("error" not in v for v in summary["tasks"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
