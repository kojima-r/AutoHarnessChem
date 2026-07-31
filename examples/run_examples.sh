#!/usr/bin/env bash
# 各 conda 環境のライブラリを直接使うデモをまとめて実行する。
#
#   bash examples/run_examples.sh            # 全部（pyscf → aizynth → reactiont5）
#   bash examples/run_examples.sh pyscf      # 1つだけ
#   bash examples/run_examples.sh pyscf aizynth
#
# 同じことは harness の CLI からもできる:  ahc demo --env all
#
# 前提:
#   conda env pyscf       … opt_tddft (pip install -e tools/OptTDDFT)
#   conda env reactiont5  … torch + transformers（初回はモデルを ~250MB DL）
#   conda env aizynth     … aizynthfinder + 学習済みモデル（AIZYNTH_CONFIG）
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
OUT="${DEMO_OUT:-$HERE/output}"

declare -A DEMOS=(
    [pyscf]="pyscf_opt_tddft_demo.py"
    [reactiont5]="reactiont5_demo.py"
    [aizynth]="aizynth_demo.py"
)
ORDER=(pyscf aizynth reactiont5)

targets=("$@")
if [ ${#targets[@]} -eq 0 ]; then
    targets=("${ORDER[@]}")
fi

status=0
for env in "${targets[@]}"; do
    script="${DEMOS[$env]:-}"
    if [ -z "$script" ]; then
        echo "unknown demo: $env (choose from: ${ORDER[*]})" >&2
        status=2
        continue
    fi
    echo ""
    echo "=============================================================="
    echo " conda env: $env   script: examples/$script"
    echo "=============================================================="
    if ! conda run --no-capture-output -n "$env" python "$HERE/$script" --out "$OUT/$env"; then
        echo "[FAILED] $env demo (exit $?)" >&2
        status=1
    fi
done

echo ""
echo "出力: $OUT"
exit $status
