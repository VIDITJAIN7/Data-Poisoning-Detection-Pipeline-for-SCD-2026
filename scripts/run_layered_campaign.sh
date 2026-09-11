#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/venv/bin/python"
OUT="$ROOT/outputs/layered_campaign"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/campaign.log") 2>&1
echo "Layered campaign started $(date -Is)"
run_one(){ "$PYTHON" "$ROOT/scripts/run_trusted_campaign.py" --attack "$1" --seed "$2" --split-seed 42 --rate "$3" --epochs 30 --output "$OUT"; }
for s in 42 43 44; do run_one label "$s" .05; done
for s in 42 43 44; do run_one backdoor "$s" .05; done
run_one label 42 .01; run_one label 42 .10
run_one backdoor 42 .01; run_one backdoor 42 .10
"$PYTHON" "$ROOT/scripts/summarize_trusted_campaign.py" --output "$OUT"
echo "Layered campaign complete $(date -Is)"
