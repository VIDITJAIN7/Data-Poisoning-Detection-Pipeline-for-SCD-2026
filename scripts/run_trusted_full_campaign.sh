#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/venv/bin/python"
OUT="$ROOT/outputs/trusted_campaign"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/campaign.log") 2>&1
echo "Trusted campaign started $(date -Is)"

run_one() {
  local attack="$1" seed="$2" rate="$3"
  echo "=== attack=$attack seed=$seed rate=$rate ==="
  "$PYTHON" "$ROOT/scripts/run_trusted_campaign.py" \
    --attack "$attack" --seed "$seed" --split-seed 42 --rate "$rate" \
    --epochs 30 --output "$OUT"
}

for seed in 42 43 44; do run_one label "$seed" .05; done
for seed in 42 43 44; do run_one backdoor "$seed" .05; done
run_one label 42 .01
run_one label 42 .10
run_one backdoor 42 .01
run_one backdoor 42 .10

"$PYTHON" "$ROOT/scripts/summarize_trusted_campaign.py" --output "$OUT"
echo "Trusted campaign complete $(date -Is)"
