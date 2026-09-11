#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/venv/bin/python"
OUT="$ROOT/outputs/validated/campaign"
mkdir -p "$OUT"

run_set() {
  local seed="$1"
  local rate="$2"
  local tag="seed-${seed}-rate-${rate//./}"
  "$PYTHON" "$ROOT/run_all.py" 0 1 2 3 4 \
    --seed "$seed" --poison-rate "$rate" \
    --run-dir "$OUT/$tag"
}

for seed in 42 43 44; do
  run_set "$seed" 0.05
done

run_set 42 0.01
run_set 42 0.10

"$PYTHON" - <<PY
import config
config.LOG_DIR = "$OUT/seed-42-rate-005/logs"
config.PLOT_DIR = "$OUT/seed-42-rate-005/plots"
from scripts.generate_graphs import main
main()
PY
echo "Validated campaign complete: $OUT"
