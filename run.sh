#!/bin/bash
# Unified pipeline runner

set -e

export PYTHONPATH="${PWD}:${PWD}/src"

CONFIG="${1:-configs/totalcapture_vicon_test.yaml}"
STAGE="${2:-all}"

echo "Config: $CONFIG"
echo "Stage: $STAGE"

python -m src.cli.run_pipeline --config "$CONFIG" --stages "$STAGE"

echo "=== Done ==="
