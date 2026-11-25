#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python -B -u reproduce_inference.py --problems MC --suites main --output "${1:-runs/mc}"
