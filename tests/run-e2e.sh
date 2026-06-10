#!/bin/bash
set -e
cd "$(dirname "$0")/.."
echo "=== Running E2E tests ==="
python3 -m pytest tests/e2e/ -v "$@"
