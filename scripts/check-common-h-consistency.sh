#!/bin/bash
# Verify all 2 canonical copies of common.h are byte-identical
# (scratch/common.h is a symlink to the ns3 build tree copy)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

COPIES=(
  "astra-sim-alibabacloud/astra-sim/network_frontend/ns3/common.h"
  "astra-sim-alibabacloud/extern/network_backend/ns3-interface/simulation/src/applications/astra-sim/network_frontend/ns3/common.h"
  "astra-sim-alibabacloud/extern/network_backend/ns3-interface/simulation/scratch/common.h"
)

BASE="${COPIES[0]}"
for ((i=1; i<${#COPIES[@]}; i++)); do
  if ! diff -q "$REPO_ROOT/$BASE" "$REPO_ROOT/${COPIES[$i]}" > /dev/null 2>&1; then
    echo "ERROR: ${COPIES[$i]} differs from $BASE"
    diff "$REPO_ROOT/$BASE" "$REPO_ROOT/${COPIES[$i]}"
    exit 1
  fi
done
echo "OK: All ${#COPIES[@]} common.h copies are identical."
