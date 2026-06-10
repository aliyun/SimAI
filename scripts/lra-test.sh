#!/bin/bash
# LRA test suite — run all existing tests, record results, clear dirty on pass.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
DIRTY_FILE="$ROOT/.lra_dirty"
LAST_TEST_FILE="$ROOT/.lra_last_test"
FAILED=0
STAGES=""

echo "=========================================="
echo "  LRA Test Suite"
echo "=========================================="

# --- Backend ---
if [ -d "$ROOT/server/tests" ]; then
    echo "--- Backend Tests ---"
    cd "$ROOT/server"
    if python3 -m pytest tests/ -v --tb=short 2>&1; then
        echo "  [PASS] Backend"; STAGES="$STAGES backend"
    else
        echo "  [FAIL] Backend"; FAILED=1
    fi
    cd "$ROOT"
fi

# --- Frontend ---
if [ -f "$ROOT/dashboard/tsconfig.json" ]; then
    echo "--- Frontend Type Check ---"
    cd "$ROOT/dashboard"
    if npx tsc --noEmit 2>&1; then
        echo "  [PASS] TypeScript"; STAGES="$STAGES tsc"
    else
        echo "  [FAIL] TypeScript"; FAILED=1
    fi
    cd "$ROOT"
fi

# --- E2E (opt-in) ---
if [ "${LRA_E2E:-0}" = "1" ]; then
    echo "--- E2E Tests ---"
    cd "$ROOT/dashboard"
    npx playwright test 2>&1 || echo "  [WARN] E2E (non-blocking)"
    STAGES="$STAGES e2e"
    cd "$ROOT"
fi

# GIT_HEAD and PASSED for later use (timestamp recorded after integrity snapshot)
GIT_HEAD=$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo "unknown")
PASSED=$([ "$FAILED" -eq 0 ] && echo "True" || echo "False")

# Check E2E requirement
NEED_E2E=0
if [ -f "$DIRTY_FILE" ]; then
    # Check if any frontend dirty files have non-deletion diffs
    python3 -c "
import json, subprocess
with open('$DIRTY_FILE') as f: dirty = json.load(f)
fe_files = [fp for fp in dirty.get('files',[]) if fp.startswith('dashboard/') and not fp.endswith('.md')]
if fe_files:
    deletions_only = True
    try:
        diff = subprocess.check_output(['git','diff','HEAD','--']+fe_files, cwd='$ROOT', stderr=subprocess.DEVNULL).decode()
        for line in diff.split('\n'):
            if line.startswith('+') and not line.startswith('+++'): deletions_only = False; break
    except: deletions_only = False
    if not deletions_only: print('NEED_E2E')
" 2>/dev/null | grep -q NEED_E2E && NEED_E2E=1
fi

echo "=========================================="
if [ "$FAILED" -eq 0 ]; then
    if [ "$NEED_E2E" = "1" ] && [ "${LRA_E2E:-0}" != "1" ]; then
        echo "  TESTS PASSED (E2E required for frontend changes)"
        echo "  Run: LRA_E2E=1 scripts/lra-test.sh"
        exit 1
    fi
    echo "  ALL TESTS PASSED (stages: $STAGES)"
    rm -f "$DIRTY_FILE"
    echo "  .lra_dirty cleared"

    # Snapshot integrity (done features + progress.md)
    python3 -c "
import sys, os, json
sys.path.insert(0, os.path.join('$ROOT', 'scripts'))
from lra_common import load_features, compute_done_hash, compute_progress_hash
features = load_features('$ROOT/feature_list.json')
done_h, done_n = compute_done_hash(features)
progress_h, progress_c = compute_progress_hash('$ROOT/progress.md')
json.dump({'done_hash': done_h, 'done_count': done_n,
           'progress_hash': progress_h, 'progress_count': progress_c},
          open('$ROOT/.lra_done_hash', 'w'))
print(f'  integrity: done={done_h}({done_n}) progress={progress_h}({progress_c})')
"

    # Record test timestamp AFTER integrity snapshot
    python3 -c "
import json, time
json.dump({'timestamp':time.time(),'git_head':'${GIT_HEAD}','passed':${PASSED},'stages':'${STAGES}'.split()}, open('${LAST_TEST_FILE}','w'))
"

    echo "=========================================="
    exit 0
else
    echo "  SOME TESTS FAILED"
    echo "  .lra_dirty preserved"
    echo "=========================================="
    exit 1
fi
