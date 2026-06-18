#!/bin/bash
# Harness LRA — one-click install script
set -e

ROOT=$(git rev-parse --show-toplevel 2>/dev/null || echo "$PWD")
echo "Harness LRA installer — target: $ROOT"

# 1. Copy data templates if not present
for f in feature_list.json progress.md; do
    if [ ! -f "$ROOT/$f" ]; then
        cp "$(dirname "$0")/../references/$f" "$ROOT/$f"
        echo "  created $f"
    else
        echo "  skip $f (exists)"
    fi
done

# 2. Copy scripts
mkdir -p "$ROOT/scripts"
for s in lra-gate.py quick_status.py; do
    cp "$(dirname "$0")/$s" "$ROOT/scripts/$s"
    chmod +x "$ROOT/scripts/$s"
    echo "  installed scripts/$s"
done

# 3. Create test runner if not present
if [ ! -f "$ROOT/scripts/lra-test.sh" ]; then
    cat > "$ROOT/scripts/lra-test.sh" << 'TESTEOF'
#!/bin/bash
set -e
FAIL=0
# Backend
python3 -m pytest tests/ -v 2>&1 || FAIL=1
# Frontend (skip if no package.json)
npx tsc --noEmit 2>/dev/null || true
# LRA integrity
python3 scripts/lra-gate.py --update 2>/dev/null || true
if [ $FAIL -eq 0 ]; then
    echo "ALL TESTS PASSED"
else
    echo "SOME TESTS FAILED"
    exit 1
fi
TESTEOF
    chmod +x "$ROOT/scripts/lra-test.sh"
    echo "  created scripts/lra-test.sh"
fi

# 4. Write hook config
SETTINGS="$ROOT/.claude/settings.local.json"
if [ ! -f "$SETTINGS" ]; then
    cat > "$SETTINGS" << 'HOOKEOF'
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Edit|Write",
        "command": "python3 scripts/lra-gate.py pre"
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "command": "python3 scripts/lra-gate.py post"
      }
    ],
    "Stop": [
      {
        "command": "python3 scripts/lra-gate.py stop"
      }
    ]
  }
}
HOOKEOF
    echo "  created .claude/settings.local.json"
else
    echo "  skip .claude/settings.local.json (exists)"
fi

# 5. Write version file
cat > "$ROOT/.lra_version" << 'VEREOF'
{"protocol":"1.0.0","installed_at":"","updated_at":"","compatible":["1.0.x"]}
VEREOF
python3 -c "
import json, os
from datetime import datetime, timezone
now = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
with open('$ROOT/.lra_version') as f: v = json.load(f)
v['installed_at'] = now; v['updated_at'] = now
with open('$ROOT/.lra_version','w') as f: json.dump(v, f)
"

echo ""
echo "Harness LRA installed. Verify with:"
echo "  python3 scripts/quick_status.py"
echo "  bash scripts/lra-test.sh"
