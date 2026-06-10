#!/bin/bash
# LRA Context Save — called when context window is low or compaction detected.
cd "$(dirname "$0")/.."
export REASON="${1:-compaction detected}"
export WARNING_FILE=".lra_context_warning"
echo "=== LRA Context Save ==="
python3 << 'PYEOF'
import sys, os, json
sys.path.insert(0, os.path.join(os.getcwd(), 'scripts'))
from lra_common import load_features

reason = os.environ.get("REASON", "compaction detected")
wf = os.environ.get("WARNING_FILE", ".lra_context_warning")

in_progress = []
features = load_features()
for f in features:
    if f.get("status") == "in_progress":
        in_progress.append({
            "id": f["id"],
            "description": f.get("description", "")[:120],
            "type": f.get("type", "?"),
            "has_created_at": bool(f.get("created_at")),
        })

dirty_files = []
if os.path.exists(".lra_dirty"):
    with open(".lra_dirty") as f:
        dirty_files = json.load(f).get("files", [])

w = {"timestamp": __import__('datetime').datetime.now(
     __import__('datetime').timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
     "reason": reason, "in_progress": in_progress, "dirty_files": dirty_files}
with open(wf, "w") as f:
    json.dump(w, f, indent=2)
print(f"State saved ({len(in_progress)} in_progress, {len(dirty_files)} dirty)")
PYEOF
