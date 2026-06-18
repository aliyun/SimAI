#!/usr/bin/env python3
"""Quick LRA status check — generic for any LRA-governed project."""
import json, os, sys

def find_root():
    """Walk up to find LRA project root (contains feature_list.json)."""
    d = os.path.abspath(os.getcwd())
    while d != '/':
        if os.path.exists(os.path.join(d, 'feature_list.json')):
            return d
        d = os.path.dirname(d)
    return None

def main():
    root = find_root()
    if not root:
        print("Not an LRA project (no feature_list.json found)")
        sys.exit(1)

    feature_path = os.path.join(root, 'feature_list.json')
    with open(feature_path) as f:
        data = json.load(f)

    s = data.get('summary', {})
    print(f"LRA Status — {data.get('updated', 'unknown')}")
    print(f"  project: {data.get('project', 'unknown')}")
    print(f"  done={s.get('done', 0)}  pending={s.get('pending', 0)}  in_progress={s.get('in_progress', 0)}")
    print()

    icons = {'done': '[OK]', 'in_progress': '[>>]', 'pending': '[--]'}
    for item in data.get('active', []):
        icon = icons.get(item.get('status', ''), '[??]')
        desc = item.get('description', '')[:100]
        conf = item.get('confidence', '')[:80]
        print(f"  {icon} {item['id']} [{item.get('type', '?')}] {item.get('status', '?')}")
        print(f"      {desc}")
        if conf:
            print(f"      【{conf}】")

    # Check dirty state
    dirty_path = os.path.join(root, '.lra_dirty')
    if os.path.exists(dirty_path):
        with open(dirty_path) as f:
            dirty = json.load(f)
        print(f"\n  [!] Dirty: {dirty.get('feature', '?')} — {len(dirty.get('files', []))} files")
        print(f"  Run: bash scripts/lra-test.sh")
    else:
        print(f"\n  Clean — no dirty files")

if __name__ == '__main__':
    main()
