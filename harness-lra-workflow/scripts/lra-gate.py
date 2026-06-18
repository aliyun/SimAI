#!/usr/bin/env python3
"""Harness LRA gate — hook-driven integrity checks."""
import json, os, sys, hashlib
from datetime import datetime, timezone

ROOT = None
def find_root():
    global ROOT
    if ROOT: return ROOT
    d = os.path.abspath(os.getcwd())
    while d != '/':
        if os.path.exists(os.path.join(d, 'feature_list.json')):
            ROOT = d
            return d
        d = os.path.dirname(d)
    sys.exit(0)

def load_features():
    with open(os.path.join(find_root(), 'feature_list.json')) as f:
        return json.load(f)

def active_feature(features):
    for item in features.get('active', []):
        if item.get('status') == 'in_progress':
            return item
    return None

def compute_hashes():
    root = find_root()
    features = load_features()
    done_items = [{k:v for k,v in i.items() if k != 'updated_at'}
                  for i in features.get('active', []) if i.get('status') == 'done']
    done_hash = hashlib.sha256(json.dumps(done_items, sort_keys=True).encode()).hexdigest()[:16]
    with open(os.path.join(root, 'progress.md')) as f:
        progress_md = f.read()
    progress_hash = hashlib.sha256(progress_md.encode()).hexdigest()[:16]
    return done_hash, progress_hash, len(done_items), len(progress_md.splitlines())

def cmd_pre():
    """PreToolUse gate."""
    features = load_features()
    feat = active_feature(features)
    if not feat:
        return  # no in_progress feature — allow (init phase)

    # Check dirty state
    dirty_path = os.path.join(find_root(), '.lra_dirty')
    if os.path.exists(dirty_path):
        with open(dirty_path) as f:
            dirty = json.load(f)
        if dirty.get('feature') != feat['id']:
            print(f"BLOCKED: .lra_dirty for '{dirty.get('feature')}' — run lra-test.sh first", file=sys.stderr)
            sys.exit(1)

    # Check verification_steps for type=feature
    if feat['type'] == 'feature' and not feat.get('verification_steps'):
        print(f"BLOCKED: feature '{feat['id']}' has no verification_steps", file=sys.stderr)
        sys.exit(1)

def cmd_post():
    """PostToolUse gate — record dirty file."""
    features = load_features()
    feat = active_feature(features)
    if not feat:
        return

    dirty_path = os.path.join(find_root(), '.lra_dirty')
    dirty = {}
    if os.path.exists(dirty_path):
        with open(dirty_path) as f:
            dirty = json.load(f)

    dirty['feature'] = feat['id']
    dirty.setdefault('files', [])
    # Append current file if passed as arg
    if len(sys.argv) > 2:
        fpath = sys.argv[2]
        if fpath not in dirty['files']:
            dirty['files'].append(fpath)

    with open(dirty_path, 'w') as f:
        json.dump(dirty, f)

def cmd_stop():
    """Stop gate — block if dirty or not verified."""
    root = find_root()
    dirty_path = os.path.join(root, '.lra_dirty')
    if os.path.exists(dirty_path):
        with open(dirty_path) as f:
            dirty = json.load(f)
        print(f"BLOCKED: .lra_dirty present for '{dirty.get('feature')}' — run lra-test.sh", file=sys.stderr)
        sys.exit(1)

    features = load_features()
    for item in features.get('active', []):
        if item.get('status') == 'done' and not item.get('passes'):
            print(f"BLOCKED: '{item['id']}' done but passes=false", file=sys.stderr)
            sys.exit(1)

def cmd_update():
    """Update integrity hashes after test pass."""
    done_hash, progress_hash, done_count, progress_count = compute_hashes()
    with open(os.path.join(find_root(), '.lra_done_hash'), 'w') as f:
        json.dump({
            'done_hash': done_hash, 'progress_hash': progress_hash,
            'done_count': done_count, 'progress_count': progress_count
        }, f)
    # Clear dirty
    dirty_path = os.path.join(find_root(), '.lra_dirty')
    if os.path.exists(dirty_path):
        os.remove(dirty_path)
    print(f"integrity: done={done_hash}({done_count}) progress={progress_hash}({progress_count})")

def cmd_health():
    """Health check."""
    root = find_root()
    ok = True
    checks = [
        ('feature_list.json exists', os.path.exists(os.path.join(root, 'feature_list.json'))),
        ('progress.md exists', os.path.exists(os.path.join(root, 'progress.md'))),
    ]
    for label, passed in checks:
        print(f"  [{'OK' if passed else 'FAIL'}] {label}")
        if not passed: ok = False

    # Check done features all pass
    features = load_features()
    for item in features.get('active', []):
        if item.get('status') == 'done' and not item.get('passes'):
            print(f"  [FAIL] {item['id']} done but passes=false")
            ok = False

    if not ok:
        sys.exit(1)

def cmd_version():
    """Check protocol version compatibility."""
    root = find_root()
    vpath = os.path.join(root, '.lra_version')
    if not os.path.exists(vpath):
        print("No .lra_version — run install.sh")
        sys.exit(1)
    with open(vpath) as f:
        v = json.load(f)
    print(f"Protocol: {v['protocol']} (installed: {v['installed_at']})")
    print(f"Compatible: {v['compatible']}")

if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'health'
    {'pre': cmd_pre, 'post': cmd_post, 'stop': cmd_stop,
     'update': cmd_update, 'health': cmd_health, 'version': cmd_version}.get(cmd, cmd_health)()
