"""Workspace management: create, cleanup, file I/O for per-user isolation."""

import json
import logging
import os
import shutil
import time

from server.config import WORKSPACE_ROOT, WORKSPACE_TTL_SECONDS

logger = logging.getLogger(__name__)


def create_workspace(username, session_id):
    """Create a per-user workspace directory. Returns the absolute path."""
    workspace_name = f"{username}_{session_id}"
    workspace_dir = os.path.join(WORKSPACE_ROOT, workspace_name)
    os.makedirs(workspace_dir, exist_ok=True)
    os.makedirs(os.path.join(workspace_dir, "logs"), exist_ok=True)

    meta_path = os.path.join(workspace_dir, ".workspace_meta.json")
    if not os.path.exists(meta_path):
        with open(meta_path, "w") as f:
            json.dump({"created_at": time.time(), "last_active": time.time()}, f)

    logger.info("Workspace created: %s", workspace_dir)
    return workspace_dir


def touch_workspace(workspace_dir):
    """Update last_active timestamp."""
    meta_path = os.path.join(workspace_dir, ".workspace_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
        meta["last_active"] = time.time()
        with open(meta_path, "w") as f:
            json.dump(meta, f)


def _validate_filename(filename):
    """Prevent path traversal attacks."""
    basename = os.path.basename(filename)
    if basename != filename or ".." in filename or filename.startswith("/"):
        return None, "Invalid filename"
    return basename, None


def save_file(workspace_dir, filename, content):
    """Save content to a file in the user's workspace.
    Returns the absolute path of the saved file.
    """
    filename, err = _validate_filename(filename)
    if err:
        raise ValueError(err)

    touch_workspace(workspace_dir)
    file_path = os.path.join(workspace_dir, filename)
    with open(file_path, "w") as f:
        f.write(content)
    logger.info("File saved: %s", file_path)
    return file_path


def load_file(workspace_dir, filename):
    """Load file content from the user's workspace.
    Returns (content, absolute_path).
    """
    filename, err = _validate_filename(filename)
    if err:
        raise ValueError(err)

    file_path = os.path.join(workspace_dir, filename)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {filename}")

    touch_workspace(workspace_dir)
    with open(file_path, "r") as f:
        content = f.read()
    return content, file_path


def list_files(workspace_dir):
    """List files in the user's workspace."""
    if not os.path.exists(workspace_dir):
        return []

    result = []
    for entry in os.listdir(workspace_dir):
        full_path = os.path.join(workspace_dir, entry)
        if os.path.isfile(full_path) and not entry.startswith("."):
            stat = os.stat(full_path)
            result.append({
                "name": entry,
                "size": stat.st_size,
                "mtime": stat.st_mtime,
            })
    return sorted(result, key=lambda x: x["mtime"], reverse=True)


def get_workspace_info(workspace_dir):
    """Get workspace summary info."""
    if not os.path.exists(workspace_dir):
        return {"workspace_dir": workspace_dir, "files": [], "disk_usage": 0}

    files = list_files(workspace_dir)
    total_size = sum(f["size"] for f in files)
    return {
        "workspace_dir": workspace_dir,
        "files": files,
        "disk_usage": total_size,
    }


def cleanup_stale_workspaces(max_age_seconds=WORKSPACE_TTL_SECONDS):
    """Remove workspaces inactive for longer than max_age_seconds.
    Returns count of removed workspaces.
    """
    if not os.path.exists(WORKSPACE_ROOT):
        return 0

    removed = 0
    now = time.time()
    for entry in os.listdir(WORKSPACE_ROOT):
        workspace_dir = os.path.join(WORKSPACE_ROOT, entry)
        if not os.path.isdir(workspace_dir):
            continue
        meta_path = os.path.join(workspace_dir, ".workspace_meta.json")
        if not os.path.exists(meta_path):
            continue
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            if now - meta.get("last_active", 0) > max_age_seconds:
                shutil.rmtree(workspace_dir, ignore_errors=True)
                removed += 1
                logger.info("Cleaned up stale workspace: %s", entry)
        except Exception as e:
            logger.warning("Error checking workspace %s: %s", entry, e)

    if removed:
        logger.info("Cleaned up %d stale workspace(s)", removed)
    return removed
