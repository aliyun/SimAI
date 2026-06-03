"""Authentication service: login, logout, token verification."""

import hashlib
import logging
import os
import re
import time
import uuid
from functools import wraps

from flask import request, jsonify

from server.config import AUTH_CONFIG_PATH
from server.db.database import get_db

logger = logging.getLogger(__name__)


def _load_users():
    """Load user credentials from auth_config.yaml.
    Returns empty dict for trust-based mode (any username accepted).
    """
    if not os.path.exists(AUTH_CONFIG_PATH):
        return {}
    try:
        import yaml
        with open(AUTH_CONFIG_PATH, "r") as f:
            config = yaml.safe_load(f)
        return config.get("users", {})
    except ImportError:
        logger.warning("PyYAML not installed; using trust-based auth")
        return {}
    except Exception as e:
        logger.error("Failed to load auth config: %s", e)
        return {}


def _hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def _validate_username(username):
    """Validate and normalize username."""
    if not username or not username.strip():
        return None, "Username cannot be empty"
    username = username.strip().lower()
    if len(username) > 32:
        return None, "Username too long (max 32 chars)"
    if not re.match(r"^[a-zA-Z0-9_]+$", username):
        return None, "Username: letters, numbers, underscore only"
    return username, None


def login(username, password=""):
    """Authenticate user and create a session.
    Returns (token, username, error).
    """
    username, err = _validate_username(username)
    if err:
        return None, None, err

    users = _load_users()
    if users:
        if username not in users:
            return None, None, "Unknown user"
        if _hash_password(password) != users[username]:
            return None, None, "Invalid password"

    token = str(uuid.uuid4())
    session_id = token[:12]

    # Create workspace directory path
    from server.workspace.workspace_service import create_workspace
    workspace_dir = create_workspace(username, session_id)

    with get_db() as conn:
        conn.execute(
            "INSERT INTO sessions (token, username, workspace_dir, created_at, last_active) "
            "VALUES (?, ?, ?, ?, ?)",
            (token, username, workspace_dir, time.time(), time.time()),
        )

    logger.info("User '%s' logged in, session %s", username, session_id)
    return token, username, None


def logout(token):
    """Remove a session."""
    with get_db() as conn:
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
    logger.info("Session logged out: %s...", token[:8])


def verify_token(token):
    """Verify session token. Returns (username, workspace_dir) or (None, None)."""
    if not token:
        return None, None
    with get_db() as conn:
        row = conn.execute(
            "SELECT username, workspace_dir FROM sessions WHERE token = ?",
            (token,),
        ).fetchone()
        if row:
            conn.execute(
                "UPDATE sessions SET last_active = ? WHERE token = ?",
                (time.time(), token),
            )
            return row["username"], row["workspace_dir"]
    return None, None


def require_auth(f):
    """Flask decorator: reject requests without a valid session token."""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get("X-Session-Token", "")
        username, workspace_dir = verify_token(token)
        if not username:
            return jsonify({"error": "Unauthorized"}), 401
        request.username = username
        request.workspace_dir = workspace_dir
        request.session_token = token
        return f(*args, **kwargs)
    return decorated
