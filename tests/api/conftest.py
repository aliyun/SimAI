"""Pytest fixtures for SimAI API testing."""

import os
import shutil
import tempfile

import pytest

# Ensure DB_PATH uses a temp dir for test isolation
_test_workspace = None


@pytest.fixture(autouse=True)
def _isolate_workspace(monkeypatch, tmp_path):
    """Redirect WORKSPACE_ROOT and DB_PATH to a temp dir for every test."""
    workspace = str(tmp_path / "workspaces")
    os.makedirs(workspace, exist_ok=True)

    import server.config as cfg
    monkeypatch.setattr(cfg, "WORKSPACE_ROOT", workspace)

    import server.db.database as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", os.path.join(workspace, "test.db"))

    # Also patch workspace_service to use our temp dir
    try:
        import server.workspace.workspace_service as ws_mod
        monkeypatch.setattr(ws_mod, "WORKSPACE_ROOT", workspace)
    except (ImportError, AttributeError):
        pass


@pytest.fixture
def app():
    """Create Flask app for testing."""
    from server.app import create_app

    app = create_app()
    app.config["TESTING"] = True
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return app.test_client()


@pytest.fixture
def auth_token(client):
    """Create authenticated session and return token."""
    response = client.post(
        "/api/auth/login",
        json={"username": "testuser"},
    )
    assert response.status_code == 200
    data = response.get_json()
    return data["token"]


@pytest.fixture
def auth_headers(auth_token):
    """Headers dict with auth token."""
    return {
        "X-Session-Token": auth_token,
        "Content-Type": "application/json",
    }
