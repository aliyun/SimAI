#!/usr/bin/env python3
"""Comprehensive tests for the SimAI multi-user backend and API client."""

import json
import os
import shutil
import signal
import sqlite3
import sys
import tempfile
import threading
import time
import unittest

# Set up project root (server/tests/ -> server/ -> SimAI/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# Override workspace root to use a temp directory for testing
TEST_WORKSPACE_ROOT = tempfile.mkdtemp(prefix="simai_test_")
os.environ["SIMAI_MULTIUSER"] = "0"  # Start in single-user, override per-test

# Monkey-patch config before importing server modules
import server.config as config
config.WORKSPACE_ROOT = TEST_WORKSPACE_ROOT
config.BIN_DIR = os.path.join(PROJECT_ROOT, "bin")

from server.db.database import get_db, init_db, DB_PATH
import server.db.database as db_module
db_module.DB_PATH = os.path.join(TEST_WORKSPACE_ROOT, "test_tracker.db")


class TestConfig(unittest.TestCase):
    """Test server/config.py."""

    def test_config_values(self):
        self.assertIsInstance(config.SERVER_PORT, int)
        self.assertEqual(config.SERVER_PORT, 5000)
        self.assertIsInstance(config.WORKSPACE_TTL_SECONDS, int)
        self.assertGreater(config.WORKSPACE_TTL_SECONDS, 0)
        self.assertIsInstance(config.MAX_PROCESSES_PER_USER, int)
        self.assertIsInstance(config.MAX_TOTAL_PROCESSES, int)
        self.assertGreater(config.MAX_PROCESSES_PER_USER, 0)
        self.assertGreater(config.MAX_TOTAL_PROCESSES, 0)

    def test_allowed_binaries(self):
        self.assertIsInstance(config.ALLOWED_BINARIES, set)
        self.assertIn("SimAI_oxc", config.ALLOWED_BINARIES)
        self.assertIn("SimAI_analytical", config.ALLOWED_BINARIES)
        self.assertIn("SimAI_simulator", config.ALLOWED_BINARIES)

    def test_paths(self):
        self.assertTrue(os.path.isabs(config.SERVER_DIR))
        self.assertTrue(os.path.isabs(config.PROJECT_ROOT))


class TestDatabase(unittest.TestCase):
    """Test server/db/database.py."""

    def setUp(self):
        # Clean DB before each test
        if os.path.exists(db_module.DB_PATH):
            os.remove(db_module.DB_PATH)

    def test_init_db(self):
        init_db()
        self.assertTrue(os.path.exists(db_module.DB_PATH))

    def test_get_db_context_manager(self):
        init_db()
        with get_db() as conn:
            self.assertIsNotNone(conn)
            # Check WAL mode
            mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
            self.assertEqual(mode, "wal")

    def test_schema_tables(self):
        init_db()
        with get_db() as conn:
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
            table_names = [t[0] for t in tables]
            self.assertIn("sessions", table_names)
            self.assertIn("processes", table_names)

    def test_sessions_columns(self):
        init_db()
        with get_db() as conn:
            info = conn.execute("PRAGMA table_info(sessions)").fetchall()
            col_names = [c[1] for c in info]
            for col in ["token", "username", "workspace_dir", "created_at", "last_active"]:
                self.assertIn(col, col_names, f"Missing column: {col}")

    def test_processes_columns(self):
        init_db()
        with get_db() as conn:
            info = conn.execute("PRAGMA table_info(processes)").fetchall()
            col_names = [c[1] for c in info]
            for col in ["id", "pid", "session_token", "username", "workspace_dir",
                         "command", "status", "started_at", "finished_at",
                         "return_code", "error_message"]:
                self.assertIn(col, col_names, f"Missing column: {col}")

    def test_insert_and_query_session(self):
        init_db()
        with get_db() as conn:
            conn.execute(
                "INSERT INTO sessions (token, username, workspace_dir, created_at, last_active) "
                "VALUES (?, ?, ?, ?, ?)",
                ("test-token", "alice", "/tmp/ws", time.time(), time.time()),
            )
        with get_db() as conn:
            row = conn.execute("SELECT * FROM sessions WHERE token = ?", ("test-token",)).fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row["username"], "alice")

    def test_rollback_on_error(self):
        init_db()
        try:
            with get_db() as conn:
                conn.execute(
                    "INSERT INTO sessions (token, username, workspace_dir, created_at, last_active) "
                    "VALUES (?, ?, ?, ?, ?)",
                    ("rollback-test", "bob", "/tmp", time.time(), time.time()),
                )
                raise ValueError("force rollback")
        except ValueError:
            pass
        with get_db() as conn:
            row = conn.execute("SELECT * FROM sessions WHERE token = ?", ("rollback-test",)).fetchone()
            self.assertIsNone(row)

    def test_concurrent_access(self):
        init_db()
        errors = []

        def writer(i):
            try:
                with get_db() as conn:
                    conn.execute(
                        "INSERT INTO sessions (token, username, workspace_dir, created_at, last_active) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (f"concurrent-{i}", f"user{i}", "/tmp", time.time(), time.time()),
                    )
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Concurrent write errors: {errors}")
        with get_db() as conn:
            count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            self.assertEqual(count, 10)


class TestWorkspaceService(unittest.TestCase):
    """Test server/workspace/workspace_service.py."""

    def setUp(self):
        # Clean workspace root
        if os.path.exists(TEST_WORKSPACE_ROOT):
            for entry in os.listdir(TEST_WORKSPACE_ROOT):
                path = os.path.join(TEST_WORKSPACE_ROOT, entry)
                if os.path.isdir(path) and not entry.endswith(".db"):
                    shutil.rmtree(path)

    def test_create_workspace(self):
        from server.workspace.workspace_service import create_workspace
        ws = create_workspace("testuser", "abc123")
        self.assertTrue(os.path.isdir(ws))
        self.assertIn("testuser_abc123", ws)
        self.assertTrue(os.path.isdir(os.path.join(ws, "logs")))
        meta_path = os.path.join(ws, ".workspace_meta.json")
        self.assertTrue(os.path.exists(meta_path))
        with open(meta_path) as f:
            meta = json.load(f)
        self.assertIn("created_at", meta)
        self.assertIn("last_active", meta)

    def test_create_workspace_idempotent(self):
        from server.workspace.workspace_service import create_workspace
        ws1 = create_workspace("user1", "sess1")
        ws2 = create_workspace("user1", "sess1")
        self.assertEqual(ws1, ws2)

    def test_save_and_load_file(self):
        from server.workspace.workspace_service import create_workspace, save_file, load_file
        ws = create_workspace("filetest", "s1")
        path = save_file(ws, "test.txt", "hello world")
        self.assertTrue(os.path.exists(path))
        content, loaded_path = load_file(ws, "test.txt")
        self.assertEqual(content, "hello world")
        self.assertEqual(path, loaded_path)

    def test_save_file_path_traversal(self):
        from server.workspace.workspace_service import create_workspace, save_file
        ws = create_workspace("security", "s1")
        with self.assertRaises(ValueError):
            save_file(ws, "../etc/passwd", "hack")
        with self.assertRaises(ValueError):
            save_file(ws, "/etc/passwd", "hack")
        with self.assertRaises(ValueError):
            save_file(ws, "../../secret.txt", "hack")
        with self.assertRaises(ValueError):
            save_file(ws, "sub/dir/file.txt", "hack")

    def test_load_file_not_found(self):
        from server.workspace.workspace_service import create_workspace, load_file
        ws = create_workspace("loadtest", "s1")
        with self.assertRaises(FileNotFoundError):
            load_file(ws, "nonexistent.txt")

    def test_load_file_path_traversal(self):
        from server.workspace.workspace_service import create_workspace, load_file
        ws = create_workspace("loadattack", "s1")
        with self.assertRaises(ValueError):
            load_file(ws, "../../../etc/passwd")

    def test_list_files(self):
        from server.workspace.workspace_service import create_workspace, save_file, list_files
        ws = create_workspace("listtest", "s1")
        save_file(ws, "a.txt", "aaa")
        save_file(ws, "b.json", '{"x": 1}')
        files = list_files(ws)
        names = [f["name"] for f in files]
        self.assertIn("a.txt", names)
        self.assertIn("b.json", names)
        # Hidden files should be excluded
        for f in files:
            self.assertFalse(f["name"].startswith("."))

    def test_list_files_empty(self):
        from server.workspace.workspace_service import list_files
        files = list_files("/nonexistent/path")
        self.assertEqual(files, [])

    def test_get_workspace_info(self):
        from server.workspace.workspace_service import (
            create_workspace, save_file, get_workspace_info
        )
        ws = create_workspace("infotest", "s1")
        save_file(ws, "data.csv", "col1,col2\n1,2\n")
        info = get_workspace_info(ws)
        self.assertEqual(info["workspace_dir"], ws)
        self.assertGreater(len(info["files"]), 0)
        self.assertGreater(info["disk_usage"], 0)

    def test_cleanup_stale_workspaces(self):
        from server.workspace.workspace_service import (
            create_workspace, cleanup_stale_workspaces
        )
        ws = create_workspace("stale", "s1")
        # Manipulate last_active to be very old
        meta_path = os.path.join(ws, ".workspace_meta.json")
        with open(meta_path, "w") as f:
            json.dump({"created_at": 1000, "last_active": 1000}, f)
        removed = cleanup_stale_workspaces(max_age_seconds=1)
        self.assertEqual(removed, 1)
        self.assertFalse(os.path.exists(ws))

    def test_cleanup_keeps_recent_workspaces(self):
        from server.workspace.workspace_service import (
            create_workspace, cleanup_stale_workspaces
        )
        ws = create_workspace("recent", "s1")
        removed = cleanup_stale_workspaces(max_age_seconds=3600)
        self.assertEqual(removed, 0)
        self.assertTrue(os.path.exists(ws))

    def test_touch_workspace(self):
        from server.workspace.workspace_service import create_workspace, touch_workspace
        ws = create_workspace("touchtest", "s1")
        meta_path = os.path.join(ws, ".workspace_meta.json")
        with open(meta_path) as f:
            old_meta = json.load(f)
        time.sleep(0.1)
        touch_workspace(ws)
        with open(meta_path) as f:
            new_meta = json.load(f)
        self.assertGreater(new_meta["last_active"], old_meta["last_active"])


class TestAuthService(unittest.TestCase):
    """Test server/auth/auth_service.py."""

    def setUp(self):
        if os.path.exists(db_module.DB_PATH):
            os.remove(db_module.DB_PATH)
        init_db()

    def test_login_trust_mode(self):
        from server.auth.auth_service import login
        token, username, err = login("alice")
        self.assertIsNone(err)
        self.assertIsNotNone(token)
        self.assertEqual(username, "alice")

    def test_login_normalizes_username(self):
        from server.auth.auth_service import login
        token, username, err = login("  Alice  ")
        self.assertIsNone(err)
        self.assertEqual(username, "alice")

    def test_login_empty_username(self):
        from server.auth.auth_service import login
        token, username, err = login("")
        self.assertIsNotNone(err)
        self.assertIn("empty", err.lower())

    def test_login_invalid_username_chars(self):
        from server.auth.auth_service import login
        token, username, err = login("al!ce@")
        self.assertIsNotNone(err)

    def test_login_too_long_username(self):
        from server.auth.auth_service import login
        token, username, err = login("a" * 33)
        self.assertIsNotNone(err)
        self.assertIn("long", err.lower())

    def test_login_creates_workspace(self):
        from server.auth.auth_service import login
        token, username, err = login("wscheck")
        self.assertIsNone(err)
        # Check session has workspace_dir
        with get_db() as conn:
            row = conn.execute("SELECT workspace_dir FROM sessions WHERE token = ?", (token,)).fetchone()
            self.assertIsNotNone(row)
            self.assertTrue(os.path.isdir(row["workspace_dir"]))

    def test_verify_token_valid(self):
        from server.auth.auth_service import login, verify_token
        token, _, _ = login("verifyuser")
        username, ws = verify_token(token)
        self.assertEqual(username, "verifyuser")
        self.assertIsNotNone(ws)

    def test_verify_token_invalid(self):
        from server.auth.auth_service import verify_token
        username, ws = verify_token("nonexistent-token")
        self.assertIsNone(username)
        self.assertIsNone(ws)

    def test_verify_token_empty(self):
        from server.auth.auth_service import verify_token
        username, ws = verify_token("")
        self.assertIsNone(username)

    def test_verify_token_none(self):
        from server.auth.auth_service import verify_token
        username, ws = verify_token(None)
        self.assertIsNone(username)

    def test_logout(self):
        from server.auth.auth_service import login, logout, verify_token
        token, _, _ = login("logoutuser")
        logout(token)
        username, ws = verify_token(token)
        self.assertIsNone(username)

    def test_multiple_sessions_same_user(self):
        from server.auth.auth_service import login, verify_token
        token1, _, _ = login("multiuser")
        token2, _, _ = login("multiuser")
        self.assertNotEqual(token1, token2)
        # Both should be valid
        u1, _ = verify_token(token1)
        u2, _ = verify_token(token2)
        self.assertEqual(u1, "multiuser")
        self.assertEqual(u2, "multiuser")

    def test_verify_updates_last_active(self):
        from server.auth.auth_service import login, verify_token
        token, _, _ = login("activeuser")
        with get_db() as conn:
            row = conn.execute("SELECT last_active FROM sessions WHERE token = ?", (token,)).fetchone()
            first_active = row["last_active"]
        time.sleep(0.1)
        verify_token(token)
        with get_db() as conn:
            row = conn.execute("SELECT last_active FROM sessions WHERE token = ?", (token,)).fetchone()
            second_active = row["last_active"]
        self.assertGreater(second_active, first_active)


class TestProcessService(unittest.TestCase):
    """Test server/process/process_service.py."""

    def setUp(self):
        if os.path.exists(db_module.DB_PATH):
            os.remove(db_module.DB_PATH)
        init_db()

    def test_check_limits_allows(self):
        from server.process.process_service import check_limits
        allowed, reason = check_limits("newuser")
        self.assertTrue(allowed)
        self.assertEqual(reason, "OK")

    def test_check_limits_per_user(self):
        from server.process.process_service import check_limits
        # Insert MAX_PROCESSES_PER_USER running processes
        with get_db() as conn:
            for i in range(config.MAX_PROCESSES_PER_USER):
                conn.execute(
                    "INSERT INTO processes (pid, session_token, username, workspace_dir, command, status, started_at) "
                    "VALUES (?, ?, ?, ?, ?, 'running', ?)",
                    (90000 + i, "tok", "limituser", "/tmp", "cmd", time.time()),
                )
        allowed, reason = check_limits("limituser")
        self.assertFalse(allowed)
        self.assertIn("Per-user", reason)

    def test_check_limits_total(self):
        from server.process.process_service import check_limits
        with get_db() as conn:
            for i in range(config.MAX_TOTAL_PROCESSES):
                conn.execute(
                    "INSERT INTO processes (pid, session_token, username, workspace_dir, command, status, started_at) "
                    "VALUES (?, ?, ?, ?, ?, 'running', ?)",
                    (80000 + i, "tok", f"user{i}", "/tmp", "cmd", time.time()),
                )
        allowed, reason = check_limits("newuser99")
        self.assertFalse(allowed)
        self.assertIn("System", reason)

    def test_launch_invalid_binary(self):
        from server.process.process_service import launch
        _, _, err = launch("tok", "user", "/tmp", {"binary": "rm"})
        self.assertIsNotNone(err)
        self.assertIn("not allowed", err)

    def test_launch_missing_binary(self):
        from server.process.process_service import launch
        _, _, err = launch("tok", "user", "/tmp", {"binary": "SimAI_oxc"})
        self.assertIsNotNone(err)
        self.assertIn("not found", err)

    def test_launch_missing_workload(self):
        from server.process.process_service import launch
        from server.workspace.workspace_service import create_workspace
        ws = create_workspace("launchtest", "s1")
        fake_bin = os.path.join(config.BIN_DIR, "SimAI_oxc")
        os.makedirs(config.BIN_DIR, exist_ok=True)
        if os.path.islink(fake_bin):
            os.unlink(fake_bin)
        if not os.path.exists(fake_bin):
            with open(fake_bin, "w") as f:
                f.write("#!/bin/bash\necho test\n")
            os.chmod(fake_bin, 0o755)
        _, _, err = launch("tok", "launchtest", ws, {
            "binary": "SimAI_oxc",
            "workload_file": "nonexistent.txt"
        })
        self.assertIsNotNone(err)
        self.assertIn("Workload", err)

    def test_launch_real_process(self):
        """Launch a real process (echo) using a whitelisted binary name."""
        from server.process.process_service import launch, list_processes, get_logs
        from server.workspace.workspace_service import create_workspace, save_file

        ws = create_workspace("reallaunch", "s1")
        save_file(ws, "workload.txt", "fake workload")
        save_file(ws, "ranktable.json", '{"rank_count": 1}')

        fake_bin = os.path.join(config.BIN_DIR, "SimAI_oxc")
        os.makedirs(config.BIN_DIR, exist_ok=True)
        if os.path.islink(fake_bin):
            os.unlink(fake_bin)
        if not os.path.exists(fake_bin):
            with open(fake_bin, "w") as f:
                f.write("#!/bin/bash\necho 'starting simulation'\nsleep 0.5\necho 'done'\n")
        os.chmod(fake_bin, 0o755)

        tid, pid, err = launch("tok123", "reallaunch", ws, {
            "binary": "SimAI_oxc",
            "workload_file": "workload.txt",
            "ranktable_file": "ranktable.json",
            "timeout": 10
        })
        self.assertIsNone(err, f"Launch error: {err}")
        self.assertIsNotNone(pid)
        self.assertIsNotNone(tid)

        # Wait for process to finish
        time.sleep(2)

        # Check process list
        procs = list_processes(username="reallaunch")
        self.assertGreater(len(procs), 0)

        # Check logs
        logs, status, log_err = get_logs("reallaunch", pid)
        self.assertIsNone(log_err)
        self.assertIn(status, ["finished", "running"])

    def test_kill_process_wrong_user(self):
        from server.process.process_service import kill_process
        with get_db() as conn:
            conn.execute(
                "INSERT INTO processes (pid, session_token, username, workspace_dir, command, status, started_at) "
                "VALUES (?, ?, ?, ?, ?, 'running', ?)",
                (99999, "tok", "alice", "/tmp", "cmd", time.time()),
            )
        success, err = kill_process("bob", 99999)
        self.assertFalse(success)
        self.assertIn("Permission", err)

    def test_kill_nonexistent_process(self):
        from server.process.process_service import kill_process
        success, err = kill_process("anyone", 11111)
        self.assertFalse(success)
        self.assertIn("not found", err.lower())

    def test_cleanup_dead_processes(self):
        from server.process.process_service import cleanup_dead_processes
        # Insert a process with a PID that definitely doesn't exist
        with get_db() as conn:
            conn.execute(
                "INSERT INTO processes (pid, session_token, username, workspace_dir, command, status, started_at) "
                "VALUES (?, ?, ?, ?, ?, 'running', ?)",
                (99998, "tok", "user", "/tmp", "cmd", time.time()),
            )
        cleaned = cleanup_dead_processes()
        self.assertGreaterEqual(cleaned, 1)
        with get_db() as conn:
            row = conn.execute("SELECT status FROM processes WHERE pid = 99998").fetchone()
            self.assertEqual(row["status"], "dead")

    def test_update_status(self):
        from server.process.process_service import update_status
        with get_db() as conn:
            conn.execute(
                "INSERT INTO processes (pid, session_token, username, workspace_dir, command, status, started_at) "
                "VALUES (?, ?, ?, ?, ?, 'running', ?)",
                (77777, "tok", "user", "/tmp", "cmd", time.time()),
            )
        update_status(77777, "finished", return_code=0)
        with get_db() as conn:
            row = conn.execute("SELECT status, return_code FROM processes WHERE pid = 77777").fetchone()
            self.assertEqual(row["status"], "finished")
            self.assertEqual(row["return_code"], 0)

    def test_get_logs_wrong_user(self):
        from server.process.process_service import get_logs
        with get_db() as conn:
            conn.execute(
                "INSERT INTO processes (pid, session_token, username, workspace_dir, command, status, started_at) "
                "VALUES (?, ?, ?, ?, ?, 'running', ?)",
                (66666, "tok", "alice", "/tmp", "cmd", time.time()),
            )
        logs, status, err = get_logs("bob", 66666)
        self.assertIsNotNone(err)
        self.assertIn("Permission", err)


class TestFlaskApp(unittest.TestCase):
    """Test server/app.py Flask API endpoints."""

    @classmethod
    def setUpClass(cls):
        if os.path.exists(db_module.DB_PATH):
            os.remove(db_module.DB_PATH)
        from server.app import create_app
        cls.app = create_app()
        cls.client = cls.app.test_client()

    def test_health_check(self):
        resp = self.client.get("/healthz")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "ok")

    def test_login(self):
        resp = self.client.post("/api/auth/login",
                                json={"username": "testflask", "password": ""})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("token", data)
        self.assertEqual(data["username"], "testflask")
        return data["token"]

    def test_login_empty_username(self):
        resp = self.client.post("/api/auth/login", json={"username": ""})
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertIn("error", data)

    def test_login_invalid_username(self):
        resp = self.client.post("/api/auth/login", json={"username": "a@b!c"})
        self.assertEqual(resp.status_code, 400)

    def _get_token(self, username="apiuser"):
        resp = self.client.post("/api/auth/login",
                                json={"username": username, "password": ""})
        return resp.get_json()["token"]

    def test_auth_me(self):
        token = self._get_token("metest")
        resp = self.client.get("/api/auth/me",
                               headers={"X-Session-Token": token})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["username"], "metest")
        self.assertIn("workspace", data)

    def test_auth_me_no_token(self):
        resp = self.client.get("/api/auth/me")
        self.assertEqual(resp.status_code, 401)

    def test_auth_me_bad_token(self):
        resp = self.client.get("/api/auth/me",
                               headers={"X-Session-Token": "invalid-token"})
        self.assertEqual(resp.status_code, 401)

    def test_logout(self):
        token = self._get_token("logoutflask")
        resp = self.client.post("/api/auth/logout",
                                headers={"X-Session-Token": token})
        self.assertEqual(resp.status_code, 200)
        # Token should be invalid now
        resp = self.client.get("/api/auth/me",
                               headers={"X-Session-Token": token})
        self.assertEqual(resp.status_code, 401)

    def test_save_file(self):
        token = self._get_token("filesaver")
        resp = self.client.post("/api/files/save",
                                headers={"X-Session-Token": token},
                                json={"filename": "workload.txt", "content": "test content"})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("path", data)
        self.assertEqual(data["filename"], "workload.txt")

    def test_save_file_no_filename(self):
        token = self._get_token("nofile")
        resp = self.client.post("/api/files/save",
                                headers={"X-Session-Token": token},
                                json={"content": "test"})
        self.assertEqual(resp.status_code, 400)

    def test_save_file_path_traversal(self):
        token = self._get_token("traversal")
        resp = self.client.post("/api/files/save",
                                headers={"X-Session-Token": token},
                                json={"filename": "../../../etc/passwd", "content": "hack"})
        self.assertEqual(resp.status_code, 400)

    def test_load_file(self):
        token = self._get_token("fileloader")
        # Save first
        self.client.post("/api/files/save",
                         headers={"X-Session-Token": token},
                         json={"filename": "test.json", "content": '{"key": "value"}'})
        # Load
        resp = self.client.get("/api/files/load",
                               headers={"X-Session-Token": token},
                               query_string={"filename": "test.json"})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["content"], '{"key": "value"}')

    def test_load_file_not_found(self):
        token = self._get_token("loader404")
        resp = self.client.get("/api/files/load",
                               headers={"X-Session-Token": token},
                               query_string={"filename": "nonexistent.txt"})
        self.assertEqual(resp.status_code, 404)

    def test_list_files(self):
        token = self._get_token("lister")
        self.client.post("/api/files/save",
                         headers={"X-Session-Token": token},
                         json={"filename": "a.txt", "content": "aaa"})
        self.client.post("/api/files/save",
                         headers={"X-Session-Token": token},
                         json={"filename": "b.txt", "content": "bbb"})
        resp = self.client.get("/api/files/list",
                               headers={"X-Session-Token": token})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        names = [f["name"] for f in data["files"]]
        self.assertIn("a.txt", names)
        self.assertIn("b.txt", names)

    def test_workspace_info(self):
        token = self._get_token("wsinfo")
        self.client.post("/api/files/save",
                         headers={"X-Session-Token": token},
                         json={"filename": "data.csv", "content": "col1,col2"})
        resp = self.client.get("/api/workspace/info",
                               headers={"X-Session-Token": token})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("workspace_dir", data)
        self.assertIn("files", data)
        self.assertIn("disk_usage", data)

    def test_process_launch_no_binary(self):
        token = self._get_token("launcher")
        resp = self.client.post("/api/process/launch",
                                headers={"X-Session-Token": token},
                                json={"binary": "not_allowed_binary"})
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertIn("error", data)

    def test_process_list(self):
        token = self._get_token("proclist")
        resp = self.client.get("/api/process/list",
                               headers={"X-Session-Token": token})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("processes", data)
        self.assertIsInstance(data["processes"], list)

    def test_process_kill_not_found(self):
        token = self._get_token("prockill")
        resp = self.client.delete("/api/process/99999",
                                  headers={"X-Session-Token": token})
        self.assertIn(resp.status_code, [403, 404])

    def test_process_logs_not_found(self):
        token = self._get_token("proclogs")
        resp = self.client.get("/api/process/logs/99999",
                               headers={"X-Session-Token": token})
        self.assertEqual(resp.status_code, 404)

    def test_file_isolation_between_users(self):
        """Two users saving the same filename should not interfere."""
        token_a = self._get_token("usera")
        token_b = self._get_token("userb")

        self.client.post("/api/files/save",
                         headers={"X-Session-Token": token_a},
                         json={"filename": "shared.txt", "content": "user_a_data"})
        self.client.post("/api/files/save",
                         headers={"X-Session-Token": token_b},
                         json={"filename": "shared.txt", "content": "user_b_data"})

        resp_a = self.client.get("/api/files/load",
                                 headers={"X-Session-Token": token_a},
                                 query_string={"filename": "shared.txt"})
        resp_b = self.client.get("/api/files/load",
                                 headers={"X-Session-Token": token_b},
                                 query_string={"filename": "shared.txt"})

        self.assertEqual(resp_a.get_json()["content"], "user_a_data")
        self.assertEqual(resp_b.get_json()["content"], "user_b_data")

    def test_cross_user_cannot_access_other_workspace(self):
        """User B cannot access User A's workspace via token."""
        token_a = self._get_token("isolatea")
        token_b = self._get_token("isolateb")

        self.client.post("/api/files/save",
                         headers={"X-Session-Token": token_a},
                         json={"filename": "secret.txt", "content": "private_data"})

        # User B should not see User A's files
        resp = self.client.get("/api/files/load",
                               headers={"X-Session-Token": token_b},
                               query_string={"filename": "secret.txt"})
        self.assertEqual(resp.status_code, 404)


class TestServerSyntax(unittest.TestCase):
    """Test server Python files for syntax correctness."""

    def _check_syntax(self, filepath):
        with open(filepath, "r") as f:
            source = f.read()
        try:
            compile(source, filepath, "exec")
        except SyntaxError as e:
            self.fail(f"Syntax error in {filepath}: {e}")

    def test_server_app(self):
        self._check_syntax(os.path.join(PROJECT_ROOT, "server", "app.py"))

    def test_server_config(self):
        self._check_syntax(os.path.join(PROJECT_ROOT, "server", "config.py"))

    def test_server_database(self):
        self._check_syntax(os.path.join(PROJECT_ROOT, "server", "db", "database.py"))

    def test_server_auth(self):
        self._check_syntax(os.path.join(PROJECT_ROOT, "server", "auth", "auth_service.py"))

    def test_server_workspace(self):
        self._check_syntax(os.path.join(PROJECT_ROOT, "server", "workspace", "workspace_service.py"))

    def test_server_process(self):
        self._check_syntax(os.path.join(PROJECT_ROOT, "server", "process", "process_service.py"))


class TestGitignore(unittest.TestCase):
    """Test .gitignore contains all necessary entries."""

    def test_gitignore_entries(self):
        gitignore_path = os.path.join(PROJECT_ROOT, ".gitignore")
        with open(gitignore_path) as f:
            content = f.read()
        for entry in ["server/workspaces/", "server/auth_config.yaml",
                       "deploy/.simai_pids"]:
            self.assertIn(entry, content, f"Missing .gitignore entry: {entry}")


def cleanup():
    """Clean up test workspace."""
    if os.path.exists(TEST_WORKSPACE_ROOT):
        shutil.rmtree(TEST_WORKSPACE_ROOT, ignore_errors=True)


if __name__ == "__main__":
    try:
        # Run with verbose output
        unittest.main(verbosity=2, exit=False)
    finally:
        cleanup()
