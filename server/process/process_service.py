"""Process management: launch, kill, track simulation subprocesses."""

import logging
import os
import signal
import subprocess
import threading
import time

from server.config import (
    ALLOWED_BINARIES,
    BIN_DIR,
    MAX_PROCESSES_PER_USER,
    MAX_TOTAL_PROCESSES,
)
from server.db.database import get_db

logger = logging.getLogger(__name__)

# In-memory log buffer: pid -> list of log lines
_log_buffers = {}
_log_lock = threading.Lock()

MAX_LOG_LINES = 500

# Default NS3 config template path
_DEFAULT_NS3_CONF = os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir,
    "astra-sim-alibabacloud", "inputs", "config", "SimAI.conf",
)


def _ensure_local_ns3_config(workspace_dir, config_file=None):
    """Generate a workspace-local NS3 config with valid output file paths.

    The stock SimAI.conf points output files to /etc/astra-sim/simulation/
    which doesn't exist on macOS (or any non-Docker host), causing a segfault
    when NS3 tries to fopen(NULL).  This function copies the config into the
    workspace and rewrites all output paths to workspace/ns3_output/.
    """
    # Resolve source config (avoid CWD-dependent os.path.abspath)
    if config_file:
        src = os.path.join(workspace_dir, config_file)
        if not os.path.isfile(src):
            # Try relative to PROJECT_ROOT first, fall back to absolute
            from server.config import PROJECT_ROOT
            src = os.path.join(PROJECT_ROOT, config_file)
            if not os.path.isfile(src):
                src = os.path.abspath(config_file)  # last resort
    else:
        src = os.path.abspath(_DEFAULT_NS3_CONF)

    if not os.path.isfile(src):
        # Fallback: let NS3 fail with a clear error rather than segfault
        logger.warning("NS3 config not found: %s", src)
        return src

    out_dir = os.path.join(workspace_dir, "ns3_output")
    os.makedirs(out_dir, exist_ok=True)

    local_conf = os.path.join(workspace_dir, "SimAI_local.conf")

    # Keys whose values are file paths that need rewriting
    path_keys = {
        "FLOW_FILE", "TRACE_FILE", "TRACE_OUTPUT_FILE",
        "FCT_OUTPUT_FILE", "PFC_OUTPUT_FILE",
        "QLEN_MON_FILE", "BW_MON_FILE", "RATE_MON_FILE", "CNP_MON_FILE",
    }

    with open(src) as f:
        lines = f.readlines()

    with open(local_conf, "w") as f:
        for line in lines:
            parts = line.split()
            if len(parts) >= 2 and parts[0] in path_keys:
                basename = os.path.basename(parts[1])
                parts[1] = os.path.join(out_dir, basename)
                f.write(" ".join(parts) + "\n")
            else:
                f.write(line)

    logger.info("Generated local NS3 config: %s", local_conf)
    return local_conf


def _stream_output(pid, process, log_path, workspace_dir=None, output_prefix=None):
    """Background thread: read subprocess stdout and buffer logs."""
    lines = []
    try:
        with open(log_path, "w") as log_file:
            for line in iter(process.stdout.readline, ""):
                line = line.rstrip("\n")
                log_file.write(line + "\n")
                log_file.flush()
                with _log_lock:
                    buf = _log_buffers.get(pid, [])
                    buf.append(line)
                    if len(buf) > MAX_LOG_LINES:
                        buf = buf[-MAX_LOG_LINES:]
                    _log_buffers[pid] = buf
    except Exception as e:
        logger.error("Log streaming error for PID %d: %s", pid, e)
    finally:
        process.wait()
        rc = process.returncode
        status = "finished" if rc == 0 else "error"
        error_msg = _signal_name(rc) if rc < 0 else None
        update_status(pid, status, rc, error_message=error_msg)
        logger.info("Process %d finished with code %d", pid, rc)
        # Rename NS3 output files to prevent cross-run clobbering.
        # NS3 hardcodes RESULT_PATH="./ncclFlowModel_" — every run writes
        # ncclFlowModel_EndToEnd.csv / _detailed_*.csv in the same workspace.
        if workspace_dir and output_prefix:
            _rename_ns3_outputs(workspace_dir, output_prefix)


def _cleanup_ns3_outputs(workspace_dir):
    """Remove stale ncclFlowModel_* files before starting a new NS3 run.

    Prevents ghost results: if the previous run crashed before writing
    its EndToEnd CSV, the stale file would otherwise be renamed as the
    new task's result.
    """
    for old_name in os.listdir(workspace_dir):
        if not old_name.startswith("ncclFlowModel_"):
            continue
        old_path = os.path.join(workspace_dir, old_name)
        if not os.path.isfile(old_path):
            continue
        try:
            os.remove(old_path)
            logger.info("Cleaned up stale NS3 output: %s", old_name)
        except OSError as e:
            logger.warning("Failed to remove %s: %s", old_name, e)


def _rename_ns3_outputs(workspace_dir, output_prefix):
    """Rename hardcoded ncclFlowModel_* files to use the task's output prefix."""
    for old_name in os.listdir(workspace_dir):
        if not old_name.startswith("ncclFlowModel_"):
            continue
        old_path = os.path.join(workspace_dir, old_name)
        if not os.path.isfile(old_path):
            continue
        # Skip empty files — the simulation crashed before writing output.
        if os.path.getsize(old_path) == 0:
            logger.warning("Skipping empty NS3 output: %s (simulation may have crashed)", old_name)
            continue
        # ncclFlowModel_EndToEnd.csv → {prefix}_EndToEnd.csv
        # ncclFlowModel_detailed_9.csv  → {prefix}_detailed_9.csv
        suffix = old_name[len("ncclFlowModel"):]  # _EndToEnd.csv, _detailed_9.csv
        new_name = output_prefix + suffix
        new_path = os.path.join(workspace_dir, new_name)
        try:
            os.rename(old_path, new_path)
            logger.info("Renamed NS3 output: %s -> %s", old_name, new_name)
        except OSError as e:
            logger.warning("Failed to rename %s: %s", old_name, e)


def _signal_name(return_code):
    """Translate negative return codes (signal numbers) to human-readable names."""
    if return_code is None or return_code >= 0:
        return None
    sig = -return_code
    names = {
        6: "进程异常终止 (SIGABRT)",
        9: "内存不足被系统终止 (SIGKILL)",
        11: "内存访问错误 (SIGSEGV)",
        15: "超时被终止 (SIGTERM)",
    }
    if sig in names:
        return names[sig]
    try:
        return f"信号异常退出: {signal.Signals(sig).name}"
    except (ValueError, AttributeError):
        return f"异常退出 (信号 {sig})"


def check_limits(username):
    """Check if a user can launch a new process.
    Returns (allowed, reason).
    """
    with get_db() as conn:
        user_count = conn.execute(
            "SELECT COUNT(*) FROM processes WHERE username = ? AND status = 'running'",
            (username,),
        ).fetchone()[0]
        total_count = conn.execute(
            "SELECT COUNT(*) FROM processes WHERE status = 'running'"
        ).fetchone()[0]

    if user_count >= MAX_PROCESSES_PER_USER:
        return False, f"Per-user limit reached ({MAX_PROCESSES_PER_USER} max)"
    if total_count >= MAX_TOTAL_PROCESSES:
        return False, f"System limit reached ({MAX_TOTAL_PROCESSES} max)"
    return True, "OK"


def launch(session_token, username, workspace_dir, params):
    """Launch a simulation subprocess.

    params: {
        "binary": "SimAI_oxc",
        "workload_file": "workload.txt",       # relative to workspace
        "ranktable_file": "ranktable.json",     # relative to workspace
        "topology_file": "topology_16g_...",    # relative to workspace
        "output_prefix": "oxc_output",          # relative to workspace
        "env_vars": { "AS_LOG_LEVEL": "INFO", ... },
        "timeout": 0,
        "extra_args": []                        # additional CLI args
    }

    Returns (tracking_id, pid, error).
    """
    allowed, reason = check_limits(username)
    if not allowed:
        return None, None, reason

    binary_name = params.get("binary", "")
    if binary_name not in ALLOWED_BINARIES:
        return None, None, f"Binary not allowed: {binary_name}"

    binary_path = os.path.join(BIN_DIR, binary_name)
    if not os.path.exists(binary_path):
        return None, None, f"Binary not found: {binary_path}"

    # Build absolute paths within workspace
    workload_path = os.path.join(workspace_dir, params.get("workload_file", "workload.txt"))
    ranktable_path = os.path.join(workspace_dir, params.get("ranktable_file", "ranktable.json"))
    output_prefix = os.path.join(workspace_dir, params.get("output_prefix",
        f"sim_result_{int(time.time())}"))

    if not os.path.exists(workload_path):
        return None, None, f"Workload file not found: {params.get('workload_file')}"
    if not os.path.exists(ranktable_path):
        return None, None, f"Ranktable file not found: {params.get('ranktable_file')}"

    # Build command — different binaries have different CLI interfaces.
    # SimAI_oxc (OxcMain.cc) uses: -w -o -ranktable -g -g_p_s
    # SimAI_simulator* (AstraSimNetwork[_oxc]) uses getopt: -t -w -n -c (NO -r)
    # SimAI_analytical* (AstraParamParse) uses: -w -g -g_p_s -r [...]
    # Note: SimAI_analytical_oxc uses AstraParamParse, NOT OxcMain!
    is_oxc = (binary_name == "SimAI_oxc")
    is_ns3 = binary_name.startswith("SimAI_simulator")
    is_analytical = binary_name.startswith("SimAI_analytical")

    # NS3 hardcodes RESULT_PATH="./ncclFlowModel_" — concurrent runs clobber.
    if is_ns3:
        with get_db() as conn:
            running_ns3 = conn.execute(
                "SELECT COUNT(*) FROM processes "
                "WHERE workspace_dir = ? AND status = 'running' "
                "AND command LIKE '%SimAI_simulator%'",
                (workspace_dir,),
            ).fetchone()[0]
        if running_ns3 > 0:
            return None, None, (
                f"已有 {running_ns3} 个 NS3 仿真在运行，NS3 不支持并发，请等待完成"
            )

    if is_oxc:
        # OXC analytical binary (OxcMain.cc): -w -o -ranktable -g -g_p_s [-oxc_url -oxc_algo]
        cmd = [binary_path, "-w", workload_path, "-o", output_prefix,
               "-ranktable", ranktable_path,
               "-g", str(params.get("num_gpus", 32)),
               "-g_p_s", str(params.get("gpus_per_server", 8))]
        # Pass through OXC-specific args, strip any -c/-n/-t that don't apply
        extra_args = params.get("extra_args", [])
        skip_next = False
        for i, arg in enumerate(extra_args):
            if skip_next:
                skip_next = False
                continue
            if arg in ("-c", "-n", "-t") and i + 1 < len(extra_args):
                skip_next = True  # skip these NS3-only flags
            elif arg.startswith("-oxc_"):
                cmd.extend([arg, extra_args[i + 1]] if i + 1 < len(extra_args) else [arg])
                skip_next = True
            # ignore other unknown flags for OXC
    elif is_ns3:
        # NS3 / NS3-OXC binary (AstraSimNetwork[_oxc]): unistd getopt -t -w -n -c
        # Does NOT accept -r; output paths come from the .conf file.
        topology_file = params.get("topology_file")
        topo_path = None
        if topology_file:
            candidate = os.path.join(workspace_dir, topology_file)
            if os.path.exists(candidate):
                topo_path = candidate
            else:
                logger.warning("Topology file not found: %s", candidate)

        if not topo_path:
            return None, None, "NS3 仿真需要拓扑文件（-n），但未找到有效的拓扑文件。请先在 EDG 调节步骤生成拓扑，或手动指定拓扑路径。"

        # Intercept -c from extra_args and strip flags that the NS3 binary rejects.
        config_file = params.get("config_file")
        extra_args = params.get("extra_args", [])
        filtered_args = []
        conf_from_extra = None
        thread_from_extra = None
        skip_next = False
        for i, arg in enumerate(extra_args):
            if skip_next:
                skip_next = False
                continue
            if arg == "-c" and i + 1 < len(extra_args):
                conf_from_extra = extra_args[i + 1]
                skip_next = True
            elif arg == "-t" and i + 1 < len(extra_args):
                thread_from_extra = extra_args[i + 1]
                skip_next = True
            elif arg in ("-r", "-n", "-w"):
                # These are owned by the launcher; skip user-supplied duplicates.
                skip_next = (i + 1 < len(extra_args))
            else:
                filtered_args.append(arg)

        conf_source = config_file or conf_from_extra
        local_conf = _ensure_local_ns3_config(workspace_dir, conf_source)
        thread_count = thread_from_extra or str(params.get("thread", 8))

        cmd = [binary_path,
               "-t", str(thread_count),
               "-w", workload_path,
               "-n", topo_path,
               "-c", local_conf]
        # Note: any remaining filtered_args are dropped — NS3 binary's getopt
        # rejects unknown options and would abort with "illegal option".
    else:
        # SimAI_analytical (AstraParamParse): -w -r [-busbw, -nv, -nic, ...]
        extra_args = params.get("extra_args", [])
        has_result = any(a in ("-r", "--result") for a in extra_args)
        cmd = [binary_path, "-w", workload_path]
        if not has_result:
            cmd += ["-r", output_prefix]
        if extra_args:
            cmd.extend(extra_args)

    # Build environment
    run_env = os.environ.copy()
    user_env = params.get("env_vars", {})
    run_env.update(user_env)
    run_env["AS_OXC_RANKTABLE"] = ranktable_path

    # Auto-enable OXC integration when launching an OXC-capable binary.
    # AS_OXC_ENABLE=1 is required by OxcIntegration::fromEnvironment().
    # The frontend may also set this via env_vars; we only fill the default here.
    if "oxc" in binary_name.lower() and "AS_OXC_ENABLE" not in run_env:
        run_env["AS_OXC_ENABLE"] = "1"

    # Let NS3 binaries write directly to a unique output prefix so each
    # task produces its own file set without clobbering or rename races.
    if is_ns3:
        ns3_prefix = os.path.basename(output_prefix) + "_"
        run_env["AS_RESULT_PATH"] = "./" + ns3_prefix

    command_str = " ".join(cmd)
    logger.info("Launching: %s", command_str)

    try:
        from server.config import PROJECT_ROOT

        # NS3 binaries hardcode RESULT_PATH="./ncclFlowModel_" in their main
        # (see AstraSimNetwork.cc/AstraSimNetwork_oxc.cc) and write to CWD.
        # Run them in the workspace so each task's csv lands in its own dir
        # instead of all clobbering PROJECT_ROOT/ncclFlowModel_EndToEnd.csv.
        # Other binaries (analytical, OXC analytical) still need PROJECT_ROOT
        # because they reference ./results/ relatively.
        run_cwd = workspace_dir if is_ns3 else PROJECT_ROOT

        # Clean up stale ncclFlowModel_* files from previous runs.
        # If a previous run crashed before writing its EndToEnd CSV, the
        # stale file would be renamed as the new result (ghost results).
        if is_ns3:
            _cleanup_ns3_outputs(run_cwd)

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=run_env,
            cwd=run_cwd,
        )
    except Exception as e:
        return None, None, f"Failed to start process: {e}"

    pid = process.pid

    # Register in DB
    with get_db() as conn:
        cursor = conn.execute(
            "INSERT INTO processes "
            "(pid, session_token, username, workspace_dir, command, status, started_at) "
            "VALUES (?, ?, ?, ?, ?, 'running', ?)",
            (pid, session_token, username, workspace_dir, command_str, time.time()),
        )
        tracking_id = cursor.lastrowid

    # Initialize log buffer
    with _log_lock:
        _log_buffers[pid] = []

    # Start background log streaming
    log_path = os.path.join(workspace_dir, "logs", f"{pid}.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    # Derive output prefix for NS3 rename (NS3 binary hardcodes "ncclFlowModel_")
    rename_prefix = None
    if is_ns3:
        extra_args = params.get("extra_args", [])
        for i, a in enumerate(extra_args):
            if a in ("-r", "--result") and i + 1 < len(extra_args):
                rename_prefix = os.path.basename(extra_args[i + 1].rstrip("/-"))
                break
        if not rename_prefix:
            rename_prefix = os.path.basename(output_prefix.rstrip("/-"))

    thread = threading.Thread(
        target=_stream_output,
        args=(pid, process, log_path, workspace_dir if is_ns3 else None, rename_prefix),
        daemon=True,
    )
    thread.start()

    # Timeout watchdog
    timeout = params.get("timeout", 172800)  # 48 hours
    if timeout and timeout > 0:
        def _watchdog():
            time.sleep(timeout)
            if process.poll() is None:
                logger.warning("Process %d timed out after %ds, killing", pid, timeout)
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                update_status(pid, "timeout")

        watchdog = threading.Thread(target=_watchdog, daemon=True)
        watchdog.start()

    return tracking_id, pid, None


def update_status(pid, status, return_code=None, error_message=None):
    """Update process status in the database."""
    with get_db() as conn:
        conn.execute(
            "UPDATE processes SET status = ?, finished_at = ?, return_code = ?, error_message = ? "
            "WHERE pid = ? AND status = 'running'",
            (status, time.time(), return_code, error_message, pid),
        )


def kill_process(username, pid):
    """Kill a process owned by the given user. Returns (success, error)."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT username FROM processes WHERE pid = ? AND status = 'running'",
            (pid,),
        ).fetchone()
        if not row:
            return False, "Process not found or not running"
        if row["username"] != username:
            return False, "Permission denied: not your process"

    try:
        os.kill(pid, signal.SIGTERM)
        time.sleep(0.5)
        try:
            os.kill(pid, 0)
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        update_status(pid, "killed")
        return True, None
    except ProcessLookupError:
        update_status(pid, "dead", error_message="Process not found")
        return True, None
    except PermissionError:
        return False, "Permission denied"


def list_processes(username=None, session_token=None, status=None):
    """Query processes with optional filters."""
    with get_db() as conn:
        query = "SELECT * FROM processes WHERE 1=1"
        params = []
        if username:
            query += " AND username = ?"
            params.append(username)
        if session_token:
            query += " AND session_token = ?"
            params.append(session_token)
        if status:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY started_at DESC LIMIT 50"
        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]


def get_logs(username, pid):
    """Get log lines for a process owned by the user.
    Returns (logs, status, error).
    """
    with get_db() as conn:
        row = conn.execute(
            "SELECT username, status, workspace_dir FROM processes WHERE pid = ?",
            (pid,),
        ).fetchone()
        if not row:
            return None, None, "Process not found"
        if row["username"] != username:
            return None, None, "Permission denied"

    # Try in-memory buffer first
    with _log_lock:
        lines = list(_log_buffers.get(pid, []))

    # Fall back to log file if buffer is empty
    if not lines:
        log_path = os.path.join(row["workspace_dir"], "logs", f"{pid}.log")
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                lines = [l.rstrip("\n") for l in f.readlines()[-MAX_LOG_LINES:]]

    return lines, row["status"], None


def cleanup_dead_processes():
    """Mark processes as 'dead' if their PID no longer exists."""
    with get_db() as conn:
        running = conn.execute(
            "SELECT pid FROM processes WHERE status = 'running'"
        ).fetchall()
        cleaned = 0
        for row in running:
            pid = row["pid"]
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                conn.execute(
                    "UPDATE processes SET status = 'dead', finished_at = ? "
                    "WHERE pid = ? AND status = 'running'",
                    (time.time(), pid),
                )
                cleaned += 1
                # Clean up log buffer
                with _log_lock:
                    _log_buffers.pop(pid, None)
        if cleaned:
            logger.info("Cleaned up %d dead process(es)", cleaned)
        return cleaned
