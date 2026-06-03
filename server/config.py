import os

# Multi-user mode toggle
MULTIUSER_ENABLED = os.environ.get("SIMAI_MULTIUSER", "0") == "1"

# Server
SERVER_HOST = os.environ.get("SIMAI_SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.environ.get("SIMAI_SERVER_PORT", "5000"))
SECRET_KEY = os.environ.get("SIMAI_SECRET_KEY", "simai-dev-key-change-in-production")

# Paths
SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SERVER_DIR)
TOPOLOGY_DIR = os.environ.get("SIMAI_TOPOLOGY_DIR", PROJECT_ROOT)
WORKSPACE_ROOT = os.path.join(SERVER_DIR, "workspaces")
BIN_DIR = os.path.join(PROJECT_ROOT, "bin")
AUTH_CONFIG_PATH = os.path.join(SERVER_DIR, "auth_config.yaml")

# Workspace
WORKSPACE_TTL_SECONDS = 24 * 3600  # 24 hours

# Process limits
MAX_PROCESSES_PER_USER = 3
MAX_TOTAL_PROCESSES = 10

# Allowed binary names (whitelist for security)
ALLOWED_BINARIES = {
    "SimAI_oxc",
    "SimAI_analytical",
    "SimAI_analytical_oxc",
    "SimAI_simulator",
    "SimAI_simulator_oxc",
}
