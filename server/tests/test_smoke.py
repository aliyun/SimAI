"""Smoke tests for all simulation modes (T001).

Runs each binary with a minimal 2-layer workload and verifies exit 0 + output CSV.
Skips binaries that aren't built.
"""

import os
import subprocess
import sys
import tempfile
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import server.config as config

BIN_DIR = config.BIN_DIR
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
NS3_CONF = os.path.join(PROJECT_ROOT, "astra-sim-alibabacloud", "inputs", "config", "SimAI.conf")
LOCAL_NS3_CONF = "/tmp/simai_local.conf"

WORKLOAD = """HYBRID_TRANSFORMER_FWD_IN_BCKWD model_parallel_NPU_group: 8 ep: 1 pp: 1 vpp: 1 ga: 1 all_gpus: 8 checkpoints: 0 checkpoint_initiates: 0
2
layer_1	-1	10000	ALLREDUCE	1048576	1	NONE	0	1	NONE	0	1
layer_2	-1	10000	ALLREDUCE	2097152	1	NONE	0	1	NONE	0	1
"""


def _fix_ns3_conf():
    """Create local NS3 config with fixed paths if not already done."""
    if os.path.exists(LOCAL_NS3_CONF):
        return LOCAL_NS3_CONF
    os.makedirs("/tmp/astra-sim/simulation", exist_ok=True)
    for name in ["flow1.txt", "trace1.txt", "llama_hpn7_mix.tr", "llama_hpn7_fct.txt",
                 "llama_hpn7_pfc.txt", "llama_hpn7_qlen.txt", "llama_hpn7_bw.txt",
                 "llama_hpn7_rate.txt", "llama_hpn7_cnp.txt"]:
        p = os.path.join("/tmp/astra-sim/simulation", name)
        if not os.path.exists(p):
            open(p, "w").close()
    with open(NS3_CONF) as f:
        content = f.read()
    content = content.replace("/etc/astra-sim/simulation/", "/tmp/astra-sim/simulation/")
    with open(LOCAL_NS3_CONF, "w") as f:
        f.write(content)
    return LOCAL_NS3_CONF


def _has_binary(name):
    path = os.path.join(BIN_DIR, name)
    if not os.path.islink(path) and not os.path.isfile(path):
        return None
    real = os.path.realpath(path) if os.path.islink(path) else path
    if not os.path.isfile(real):
        return None
    return real


class TestSmokeAnalytical(unittest.TestCase):
    """Smoke test: SimAI_analytical."""

    def setUp(self):
        self.binary = _has_binary("SimAI_analytical")
        if not self.binary:
            self.skipTest("SimAI_analytical not built")
        self.workload = os.path.join(tempfile.gettempdir(), "smoke_wl.txt")
        with open(self.workload, "w") as f:
            f.write(WORKLOAD)

    def test_analytical_runs_and_produces_csv(self):
        result = subprocess.run(
            [self.binary, "-w", self.workload, "-g", "8", "-g_p_s", "8", "-r", "smoke_"],
            capture_output=True, text=True, timeout=30, cwd=PROJECT_ROOT,
        )
        self.assertEqual(result.returncode, 0, f"stderr: {result.stderr[:500]}")
        csv_path = os.path.join(RESULTS_DIR, "smoke_EndToEnd.csv")
        self.assertTrue(os.path.exists(csv_path), f"Missing {csv_path}")
        with open(csv_path) as f:
            header = f.readline()
        self.assertIn("Expose TP comm", header)


class TestSmokeAnalyticalOxc(unittest.TestCase):
    """Smoke test: SimAI_analytical_oxc."""

    def setUp(self):
        self.binary = _has_binary("SimAI_analytical_oxc")
        if not self.binary:
            self.skipTest("SimAI_analytical_oxc not built")
        self.workload = os.path.join(tempfile.gettempdir(), "smoke_wl_oxc.txt")
        with open(self.workload, "w") as f:
            f.write(WORKLOAD)

    def test_analytical_oxc_runs(self):
        result = subprocess.run(
            [self.binary, "-w", self.workload, "-g", "8", "-g_p_s", "8", "-r", "smoke_oxc_"],
            capture_output=True, text=True, timeout=30, cwd=PROJECT_ROOT,
        )
        self.assertEqual(result.returncode, 0, f"stderr: {result.stderr[:500]}")
        csv_path = os.path.join(RESULTS_DIR, "smoke_oxc_EndToEnd.csv")
        self.assertTrue(os.path.exists(csv_path), f"Missing {csv_path}")


class TestSmokeNS3(unittest.TestCase):
    """Smoke test: SimAI_simulator (NS3)."""

    def setUp(self):
        self.binary = _has_binary("SimAI_simulator")
        if not self.binary:
            self.skipTest("SimAI_simulator not built")
        self.conf = _fix_ns3_conf()
        self.workload = os.path.join(tempfile.gettempdir(), "smoke_wl_ns3.txt")
        with open(self.workload, "w") as f:
            f.write(WORKLOAD)
        # Small topology for 8 NPUs
        self.topo = os.path.join(PROJECT_ROOT, "Spectrum-X_8g_8gps_100Gbps_A100")
        if not os.path.exists(self.topo):
            self.skipTest(f"Topology not found: {self.topo}")

    def test_ns3_runs_and_produces_csv(self):
        result = subprocess.run(
            [self.binary, "-t", "1", "-w", self.workload, "-n", self.topo, "-c", self.conf],
            capture_output=True, text=True, timeout=60, cwd=PROJECT_ROOT,
            env={**os.environ, "AS_LOG_LEVEL": "WARN"},
        )
        self.assertEqual(result.returncode, 0, f"stderr: {result.stderr[:500]}")
        csv_path = os.path.join(PROJECT_ROOT, "ncclFlowModel_EndToEnd.csv")
        self.assertTrue(os.path.exists(csv_path), f"Missing {csv_path}")


class TestSmokeNS3Oxc(unittest.TestCase):
    """Smoke test: SimAI_simulator_oxc (NS3 + OXC)."""

    def setUp(self):
        self.binary = _has_binary("SimAI_simulator_oxc")
        if not self.binary:
            self.skipTest("SimAI_simulator_oxc not built")
        self.conf = _fix_ns3_conf()
        self.workload = os.path.join(tempfile.gettempdir(), "smoke_wl_ns3oxc.txt")
        with open(self.workload, "w") as f:
            f.write(WORKLOAD)
        self.topo = os.path.join(PROJECT_ROOT, "Spectrum-X_8g_8gps_100Gbps_A100")
        if not os.path.exists(self.topo):
            self.skipTest(f"Topology not found: {self.topo}")

    def test_ns3_oxc_runs(self):
        result = subprocess.run(
            [self.binary, "-t", "1", "-w", self.workload, "-n", self.topo, "-c", self.conf],
            capture_output=True, text=True, timeout=60, cwd=PROJECT_ROOT,
            env={**os.environ, "AS_LOG_LEVEL": "WARN"},
        )
        self.assertEqual(result.returncode, 0, f"stderr: {result.stderr[:500]}")
