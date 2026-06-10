"""
Smoke test to verify pytest infrastructure works.

Run: python3 -m pytest tests/unit/test_smoke.py -v
"""
import os
import sys

import pytest


@pytest.mark.unit
class TestPytestInfrastructure:
    """Verify that pytest fixtures and markers work correctly."""

    def test_project_root_exists(self, project_root):
        assert os.path.isdir(project_root)
        assert os.path.isfile(os.path.join(project_root, "CLAUDE.md"))

    def test_gui_dir_exists(self, gui_dir):
        assert os.path.isdir(gui_dir)
        assert os.path.isfile(os.path.join(gui_dir, "app.py"))

    def test_sample_ranktable_structure(self, sample_ranktable):
        assert sample_ranktable["version"] == "2.0"
        assert sample_ranktable["rank_count"] == 16
        assert len(sample_ranktable["rank_list"]) == 16
        # Verify first rank
        rank_0 = sample_ranktable["rank_list"][0]
        assert rank_0["rank_id"] == 0
        assert len(rank_0["level_list"]) == 1

    def test_sample_rank_rack_map(self, sample_rank_rack_map):
        assert len(sample_rank_rack_map) == 16
        assert sample_rank_rack_map["0"] == "rack_0"
        assert sample_rank_rack_map["8"] == "rack_1"

    def test_temp_workspace_is_writable(self, temp_workspace):
        test_file = temp_workspace / "test.txt"
        test_file.write_text("hello")
        assert test_file.read_text() == "hello"

    def test_ranktable_json_file(self, ranktable_json_file):
        import json
        with open(ranktable_json_file) as f:
            data = json.load(f)
        assert data["rank_count"] == 16

    def test_gui_modules_importable(self, gui_dir):
        """Verify GUI utility modules can be imported."""
        sys.path.insert(0, gui_dir)
        try:
            # workload_generator has no heavy deps, should always import
            from utils import workload_generator
            assert hasattr(workload_generator, "generate_workload_content")
        except ImportError as e:
            pytest.skip(f"GUI dependencies not installed: {e}")
        finally:
            if gui_dir in sys.path:
                sys.path.remove(gui_dir)
