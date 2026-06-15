"""
Tests for the model registry module (workload_generator/registry.py).

Covers: register_model, lookup, get_registered_frames, is_registered,
clear_registry, duplicate-registration rejection, and KeyError messages.
"""

import pytest
import sys
import os

# Ensure the aicb package root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from workload_generator.registry import (
    ModelEntry,
    model_registry,
    register_model,
    lookup,
    get_registered_frames,
    is_registered,
    clear_registry,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class DummyModel:
    """A trivial model stand-in for registry tests."""
    def __init__(self, args):
        self.args = args


class DummyWorkload:
    """A trivial workload-generator stand-in for registry tests."""
    def __init__(self, args, model):
        self.args = args
        self.model = model


@pytest.fixture(autouse=True)
def _clean_registry():
    """Automatically clear the registry before every test so tests are isolated."""
    clear_registry()
    yield
    clear_registry()


# ---------------------------------------------------------------------------
# register_model
# ---------------------------------------------------------------------------

class TestRegisterModel:
    def test_register_single_model(self):
        register_model("TestFrame", DummyModel, DummyWorkload, "A test frame")
        assert "TestFrame" in model_registry
        entry = model_registry["TestFrame"]
        assert isinstance(entry, ModelEntry)
        assert entry.name == "TestFrame"
        assert entry.model_cls is DummyModel
        assert entry.wl_cls is DummyWorkload
        assert entry.description == "A test frame"

    def test_register_model_with_none_model_cls(self):
        """collective_test uses model_cls=None."""
        register_model("collective_test", None, DummyWorkload)
        entry = model_registry["collective_test"]
        assert entry.model_cls is None
        assert entry.wl_cls is DummyWorkload

    def test_register_duplicate_raises_valueerror(self):
        register_model("Dup", DummyModel, DummyWorkload)
        with pytest.raises(ValueError, match="already registered"):
            register_model("Dup", DummyModel, DummyWorkload)

    def test_register_multiple_models(self):
        register_model("A", DummyModel, DummyWorkload)
        register_model("B", DummyModel, DummyWorkload)
        register_model("C", None, DummyWorkload)
        assert len(model_registry) == 3
        assert get_registered_frames() == ["A", "B", "C"]


# ---------------------------------------------------------------------------
# lookup
# ---------------------------------------------------------------------------

class TestLookup:
    def test_lookup_existing_model(self):
        register_model("Known", DummyModel, DummyWorkload, "Known model")
        entry = lookup("Known")
        assert entry.name == "Known"
        assert entry.model_cls is DummyModel
        assert entry.wl_cls is DummyWorkload

    def test_lookup_unknown_raises_keyerror_with_helpful_message(self):
        register_model("OnlyOne", DummyModel, DummyWorkload)
        with pytest.raises(KeyError, match="Unknown frame 'Missing'"):
            lookup("Missing")

    def test_lookup_unknown_message_lists_available(self):
        register_model("Alpha", DummyModel, DummyWorkload)
        register_model("Beta", DummyModel, DummyWorkload)
        try:
            lookup("Gamma")
        except KeyError as e:
            msg = str(e)
            assert "Alpha" in msg
            assert "Beta" in msg
            assert "Gamma" in msg

    def test_lookup_with_empty_registry(self):
        with pytest.raises(KeyError, match="none registered"):
            lookup("Anything")


# ---------------------------------------------------------------------------
# get_registered_frames / is_registered
# ---------------------------------------------------------------------------

class TestQueryFunctions:
    def test_get_registered_frames_empty(self):
        assert get_registered_frames() == []

    def test_get_registered_frames_sorted(self):
        register_model("Zebra", DummyModel, DummyWorkload)
        register_model("Alpha", DummyModel, DummyWorkload)
        register_model("Beta", None, DummyWorkload)
        assert get_registered_frames() == ["Alpha", "Beta", "Zebra"]

    def test_is_registered_true(self):
        register_model("Present", DummyModel, DummyWorkload)
        assert is_registered("Present") is True

    def test_is_registered_false(self):
        assert is_registered("Absent") is False


# ---------------------------------------------------------------------------
# clear_registry
# ---------------------------------------------------------------------------

class TestClearRegistry:
    def test_clear_removes_all_entries(self):
        register_model("X", DummyModel, DummyWorkload)
        register_model("Y", DummyModel, DummyWorkload)
        clear_registry()
        assert len(model_registry) == 0
        assert get_registered_frames() == []


# ---------------------------------------------------------------------------
# Integration: simulate the aicb.py dispatch flow
# ---------------------------------------------------------------------------

class TestDispatchFlow:
    def test_full_dispatch_flow(self):
        """Simulate what aicb.py does: register -> lookup -> instantiate."""
        register_model("Megatron", DummyModel, DummyWorkload, "Megatron-LM")

        # Simulate argument parsing
        frame = "Megatron"

        # Simulate aicb.py dispatch
        entry = lookup(frame)
        assert entry.model_cls is DummyModel
        assert entry.wl_cls is DummyWorkload

        # Instantiate
        class FakeArgs:
            pass
        args = FakeArgs()
        model = entry.model_cls(args)
        wl = entry.wl_cls(args, model)

        assert isinstance(model, DummyModel)
        assert isinstance(wl, DummyWorkload)
        assert model.args is args
        assert wl.args is args
        assert wl.model is model

    def test_collective_test_dispatch(self):
        """collective_test has no model class."""
        register_model("collective_test", None, DummyWorkload, "Collective test")

        entry = lookup("collective_test")
        assert entry.model_cls is None
        assert entry.wl_cls is DummyWorkload
