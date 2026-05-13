import importlib
import sys
import pytest


def test_openbabel_missing_raises_import_error():
    """Simulate missing openbabel and verify the user-facing error message."""
    compat = importlib.import_module("clusttraj.openbabel_compat")
    saved = sys.modules.get("openbabel")
    try:
        sys.modules["openbabel"] = None
        with pytest.raises(ImportError, match="Open Babel is required"):
            importlib.reload(compat)
    finally:
        sys.modules["openbabel"] = saved
        if saved is not None:
            importlib.reload(compat)
