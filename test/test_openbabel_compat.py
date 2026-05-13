import importlib
import sys
import pytest
import clusttraj.openbabel_compat


def test_openbabel_missing_raises_import_error():
    """Simulate missing openbabel and verify the user-facing error message."""
    saved = sys.modules.get("openbabel")
    try:
        sys.modules["openbabel"] = None
        with pytest.raises(ImportError, match="Open Babel is required"):
            importlib.reload(clusttraj.openbabel_compat)
    finally:
        sys.modules["openbabel"] = saved
        if saved is not None:
            importlib.reload(clusttraj.openbabel_compat)
