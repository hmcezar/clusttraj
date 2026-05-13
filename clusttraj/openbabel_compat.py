"""Open Babel import helpers."""

OPENBABEL_INSTALL_MESSAGE = (
    "Open Babel is required to read and write molecular structures. "
    "Install it with conda using 'conda install -c conda-forge openbabel' "
    "or, in pip-only environments, with 'pip install \"clusttraj[openbabel]\"'. "
    "Avoid mixing conda Open Babel and openbabel-wheel in the same environment; "
    "if you see linker errors, remove one provider and reinstall Open Babel from "
    "the package manager used by the environment."
)

try:
    from openbabel import openbabel, pybel
except ImportError as exc:
    raise ImportError(OPENBABEL_INSTALL_MESSAGE) from exc
